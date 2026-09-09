package infer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"io/fs"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// ErrBusy is returned when too many requests are already waiting for the
// inference slot. Shedding load is better than queueing without bound on a
// 2 vCPU box, where a backlog turns into swap pressure and 30s page loads.
var ErrBusy = errors.New("inference queue is full")

// Grade is an APTOS diabetic-retinopathy severity level, 0 (none) to 4.
type Grade int

const NumGrades = 5

// Label pairs a grade with its display names.
type Label struct {
	Grade int    `json:"grade"`
	EN    string `json:"en"`
	ID    string `json:"id"`
}

// LabelSet is model/labels.json.
type LabelSet struct {
	ModelID          string  `json:"model_id"`
	ModelFile        string  `json:"model_file"`
	ModelSHA256      string  `json:"model_sha256"`
	OrderingVerified bool    `json:"ordering_verified"`
	OrderingNote     string  `json:"ordering_note"`
	Labels           []Label `json:"labels"`
}

// Result is one prediction.
type Result struct {
	Grade      int
	LabelEN    string
	LabelID    string
	Confidence float32
	// Probs holds all NumGrades softmax outputs, not just the winner, so the
	// UI can show the full distribution and analytics can flag near-ties.
	Probs []float32
	// Heatmap is the attention-rollout saliency grid (14x14 for ViT-B/16),
	// row-major and min-max normalized to [0,1]. Rendering is overlay.go's job.
	Heatmap     []float32
	HeatmapDim  int
	InferenceMS int
}

// Config configures an Engine.
type Config struct {
	ModelPath      string
	LabelsPath     string
	PreprocessPath string
	// GatePath is model/gate.json. A missing file disables the non-fundus
	// gate rather than failing startup, so a deployment upgraded ahead of its
	// calibration run still boots.
	GatePath       string
	ORTLibPath     string
	IntraOpThreads int
	InterOpThreads int
	MaxQueueDepth  int
}

// Engine runs the exported ViT. It is safe for concurrent use, but deliberately
// serializes the actual forward pass — see sem.
type Engine struct {
	session *ort.DynamicAdvancedSession
	pre     *PreprocessConfig
	labels  *LabelSet

	// gate holds the calibrated thresholds and never changes: it is measured
	// evidence from model/gate.py, not a preference. Whether the gate runs is
	// a separate, operator-owned decision, so it lives in gateEnabled — an
	// atomic because every upload reads it and an administrator can flip it
	// from another goroutine at any moment.
	gate        *GateConfig
	gateEnabled atomic.Bool

	modelSHA256 string

	// sem has capacity 1: the target VPS has 2 vCPU, and ONNX Runtime is
	// already configured to use both for a single op. Running two forward
	// passes concurrently would halve each one's threads and thrash cache.
	sem chan struct{}
	// waiting counts goroutines blocked on sem, so the handler can shed load.
	waiting       atomic.Int32
	maxQueueDepth int

	// inflightHook, when set, is called with +1 on entering the serialized
	// section and -1 on leaving it. Tests use it to assert that the semaphore
	// really does admit only one forward pass at a time. Set before serving.
	inflightHook func(delta int)

	closeOnce sync.Once
}

// ortInit guards the process-global ONNX Runtime environment.
var ortInit struct {
	once sync.Once
	err  error
}

func initORT(libPath string) error {
	ortInit.once.Do(func() {
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		if err := ort.InitializeEnvironment(); err != nil {
			ortInit.err = fmt.Errorf("initialize onnxruntime (lib=%q): %w", libPath, err)
		}
	})
	return ortInit.err
}

// New loads the model and its sidecar metadata.
func New(cfg Config) (*Engine, error) {
	pre, err := LoadPreprocessConfig(cfg.PreprocessPath)
	if err != nil {
		return nil, err
	}

	rawLabels, err := os.ReadFile(cfg.LabelsPath)
	if err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	var labels LabelSet
	if err := json.Unmarshal(rawLabels, &labels); err != nil {
		return nil, fmt.Errorf("parse labels: %w", err)
	}
	if len(labels.Labels) != NumGrades {
		return nil, fmt.Errorf("labels.json has %d labels, expected %d", len(labels.Labels), NumGrades)
	}

	gate, err := LoadGateConfig(cfg.GatePath)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		log.Printf("infer: %s missing, non-fundus gate DISABLED "+
			"(run model/gate.py to calibrate it)", cfg.GatePath)
		gate = nil
	case err != nil:
		return nil, err
	case !gate.Enabled:
		log.Println("infer: non-fundus gate disabled by config")
	}

	sum, err := fileSHA256(cfg.ModelPath)
	if err != nil {
		return nil, err
	}

	if err := initORT(cfg.ORTLibPath); err != nil {
		return nil, err
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()
	if n := cfg.IntraOpThreads; n > 0 {
		if err := opts.SetIntraOpNumThreads(n); err != nil {
			return nil, fmt.Errorf("set intra-op threads: %w", err)
		}
	}
	if n := cfg.InterOpThreads; n > 0 {
		if err := opts.SetInterOpNumThreads(n); err != nil {
			return nil, fmt.Errorf("set inter-op threads: %w", err)
		}
	}

	session, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{"pixel_values"},
		[]string{"logits", "attentions"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("load model %q: %w", cfg.ModelPath, err)
	}

	depth := cfg.MaxQueueDepth
	if depth <= 0 {
		depth = 4
	}

	e := &Engine{
		session:       session,
		pre:           pre,
		labels:        &labels,
		gate:          gate,
		modelSHA256:   sum,
		sem:           make(chan struct{}, 1),
		maxQueueDepth: depth,
	}
	// gate.json's own flag is only the boot default; a stored admin preference
	// overrides it via SetGateEnabled once the database is reachable.
	e.gateEnabled.Store(gate != nil && gate.Enabled)
	return e, nil
}

// Close releases the ONNX session.
func (e *Engine) Close() {
	e.closeOnce.Do(func() {
		if e.session != nil {
			e.session.Destroy()
		}
	})
}

// ModelID reports which checkpoint is loaded; stored on every scan row.
func (e *Engine) ModelID() string { return e.labels.ModelID }

// ModelSHA256 is the hash of the served .onnx file.
func (e *Engine) ModelSHA256() string { return e.modelSHA256 }

// OrderingVerified reports whether model/eval.py has confirmed that output
// index i really means DR grade i. The UI must warn when this is false.
func (e *Engine) OrderingVerified() bool { return e.labels.OrderingVerified }

// Labels exposes the display names.
func (e *Engine) Labels() []Label { return e.labels.Labels }

// GateAvailable reports whether calibrated thresholds were loaded. Without
// them there is nothing to enable, so the UI must not offer the choice.
func (e *Engine) GateAvailable() bool { return e.gate != nil }

// GateEnabled reports whether the gate is currently running.
func (e *Engine) GateEnabled() bool { return e.gateEnabled.Load() }

// GateCalibration returns what the thresholds were measured against, or nil.
func (e *Engine) GateCalibration() *Calibration {
	if e.gate == nil {
		return nil
	}
	return e.gate.Calibration
}

// SetGateEnabled turns the gate on or off for every subsequent upload, with no
// restart. It reports whether the request was honoured: enabling is refused
// when no calibration is loaded, because there would be no thresholds to apply
// and the caller must be able to tell that apart from success.
func (e *Engine) SetGateEnabled(on bool) bool {
	if on && !e.GateAvailable() {
		return false
	}
	e.gateEnabled.Store(on)
	return true
}

// CheckImage runs the gate's image heuristics.
//
// It is exposed separately from Predict so callers can reject an upload before
// it reaches the serialized forward pass: a junk image should never occupy the
// single inference slot, and on a hard rejection nothing needs to be stored.
func (e *Engine) CheckImage(img image.Image) GateReport {
	if !e.gateEnabled.Load() {
		return GateReport{Passed: true}
	}
	return e.gate.CheckImage(img)
}

// Predict runs one image through the model.
//
// It blocks until the single inference slot is free, honouring ctx so a client
// that disconnects releases its place rather than holding the slot.
func (e *Engine) Predict(ctx context.Context, img image.Image) (*Result, error) {
	if int(e.waiting.Load()) >= e.maxQueueDepth {
		return nil, ErrBusy
	}
	e.waiting.Add(1)
	defer e.waiting.Add(-1)

	select {
	case e.sem <- struct{}{}:
		defer func() { <-e.sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	if e.inflightHook != nil {
		e.inflightHook(1)
		defer e.inflightHook(-1)
	}

	// Check again after acquiring: a long queue may have drained into a
	// cancelled context while we waited.
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	start := time.Now()

	pixels := e.pre.Tensor(img)
	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(e.pre.Height), int64(e.pre.Width)), pixels)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// nil outputs are allocated by the runtime and returned in place.
	outputs := []ort.Value{nil, nil}
	if err := e.session.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return nil, fmt.Errorf("run inference: %w", err)
	}
	for _, v := range outputs {
		if v != nil {
			defer v.Destroy()
		}
	}

	logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected logits type %T", outputs[0])
	}
	logits := logitsTensor.GetData()
	if len(logits) != NumGrades {
		return nil, fmt.Errorf("model produced %d logits, expected %d", len(logits), NumGrades)
	}

	// Second-tier gate. It lives inside Predict rather than beside the caller's
	// CheckImage call so that it cannot be skipped: any future path to the
	// model passes through here. Reads the same flag as CheckImage, so an
	// administrator switching the gate off disables both tiers at once.
	if e.gateEnabled.Load() {
		if rep := e.gate.CheckLogits(logits); !rep.Passed {
			return nil, fmt.Errorf("%w (%s energy=%.3f)",
				ErrNotFundus, rep.Reason, rep.Metrics["energy"])
		}
	}

	probs := Softmax(logits)
	grade := argmax(probs)

	res := &Result{
		Grade:      grade,
		LabelEN:    e.labels.Labels[grade].EN,
		LabelID:    e.labels.Labels[grade].ID,
		Confidence: probs[grade],
		Probs:      probs,
	}

	// Attentions are optional: a model exported without them still classifies,
	// it just cannot explain itself.
	if attnTensor, ok := outputs[1].(*ort.Tensor[float32]); ok {
		shape := attnTensor.GetShape()
		grid, dim, err := AttentionRollout(attnTensor.GetData(), shape)
		if err == nil {
			res.Heatmap, res.HeatmapDim = grid, dim
		}
	}

	res.InferenceMS = int(time.Since(start).Milliseconds())
	return res, nil
}

// Softmax converts logits to probabilities, shifted by the max for stability.
func Softmax(logits []float32) []float32 {
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	out := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		e := float32(exp(float64(v - maxLogit)))
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open model: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("hash model: %w", err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
