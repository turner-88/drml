package infer

import (
	"context"
	"errors"
	"image"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// newTestEngine builds an Engine from the exported artifacts, skipping when
// they are absent (CI without the model) or when onnxruntime is not installed.
func newTestEngine(t *testing.T) *Engine {
	t.Helper()
	// An empty GatePath is the "no gate.json" case, which must degrade to a
	// disabled gate rather than failing startup.
	return newTestEngineWithGate(t, "")
}

func newTestEngineWithGate(t *testing.T, gatePath string) *Engine {
	t.Helper()
	return newTestEngineWithConfig(t, func(c *Config) { c.GatePath = gatePath })
}

// newTestEngineWithConfig starts an engine on the served artifact, letting the
// test adjust the config first. DRML_TEST_MODEL names an alternative .onnx
// under model/ (vit_dr.onnx, for instance, to check parity without int8
// noise).
func newTestEngineWithConfig(t *testing.T, adjust func(*Config)) *Engine {
	t.Helper()
	dir := modelDir(t)

	modelFile := os.Getenv("DRML_TEST_MODEL")
	if modelFile == "" {
		modelFile = "vit_dr_int8.onnx"
	}
	modelPath := modelFile
	if !filepath.IsAbs(modelPath) {
		modelPath = filepath.Join(dir, modelFile)
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model missing (run model/export.py): %v", err)
	}

	cfg := Config{
		ModelPath:      modelPath,
		LabelsPath:     filepath.Join(dir, "labels.json"),
		PreprocessPath: filepath.Join(dir, "preprocess.json"),
		ORTLibPath:     ortLibPathForTest(),
		IntraOpThreads: 2,
		InterOpThreads: 1,
		MaxQueueDepth:  4,
	}
	if adjust != nil {
		adjust(&cfg)
	}
	eng, err := New(cfg)
	if err != nil {
		t.Skipf("cannot start engine (is onnxruntime installed?): %v", err)
	}
	t.Cleanup(eng.Close)
	return eng
}

// isFP32Model reports whether the test engine is running the unquantized
// graph, where parity tolerances can be tight.
func isFP32Model() bool { return os.Getenv("DRML_TEST_MODEL") == "vit_dr.onnx" }

// runRaw feeds a ready-made tensor through the session and returns the raw
// attention outputs, bypassing preprocessing. grads is nil when the graph has
// no attn_grads output.
func runRaw(t *testing.T, eng *Engine, pixelValues []float32) (attn, grads []float32, shape []int64) {
	t.Helper()
	in, err := ort.NewTensor(ort.NewShape(1, 3, int64(eng.pre.Height), int64(eng.pre.Width)), pixelValues)
	if err != nil {
		t.Fatalf("input tensor: %v", err)
	}
	defer in.Destroy()
	outputs := make([]ort.Value, len(eng.outputNames))
	if err := eng.session.Run([]ort.Value{in}, outputs); err != nil {
		t.Fatalf("run: %v", err)
	}
	defer func() {
		for _, v := range outputs {
			if v != nil {
				v.Destroy()
			}
		}
	}()
	if eng.attnIdx < 0 {
		t.Skip("graph has no attentions output")
	}
	at := outputs[eng.attnIdx].(*ort.Tensor[float32])
	shape = at.GetShape()
	attn = append([]float32(nil), at.GetData()...)
	if eng.gradsIdx >= 0 {
		grads = append([]float32(nil), outputs[eng.gradsIdx].(*ort.Tensor[float32]).GetData()...)
	}
	return attn, grads, shape
}

func ortLibPathForTest() string {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		return p
	}
	for _, p := range []string{
		"/opt/homebrew/lib/libonnxruntime.dylib",
		"/usr/local/lib/libonnxruntime.dylib",
		"/usr/local/lib/libonnxruntime.so",
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// TestLogitsParity is the decisive preprocessing check: it runs the fixture
// image through the full Go path and compares against PyTorch's logits. Small
// per-pixel rounding differences are acceptable only if they do not move the
// model's output, which is what this measures.
func TestLogitsParity(t *testing.T) {
	g, dir := loadGolden(t)
	eng := newTestEngine(t)

	f, err := os.Open(filepath.Join(dir, "testdata", g.Image))
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	res, err := eng.Predict(context.Background(), img)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}

	// Compare probability distributions rather than raw logits: the served
	// model is int8-quantized, so its logits legitimately differ from the fp32
	// PyTorch reference, but the resulting probabilities must still agree.
	want := Softmax(g.Logits)

	var maxDiff float64
	for i := range want {
		d := math.Abs(float64(res.Probs[i] - want[i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("probs go=%v", res.Probs)
	t.Logf("probs py=%v", want)
	t.Logf("max|prob diff| = %.6g; grade=%d (%s) confidence=%.4f in %dms",
		maxDiff, res.Grade, res.LabelEN, res.Confidence, res.InferenceMS)

	// Tolerance covers int8 quantization plus one LSB of resize rounding.
	const tol = 0.05
	if maxDiff > tol {
		t.Errorf("probability mismatch vs PyTorch: max|diff| = %.6g, want <= %g", maxDiff, tol)
	}
	if argmax(want) != res.Grade {
		t.Errorf("argmax disagrees: go=%d python=%d", res.Grade, argmax(want))
	}
}

func TestPredictProducesWellFormedResult(t *testing.T) {
	eng := newTestEngine(t)

	img := image.NewRGBA(image.Rect(0, 0, 300, 240))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 90, 40, 20, 255
	}

	res, err := eng.Predict(context.Background(), img)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}

	if len(res.Probs) != NumGrades {
		t.Fatalf("got %d probs, want %d", len(res.Probs), NumGrades)
	}
	var sum float32
	for _, p := range res.Probs {
		if p < 0 || p > 1 {
			t.Errorf("probability out of range: %v", p)
		}
		sum += p
	}
	if math.Abs(float64(sum)-1) > 1e-4 {
		t.Errorf("probabilities sum to %v, want 1", sum)
	}
	if res.Grade < 0 || res.Grade >= NumGrades {
		t.Errorf("grade %d out of range", res.Grade)
	}
	if res.Confidence != res.Probs[res.Grade] {
		t.Errorf("confidence %v != probs[%d] = %v", res.Confidence, res.Grade, res.Probs[res.Grade])
	}

	// The saliency map must be present and shaped for a 224px ViT-B/16.
	if res.Saliency == nil {
		t.Fatal("no saliency map produced")
	}
	if res.Saliency.Side != 14 {
		t.Errorf("saliency side = %d, want 14", res.Saliency.Side)
	}
	if len(res.Saliency.Grid) != res.Saliency.Side*res.Saliency.Side {
		t.Errorf("saliency has %d values, want %d",
			len(res.Saliency.Grid), res.Saliency.Side*res.Saliency.Side)
	}
	for _, v := range res.Saliency.Grid {
		if v < 0 || v > 1 {
			t.Fatalf("saliency value %v outside [0,1]", v)
		}
	}
	// Uniform attention scores 1.0, so anything below that is a broken
	// statistic rather than a flat map.
	if res.Saliency.Concentration < 1 {
		t.Errorf("concentration = %v, want >= 1", res.Saliency.Concentration)
	}
	if res.Saliency.Method != eng.ExplainMethod() {
		t.Errorf("saliency method %q != engine method %q", res.Saliency.Method, eng.ExplainMethod())
	}
	if eng.gradsIdx >= 0 && res.Saliency.Method != ExplainGradRelevance {
		t.Errorf("graph has gradients but method = %q", res.Saliency.Method)
	}
}

// TestRelevanceParity is the decisive saliency check: the Go GradRelevance on
// the graph's own tensors must reproduce the Python reference recorded in
// golden.json. Both are compared after normalization, which is what gets
// rendered.
func TestRelevanceParity(t *testing.T) {
	g, _ := loadGolden(t)
	if g.Relevance == nil {
		t.Skip("golden.json predates attn_grads (re-run model/export.py)")
	}
	eng := newTestEngine(t)
	if eng.gradsIdx < 0 {
		t.Skip("served graph has no attn_grads output")
	}

	// int8 weights perturb the gradient more than the logits, and after
	// normalization one cell can land a few tenths away from the reference, so
	// the served artifact is judged on its mean error and correlation. Measured
	// on export: int8-vs-fp32 relevance cosine >= 0.96 over the fundus
	// samples. The fp32 graph must match to float precision.
	maxTol, meanTol, minCorr := 0.5, 0.05, 0.95
	if isFP32Model() {
		maxTol, meanTol, minCorr = 1e-3, 1e-4, 0.9999
	}

	check := func(name string, pixelValues, want []float32) {
		attn, grads, shape := runRaw(t, eng, pixelValues)
		got, _, err := gradRelevanceRaw(attn, grads, shape)
		if err != nil {
			t.Fatalf("%s: relevance: %v", name, err)
		}
		if len(got) != len(want) {
			t.Fatalf("%s: got %d cells, golden has %d", name, len(got), len(want))
		}
		want = append([]float32(nil), want...)
		normalizeSaliency(got, nil)
		normalizeSaliency(want, nil)

		var maxDiff, sumDiff float64
		for i := range got {
			d := math.Abs(float64(got[i] - want[i]))
			sumDiff += d
			if d > maxDiff {
				maxDiff = d
			}
		}
		meanDiff := sumDiff / float64(len(got))
		corr := pearson(got, want)
		t.Logf("%s: max|diff| = %.4g, mean|diff| = %.4g, pearson = %.4f, peak go=%d py=%d",
			name, maxDiff, meanDiff, corr, argmax(got), argmax(want))
		if maxDiff > maxTol || meanDiff > meanTol {
			t.Errorf("%s: relevance mismatch vs Python: max|diff| = %.4g (<= %g), mean|diff| = %.4g (<= %g)",
				name, maxDiff, maxTol, meanDiff, meanTol)
		}
		if corr < minCorr {
			t.Errorf("%s: relevance correlation %.4f below %.4f", name, corr, minCorr)
		}
	}

	check("noise fixture", g.PixelValues, g.Relevance)
	if g.FundusFixture != nil {
		check("fundus fixture", g.FundusFixture.PixelValues, g.FundusFixture.Relevance)
	}
}

// TestExplainMethodSelection covers the operator override: forcing rollout on
// a gradient-capable graph must work, and asking for gradients on a graph
// that has them must report the class-specific method.
func TestExplainMethodSelection(t *testing.T) {
	forced := newTestEngineWithConfig(t, func(c *Config) { c.Explain = "rollout" })
	if forced.ExplainMethod() != ExplainRollout || forced.gradsIdx >= 0 {
		t.Errorf("forced rollout: method=%q gradsIdx=%d", forced.ExplainMethod(), forced.gradsIdx)
	}
	img := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 90, 40, 20, 255
	}
	res, err := forced.Predict(context.Background(), img)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	if res.Saliency == nil || res.Saliency.Method != ExplainRollout {
		t.Errorf("forced rollout produced %+v", res.Saliency)
	}

	auto := newTestEngine(t)
	if auto.gradsIdx < 0 {
		t.Skip("served graph has no attn_grads output")
	}
	if auto.ExplainMethod() != ExplainGradRelevance {
		t.Errorf("auto on a gradient graph: method = %q", auto.ExplainMethod())
	}
	strict := newTestEngineWithConfig(t, func(c *Config) { c.Explain = "grad-relevance" })
	if strict.ExplainMethod() != ExplainGradRelevance {
		t.Errorf("explicit grad-relevance: method = %q", strict.ExplainMethod())
	}
}

// pearson is the correlation coefficient of two equal-length vectors.
func pearson(a, b []float32) float64 {
	var ma, mb float64
	for i := range a {
		ma += float64(a[i])
		mb += float64(b[i])
	}
	ma /= float64(len(a))
	mb /= float64(len(b))
	var cov, va, vb float64
	for i := range a {
		da, db := float64(a[i])-ma, float64(b[i])-mb
		cov += da * db
		va += da * da
		vb += db * db
	}
	if va == 0 || vb == 0 {
		return 0
	}
	return cov / math.Sqrt(va*vb)
}

// TestPredictSerializes verifies the capacity-1 semaphore actually holds. On a
// 2 vCPU box, concurrent forward passes are the difference between steady
// latency and swap-thrash.
func TestPredictSerializes(t *testing.T) {
	eng := newTestEngine(t)

	img := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for i := range img.Pix {
		img.Pix[i] = 128
	}

	var mu sync.Mutex
	concurrent, maxConcurrent := 0, 0

	// Set the hook before any goroutine starts, so this is not itself a race.
	eng.inflightHook = func(delta int) {
		mu.Lock()
		defer mu.Unlock()
		concurrent += delta
		if concurrent > maxConcurrent {
			maxConcurrent = concurrent
		}
	}

	// Allow every caller to queue rather than being shed, so this measures
	// serialization rather than load shedding.
	eng.maxQueueDepth = 16

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := eng.Predict(context.Background(), img); err != nil && !errors.Is(err, ErrBusy) {
				t.Errorf("predict: %v", err)
			}
		}()
	}
	wg.Wait()

	if maxConcurrent > 1 {
		t.Errorf("observed %d concurrent inferences, want at most 1", maxConcurrent)
	}
}

func TestPredictShedsLoadWhenQueueFull(t *testing.T) {
	eng := newTestEngine(t)
	eng.maxQueueDepth = 1

	img := image.NewRGBA(image.Rect(0, 0, 224, 224))

	// Occupy the slot so subsequent callers queue.
	eng.sem <- struct{}{}
	eng.waiting.Store(1)
	defer func() { <-eng.sem; eng.waiting.Store(0) }()

	if _, err := eng.Predict(context.Background(), img); !errors.Is(err, ErrBusy) {
		t.Fatalf("expected ErrBusy, got %v", err)
	}
}

func TestPredictRespectsContextCancellation(t *testing.T) {
	eng := newTestEngine(t)
	img := image.NewRGBA(image.Rect(0, 0, 224, 224))

	// Hold the slot so Predict must wait, then let the context expire.
	eng.sem <- struct{}{}
	defer func() { <-eng.sem }()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := eng.Predict(ctx, img)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded, got %v", err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Predict took %v to honour cancellation", elapsed)
	}
}
