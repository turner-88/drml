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
	dir := modelDir(t)

	modelPath := filepath.Join(dir, "vit_dr_int8.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model missing (run model/export.py): %v", err)
	}

	eng, err := New(Config{
		ModelPath:      modelPath,
		LabelsPath:     filepath.Join(dir, "labels.json"),
		PreprocessPath: filepath.Join(dir, "preprocess.json"),
		GatePath:       gatePath,
		ORTLibPath:     ortLibPathForTest(),
		IntraOpThreads: 2,
		InterOpThreads: 1,
		MaxQueueDepth:  4,
	})
	if err != nil {
		t.Skipf("cannot start engine (is onnxruntime installed?): %v", err)
	}
	t.Cleanup(eng.Close)
	return eng
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

	// The heatmap must be present and shaped for a 224px ViT-B/16.
	if res.HeatmapDim != 14 {
		t.Errorf("heatmap dim = %d, want 14", res.HeatmapDim)
	}
	if len(res.Heatmap) != res.HeatmapDim*res.HeatmapDim {
		t.Errorf("heatmap has %d values, want %d", len(res.Heatmap), res.HeatmapDim*res.HeatmapDim)
	}
	for _, v := range res.Heatmap {
		if v < 0 || v > 1 {
			t.Fatalf("heatmap value %v outside [0,1]", v)
		}
	}
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
