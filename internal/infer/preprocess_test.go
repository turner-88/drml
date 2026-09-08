package infer

import (
	"encoding/json"
	"image"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// goldenFixture is written by model/export.py.
type goldenFixture struct {
	Image            string           `json:"image"`
	ImageHeight      int              `json:"image_height"`
	ImageWidth       int              `json:"image_width"`
	Preprocess       PreprocessConfig `json:"preprocess"`
	PixelValuesShape []int            `json:"pixel_values_shape"`
	PixelValues      []float32        `json:"pixel_values"`
	Logits           []float32        `json:"logits"`
	AttentionsShape  []int            `json:"attentions_shape"`
}

func modelDir(t *testing.T) string {
	t.Helper()
	// tests run in internal/infer; the artifacts live at <repo>/model
	dir, err := filepath.Abs(filepath.Join("..", "..", "model"))
	if err != nil {
		t.Fatalf("resolve model dir: %v", err)
	}
	return dir
}

func loadGolden(t *testing.T) (goldenFixture, string) {
	t.Helper()
	dir := modelDir(t)
	raw, err := os.ReadFile(filepath.Join(dir, "testdata", "golden.json"))
	if err != nil {
		t.Skipf("golden fixture missing (run model/export.py): %v", err)
	}
	var g goldenFixture
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden.json: %v", err)
	}
	return g, dir
}

// TestPreprocessParity is the test that catches resize-filter, channel-order
// and normalization mistakes — the usual reasons a reimplemented preprocessing
// pipeline produces confidently wrong predictions rather than obvious errors.
func TestPreprocessParity(t *testing.T) {
	g, dir := loadGolden(t)

	f, err := os.Open(filepath.Join(dir, "testdata", g.Image))
	if err != nil {
		t.Fatalf("open fixture image: %v", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatalf("decode fixture image: %v", err)
	}

	if got := img.Bounds().Dx(); got != g.ImageWidth {
		t.Fatalf("fixture width = %d, want %d", got, g.ImageWidth)
	}

	cfg := g.Preprocess
	got := cfg.Tensor(img)

	if len(got) != len(g.PixelValues) {
		t.Fatalf("tensor length = %d, want %d", len(got), len(g.PixelValues))
	}

	var maxDiff float64
	var sumDiff float64
	worstAt := -1
	for i := range got {
		d := math.Abs(float64(got[i] - g.PixelValues[i]))
		sumDiff += d
		if d > maxDiff {
			maxDiff, worstAt = d, i
		}
	}
	meanDiff := sumDiff / float64(len(got))

	// Go's x/image and PIL implement bilinear kernels independently and round
	// to 8-bit differently, so bit-exact equality is unreachable. The floor is
	// one quantization step: with rescale=1/255 and std=0.5 a single 8-bit LSB
	// is (1/255)/0.5 = 0.00784 in normalized units. Anything at or below that
	// is rounding, not a pipeline bug — and TestLogitsParity is the check that
	// actually decides whether it matters.
	lsb := float64(cfg.RescaleFactor / cfg.ImageStd[0])
	maxTol := lsb * 1.05
	meanTol := lsb * 0.5

	t.Logf("pixel_values: max|diff|=%.6g (%.2f LSB) mean|diff|=%.6g (%.2f LSB), 1 LSB=%.6g, worst index %d",
		maxDiff, maxDiff/lsb, meanDiff, meanDiff/lsb, lsb, worstAt)

	if maxDiff > maxTol {
		t.Errorf("preprocessing mismatch: max|diff| = %.6g (%.2f LSB), want <= %.6g",
			maxDiff, maxDiff/lsb, maxTol)
	}
	if meanDiff > meanTol {
		t.Errorf("preprocessing mean drift = %.6g (%.2f LSB), want <= %.6g — suggests a "+
			"systematic filter difference, not rounding", meanDiff, meanDiff/lsb, meanTol)
	}
}

// TestPreprocessShapeAndRange guards the tensor layout independently of the
// fixture, so a layout regression is still caught if the fixture is stale.
func TestPreprocessShapeAndRange(t *testing.T) {
	cfg := &PreprocessConfig{
		Height: 4, Width: 4,
		ImageMean:     []float32{0.5, 0.5, 0.5},
		ImageStd:      []float32{0.5, 0.5, 0.5},
		RescaleFactor: 1.0 / 255.0,
		DoNormalize:   true,
		DoRescale:     true,
		ResampleCode:  2,
	}

	// A pure-red image must normalize to +1 on channel 0 and -1 on 1 and 2.
	src := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i], src.Pix[i+1], src.Pix[i+2], src.Pix[i+3] = 255, 0, 0, 255
	}

	got := cfg.Tensor(src)
	if len(got) != 3*4*4 {
		t.Fatalf("tensor length = %d, want %d", len(got), 3*4*4)
	}
	plane := 4 * 4
	for i := 0; i < plane; i++ {
		if math.Abs(float64(got[i]-1)) > 1e-6 {
			t.Fatalf("R plane [%d] = %v, want 1", i, got[i])
		}
		if math.Abs(float64(got[plane+i]+1)) > 1e-6 {
			t.Fatalf("G plane [%d] = %v, want -1", i, got[plane+i])
		}
		if math.Abs(float64(got[2*plane+i]+1)) > 1e-6 {
			t.Fatalf("B plane [%d] = %v, want -1", i, got[2*plane+i])
		}
	}
}

func TestLoadPreprocessConfigRejectsBadInput(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]string{
		"zero size":  `{"height":0,"width":224,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`,
		"short mean": `{"height":224,"width":224,"image_mean":[0.5],"image_std":[0.5,0.5,0.5]}`,
		"zero std":   `{"height":224,"width":224,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0,0.5]}`,
		"not json":   `{`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			p := filepath.Join(dir, "cfg.json")
			if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadPreprocessConfig(p); err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}
