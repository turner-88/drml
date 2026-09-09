package infer

import (
	"encoding/json"
	"errors"
	"image"
	"image/color"
	_ "image/png"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// --- synthetic images -------------------------------------------------------
//
// These pin the decision logic without depending on any download. The real
// images fetched by model/gate.py are asserted separately, and skipped when
// absent.

func solid(w, h int, r, g, b uint8) image.Image {
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	return m
}

func clamp8(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// noisy is a flat colour with per-pixel noise: texture but no structure.
func noisy(w, h, r, g, b int, seed int64) image.Image {
	rng := rand.New(rand.NewSource(seed))
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			n := rng.Intn(60) - 30
			m.Set(x, y, color.RGBA{clamp8(r + n), clamp8(g + n), clamp8(b + n), 255})
		}
	}
	return m
}

// fundusLike is a caricature of a fundus frame: a textured red-orange disc on
// a near-black field. It exercises the ROI and red-dominance features together
// without needing a real retina.
func fundusLike(w, h int) image.Image {
	rng := rand.New(rand.NewSource(11))
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	cx, cy := float64(w)/2, float64(h)/2
	rad := math.Min(cx, cy) * 0.95
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if math.Hypot(float64(x)-cx, float64(y)-cy) > rad {
				m.Set(x, y, color.RGBA{2, 1, 1, 255})
				continue
			}
			n := rng.Intn(70) - 35
			m.Set(x, y, color.RGBA{clamp8(190 + n), clamp8(90 + n), clamp8(55 + n), 255})
		}
	}
	return m
}

// pageLike is a light background with dark text-like bars: the shape of a
// screenshot or a scanned document.
func pageLike(w, h int) image.Image {
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.RGBA{248, 249, 250, 255}
			if y%40 < 12 && x > 60 && x < w-60 {
				c = color.RGBA{40, 42, 46, 255}
			}
			m.Set(x, y, c)
		}
	}
	return m
}

func TestCheckImage(t *testing.T) {
	g := DefaultGateConfig()

	cases := []struct {
		name       string
		img        image.Image
		wantPass   bool
		wantReason string
	}{
		{"fundus-like disc", fundusLike(512, 512), true, ""},
		{"thumbnail", fundusLike(64, 64), false, ReasonTooSmall},
		{"just under min_dim", fundusLike(223, 400), false, ReasonTooSmall},
		{"widescreen screenshot", pageLike(1920, 800), false, ReasonAspect},
		{"solid grey", solid(512, 512, 128, 128, 128), false, ReasonNotFundus},
		{"blank white page", solid(512, 512, 250, 250, 250), false, ReasonNotFundus},
		{"solid black frame", solid(512, 512, 0, 0, 0), false, ReasonNotFundus},
		{"flat skin tone", solid(512, 512, 224, 172, 145), false, ReasonNotFundus},
		{"noisy skin tone", noisy(512, 512, 224, 172, 145, 3), false, ReasonNotFundus},
		{"foliage", noisy(512, 512, 60, 140, 55, 5), false, ReasonNotFundus},
		{"document page", pageLike(1000, 800), false, ReasonNotFundus},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep := g.CheckImage(tc.img)
			if rep.Passed != tc.wantPass {
				t.Errorf("Passed = %v, want %v (reason %q, score %.3f, red %.3f, ratio %.3f)",
					rep.Passed, tc.wantPass, rep.Reason,
					rep.Metrics["score"], rep.Metrics["red_ratio"],
					rep.Metrics["centre_contrast"])
			}
			if rep.Reason != tc.wantReason {
				t.Errorf("Reason = %q, want %q", rep.Reason, tc.wantReason)
			}
		})
	}
}

// TestCheckImageRejectsGoldenFixture guards against the mistake of treating
// model/testdata/fixture.png as a fundus image. It is seeded uniform noise from
// export.py, used for the preprocessing parity test — a useful negative here.
func TestCheckImageRejectsGoldenFixture(t *testing.T) {
	f, err := os.Open(filepath.Join(modelDir(t), "testdata", "fixture.png"))
	if err != nil {
		t.Skipf("fixture missing (run model/export.py): %v", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	g := DefaultGateConfig()
	// Bypass the size prefilter so this asserts the content features, not the
	// fact that the fixture happens to be 173x137.
	g.MinDim = 0
	rep := g.CheckImage(img)
	if rep.Passed {
		t.Errorf("random-noise fixture passed the gate (score %.3f, red_ratio %.3f)",
			rep.Metrics["score"], rep.Metrics["red_ratio"])
	}
}

// loadSamples reads the real images model/gate.py caches during calibration.
func loadSamples(t *testing.T, kind string) []image.Image {
	t.Helper()
	dir := filepath.Join(modelDir(t), "testdata", kind)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Skipf("no %s samples (run model/gate.py to cache them): %v", kind, err)
	}
	var out []image.Image
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".png" {
			continue
		}
		f, err := os.Open(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("open %s: %v", e.Name(), err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatalf("decode %s: %v", e.Name(), err)
		}
		out = append(out, img)
	}
	if len(out) == 0 {
		t.Skipf("no %s samples cached", kind)
	}
	return out
}

// TestGateOnRealImages is the one that actually matters: synthetic shapes prove
// the arithmetic, real retinas prove the thresholds. It asserts on the rate
// rather than per-image, because the thresholds are deliberately calibrated to
// admit a small number of misses rather than reject real scans.
func TestGateOnRealImages(t *testing.T) {
	cfg, err := LoadGateConfig(filepath.Join(modelDir(t), "gate.json"))
	if err != nil {
		t.Skipf("gate.json missing (run model/gate.py): %v", err)
	}

	fundus := loadSamples(t, "fundus")
	var admitted int
	for _, img := range fundus {
		if cfg.CheckImage(img).Passed {
			admitted++
		}
	}
	if admitted != len(fundus) {
		t.Errorf("gate rejected %d/%d real fundus images; false rejections are the "+
			"expensive error, recalibrate with model/gate.py", len(fundus)-admitted, len(fundus))
	}

	negatives := loadSamples(t, "negatives")
	var leaked int
	for _, img := range negatives {
		if cfg.CheckImage(img).Passed {
			leaked++
		}
	}
	// Not zero: the gate is tuned to protect real scans, so some hard negatives
	// get through to the classifier and its low-confidence warning. This only
	// fails if the gate has stopped discriminating at all.
	if leaked > len(negatives)/2 {
		t.Errorf("gate admitted %d/%d non-fundus images, expected under half",
			leaked, len(negatives))
	}
	t.Logf("real images: fundus admitted %d/%d, non-fundus admitted %d/%d",
		admitted, len(fundus), leaked, len(negatives))
}

func TestCheckLogits(t *testing.T) {
	g := DefaultGateConfig()
	g.EnergyEnabled = true
	g.EnergyMax = -5.0

	// logsumexp([8,1,1,1,1]) ~= 8.02, so energy ~= -8.02, below the threshold.
	confident := []float32{8, 1, 1, 1, 1}
	if rep := g.CheckLogits(confident); !rep.Passed {
		t.Errorf("confident logits rejected: energy %.3f", rep.Metrics["energy"])
	}
	// logsumexp([0.1 x5]) ~= 1.71, energy ~= -1.71, above -5.
	diffuse := []float32{0.1, 0.1, 0.1, 0.1, 0.1}
	rep := g.CheckLogits(diffuse)
	if rep.Passed {
		t.Errorf("diffuse logits admitted: energy %.3f", rep.Metrics["energy"])
	}
	if rep.Reason != ReasonOODEnergy {
		t.Errorf("Reason = %q, want %q", rep.Reason, ReasonOODEnergy)
	}

	// Disabled is the shipped default and must be inert.
	off := DefaultGateConfig()
	if !off.CheckLogits(diffuse).Passed {
		t.Error("energy tier rejected while disabled")
	}
}

func TestEnergyScore(t *testing.T) {
	// -logsumexp is shift-covariant: adding c to every logit lowers energy by c.
	base := []float32{1, 2, 3, 4, 5}
	shifted := []float32{3, 4, 5, 6, 7}
	if got, want := EnergyScore(shifted), EnergyScore(base)-2; math.Abs(got-want) > 1e-5 {
		t.Errorf("EnergyScore(shifted) = %.6f, want %.6f", got, want)
	}
	if got := EnergyScore(nil); got != 0 {
		t.Errorf("EnergyScore(nil) = %v, want 0", got)
	}
}

// A nil or disabled config must be completely inert, because that is what a
// deployment without a calibrated gate.json runs.
func TestGateDisabled(t *testing.T) {
	var nilCfg *GateConfig
	if !nilCfg.CheckImage(solid(10, 10, 0, 0, 0)).Passed {
		t.Error("nil config rejected an image")
	}
	if !nilCfg.CheckLogits([]float32{0, 0, 0, 0, 0}).Passed {
		t.Error("nil config rejected logits")
	}

	off := DefaultGateConfig()
	off.Enabled = false
	if !off.CheckImage(solid(10, 10, 0, 0, 0)).Passed {
		t.Error("disabled config rejected an image")
	}
}

func TestLoadGateConfig(t *testing.T) {
	dir := t.TempDir()

	t.Run("missing file reports fs.ErrNotExist", func(t *testing.T) {
		// engine.New relies on this to disable the gate instead of failing
		// startup, so the error must stay unwrapped enough for errors.Is.
		_, err := LoadGateConfig(filepath.Join(dir, "absent.json"))
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("err = %v, want fs.ErrNotExist", err)
		}
	})

	valid := map[string]any{
		"enabled": true, "min_dim": 224, "max_aspect": 2.0, "luma_floor": 0.12,
		"red_weight": 0.45, "roi_weight": 0.35, "texture_weight": 0.2,
		"score_min": 0.5, "energy_enabled": false, "energy_max": 0.0,
	}

	bad := []struct {
		name   string
		mutate func(m map[string]any)
	}{
		{"negative min_dim", func(m map[string]any) { m["min_dim"] = -1 }},
		{"luma_floor at 1", func(m map[string]any) { m["luma_floor"] = 1.0 }},
		{"score_min above 1", func(m map[string]any) { m["score_min"] = 1.5 }},
		{"zero weights", func(m map[string]any) {
			m["red_weight"], m["roi_weight"], m["texture_weight"] = 0.0, 0.0, 0.0
		}},
	}
	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			m := map[string]any{}
			for k, v := range valid {
				m[k] = v
			}
			tc.mutate(m)
			p := filepath.Join(dir, "bad.json")
			raw, _ := json.Marshal(m)
			if err := os.WriteFile(p, raw, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadGateConfig(p); err == nil {
				t.Error("expected an error, got nil")
			}
		})
	}

	t.Run("valid", func(t *testing.T) {
		p := filepath.Join(dir, "ok.json")
		raw, _ := json.Marshal(valid)
		if err := os.WriteFile(p, raw, 0o600); err != nil {
			t.Fatal(err)
		}
		cfg, err := LoadGateConfig(p)
		if err != nil {
			t.Fatalf("LoadGateConfig: %v", err)
		}
		if cfg.MinDim != 224 || cfg.ScoreMin != 0.5 || !cfg.Enabled {
			t.Errorf("round-trip mismatch: %+v", cfg)
		}
	})
}

// The gate runs on every upload before the inference semaphore, so it has to be
// negligible next to the ~131 ms forward pass.
func BenchmarkCheckImage(b *testing.B) {
	g := DefaultGateConfig()
	img := fundusLike(2048, 2048)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.CheckImage(img)
	}
}

// TestEngineToleratesMissingGate covers the upgrade path: a binary that knows
// about the gate, deployed before model/gate.py has been run. It must serve
// with the gate off rather than refusing to start.
func TestEngineToleratesMissingGate(t *testing.T) {
	eng := newTestEngineWithGate(t, filepath.Join(t.TempDir(), "absent.json"))
	if rep := eng.CheckImage(solid(512, 512, 128, 128, 128)); !rep.Passed {
		t.Errorf("gate active despite missing gate.json: reason %q", rep.Reason)
	}
}

// TestEngineLoadsGate is the mirror: with a config present, the engine applies
// it, so the wiring between New and CheckImage is real.
func TestEngineLoadsGate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gate.json")
	raw, err := json.Marshal(DefaultGateConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	eng := newTestEngineWithGate(t, path)
	if rep := eng.CheckImage(solid(512, 512, 128, 128, 128)); rep.Passed {
		t.Error("engine admitted a solid grey image with the gate configured")
	}
	if rep := eng.CheckImage(fundusLike(512, 512)); !rep.Passed {
		t.Errorf("engine rejected a fundus-like image: %q", rep.Reason)
	}
}

// TestSetGateEnabled covers the admin toggle: the flag must reach the request
// path with no restart, in both directions.
func TestSetGateEnabled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gate.json")
	raw, err := json.Marshal(DefaultGateConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	eng := newTestEngineWithGate(t, path)
	junk := solid(512, 512, 128, 128, 128)

	if !eng.GateAvailable() {
		t.Fatal("GateAvailable() = false with a calibration file present")
	}
	if !eng.GateEnabled() {
		t.Fatal("gate not enabled by default from a config with enabled:true")
	}
	if eng.CheckImage(junk).Passed {
		t.Fatal("solid grey admitted while the gate is on")
	}

	if ok := eng.SetGateEnabled(false); !ok {
		t.Fatal("SetGateEnabled(false) refused")
	}
	if eng.GateEnabled() {
		t.Error("GateEnabled() still true after disabling")
	}
	if !eng.CheckImage(junk).Passed {
		t.Error("solid grey still rejected after disabling the gate")
	}

	if ok := eng.SetGateEnabled(true); !ok {
		t.Fatal("SetGateEnabled(true) refused with calibration loaded")
	}
	if eng.CheckImage(junk).Passed {
		t.Error("solid grey admitted after re-enabling the gate")
	}
}

// TestSetGateEnabledWithoutCalibration: there is nothing to enable when
// gate.json is absent, and the caller must be able to tell that from success —
// otherwise the admin UI would show a switch that silently does nothing.
func TestSetGateEnabledWithoutCalibration(t *testing.T) {
	eng := newTestEngineWithGate(t, filepath.Join(t.TempDir(), "absent.json"))

	if eng.GateAvailable() {
		t.Error("GateAvailable() = true with no calibration file")
	}
	if eng.GateEnabled() {
		t.Error("GateEnabled() = true with no calibration file")
	}
	if ok := eng.SetGateEnabled(true); ok {
		t.Error("SetGateEnabled(true) reported success with no calibration")
	}
	if eng.GateEnabled() {
		t.Error("gate enabled despite having no thresholds to apply")
	}
	// Disabling an already-disabled gate is not an error.
	if ok := eng.SetGateEnabled(false); !ok {
		t.Error("SetGateEnabled(false) refused")
	}
}

// TestGateToggleIsRaceFree runs readers against a writer, which is exactly the
// production shape: uploads read the flag while an administrator flips it.
// Meaningful under -race.
func TestGateToggleIsRaceFree(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gate.json")
	raw, _ := json.Marshal(DefaultGateConfig())
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	eng := newTestEngineWithGate(t, path)
	img := solid(300, 300, 128, 128, 128)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					eng.CheckImage(img)
				}
			}
		}()
	}
	for i := 0; i < 200; i++ {
		eng.SetGateEnabled(i%2 == 0)
	}
	close(stop)
	wg.Wait()
}
