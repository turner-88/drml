package infer

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"golang.org/x/image/webp"
)

func init() {
	// Match the decoders the upload path registers, so the sweep sees the same
	// corpus the application does.
	image.RegisterFormat("webp", "RIFF????WEBPVP8", webp.Decode, webp.DecodeConfig)
}

// TestSaliencyConcentrationSweep measures Saliency.Concentration over every
// stored scan, which is how HeatmapOptions.MinConcentration and
// FullConcentration get their values.
//
// It is a measurement, not an assertion: guessing those two thresholds would
// reintroduce the exact problem the ramp exists to fix. Run it against a real
// corpus and read the printed distribution:
//
//	DRML_SWEEP_DIR=/abs/path/to/scans go test ./internal/infer/ \
//	    -run TestSaliencyConcentrationSweep -v
//
// Set DRML_SWEEP_OUT as well to write the rendered overlays there, which is the
// only way to judge whether a candidate threshold looks right rather than
// merely reading well as a number.
func TestSaliencyConcentrationSweep(t *testing.T) {
	dir := os.Getenv("DRML_SWEEP_DIR")
	if dir == "" {
		t.Skip("set DRML_SWEEP_DIR to a directory of fundus images to run the sweep")
	}
	eng := newTestEngine(t)

	var files []string
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		switch filepath.Ext(p) {
		case ".jpg", ".jpeg", ".png", ".webp":
			files = append(files, p)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", dir, err)
	}
	if len(files) == 0 {
		t.Skipf("no images under %s", dir)
	}
	sort.Strings(files)

	var concs []float64
	for _, p := range files {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Logf("%s: read: %v", p, err)
			continue
		}
		img, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Logf("%s: decode: %v", p, err)
			continue
		}
		img = Upright(img, raw)

		res, err := eng.Predict(context.Background(), img)
		if err != nil {
			t.Logf("%s: predict: %v", p, err)
			continue
		}
		if res.Saliency == nil {
			t.Logf("%s: no saliency", p)
			continue
		}

		// How much of the map survived the ROI mask says whether masking is
		// doing anything on this corpus at all.
		var lit int
		for _, v := range res.Saliency.Grid {
			if v > 0 {
				lit++
			}
		}
		fmt.Printf("%-44s exif=%d grade=%d conf=%.3f conc=%.4f lit=%d/%d\n",
			filepath.Base(p), JPEGOrientation(raw), res.Grade, res.Confidence,
			res.Saliency.Concentration, lit, len(res.Saliency.Grid))
		concs = append(concs, res.Saliency.Concentration)

		if out := os.Getenv("DRML_SWEEP_OUT"); out != "" {
			writeOverlay(t, out, filepath.Base(p), img, res.Saliency)
		}
	}

	if len(concs) == 0 {
		t.Skip("no images produced a saliency map")
	}
	sort.Float64s(concs)
	fmt.Printf("\nconcentration over %d images:\n", len(concs))
	for _, p := range []float64{0, 5, 25, 50, 75, 95, 100} {
		fmt.Printf("  p%-3.0f %.4f\n", p, percentileSorted(concs, p))
	}
	fmt.Println("\nSuggested: MinConcentration near p5, FullConcentration near p75.")
}

func writeOverlay(t *testing.T, dir, name string, img image.Image, sal *Saliency) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	overlay := RenderHeatmap(img, sal, DefaultHeatmapOptions())
	f, err := os.Create(filepath.Join(dir, name+".overlay.jpg"))
	if err != nil {
		t.Fatalf("create overlay: %v", err)
	}
	defer f.Close()
	if err := jpeg.Encode(f, overlay, &jpeg.Options{Quality: 85}); err != nil {
		t.Fatalf("encode overlay: %v", err)
	}
}
