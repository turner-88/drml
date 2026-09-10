package infer

import (
	"context"
	"fmt"
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestMemoryFootprint measures what the inference path costs in resident
// memory, which is the number that decides whether a graph fits beside MySQL
// on the 2 GB VPS. It is a diagnostic, not an assertion:
//
//	DRML_MEASURE_RSS=1 go test ./internal/infer/ -run TestMemoryFootprint -v
//
// It reports RSS before the engine loads, after loading, and after 40
// predictions with heatmap rendering over model/testdata/fundus, then prints
// the mean latency. Combine with DRML_TEST_MODEL to compare artifacts.
func TestMemoryFootprint(t *testing.T) {
	if os.Getenv("DRML_MEASURE_RSS") == "" {
		t.Skip("set DRML_MEASURE_RSS=1 to measure the inference memory footprint")
	}
	before := rssMB(t)
	eng := newTestEngine(t)
	loaded := rssMB(t)

	dir := filepath.Join(modelDir(t), "testdata", "fundus")
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) == 0 {
		t.Skipf("no fundus samples under %s", dir)
	}
	var imgs []image.Image
	for _, e := range entries {
		f, err := os.Open(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatalf("decode %s: %v", e.Name(), err)
		}
		imgs = append(imgs, img)
	}

	const runs = 40
	start := time.Now()
	for i := 0; i < runs; i++ {
		img := imgs[i%len(imgs)]
		res, err := eng.Predict(context.Background(), img)
		if err != nil {
			t.Fatalf("predict: %v", err)
		}
		if res.Saliency != nil {
			RenderHeatmap(img, res.Saliency, DefaultHeatmapOptions())
		}
	}
	perRun := time.Since(start) / runs
	after := rssMB(t)

	fmt.Printf("outputs=%v method=%s\n", eng.outputNames, explainLabel(eng.ExplainMethod()))
	fmt.Printf("rss: before load %d MB, loaded %d MB (+%d), after %d predictions %d MB (+%d)\n",
		before, loaded, loaded-before, runs, after, after-before)
	fmt.Printf("latency: %d ms per prediction incl. preprocessing, saliency and heatmap render\n",
		perRun.Milliseconds())
}

// rssMB reads this process's resident set size through ps, which works the
// same on macOS and Linux and is exact enough for a diagnostic.
func rssMB(t *testing.T) int {
	t.Helper()
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Skipf("ps: %v", err)
	}
	kb, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		t.Skipf("parse ps output %q: %v", out, err)
	}
	return kb / 1024
}
