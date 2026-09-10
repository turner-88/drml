package infer

import (
	"image"
	"testing"
)

// discImage draws a bright disc of the given radius on black, the shape every
// fundus camera produces.
func discImage(size int, radius float64) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	c := float64(size-1) / 2
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := img.PixOffset(x, y)
			dx, dy := float64(x)-c, float64(y)-c
			if dx*dx+dy*dy <= radius*radius {
				img.Pix[i], img.Pix[i+1], img.Pix[i+2] = 200, 120, 90
			}
			img.Pix[i+3] = 255
		}
	}
	return img
}

func TestFundusMaskExcludesTheBlackSurround(t *testing.T) {
	const side = 14
	img := discImage(224, 100) // disc inscribed in the frame, corners black
	mask := FundusMask(img, side, DefaultGateConfig().LumaFloor)
	if mask == nil {
		t.Fatal("mask is nil; a disc on black is exactly the case it must handle")
	}

	corners := []int{
		0, side - 1, (side - 1) * side, side*side - 1,
	}
	for _, i := range corners {
		if mask[i] {
			t.Errorf("cell %d is a black corner but was kept", i)
		}
	}
	centre := (side/2)*side + side/2
	if !mask[centre] {
		t.Error("centre cell was masked out")
	}
}

func TestFundusMaskKeepsEverythingOnATightCrop(t *testing.T) {
	// A fundus cropped to the disc has no border to find; masking must be a
	// no-op rather than eating the image.
	img := discImage(224, 1000)
	mask := FundusMask(img, 14, DefaultGateConfig().LumaFloor)
	if mask == nil {
		return // nil means "keep every cell", also correct
	}
	for i, in := range mask {
		if !in {
			t.Fatalf("cell %d masked out of a fully illuminated frame", i)
		}
	}
}

func TestFundusMaskRejectsAnUnusableMask(t *testing.T) {
	// An all-black frame would mask out everything; that is a failed heuristic,
	// not a saliency map with no valid cells.
	black := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for i := 3; i < len(black.Pix); i += 4 {
		black.Pix[i] = 255
	}
	if mask := FundusMask(black, 14, DefaultGateConfig().LumaFloor); mask != nil {
		t.Error("expected nil (no trustworthy mask), got one")
	}
}

func TestRolloutMaskZeroesTheSurround(t *testing.T) {
	const layers, heads, tokens = 2, 2, 17 // 16 patches -> 4x4
	const target = 1                       // patch 0, which the mask excludes

	attn := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		if j == target {
			return 1
		}
		return 0
	})

	mask := make([]bool, 16)
	for i := range mask {
		mask[i] = i != 0
	}

	sal, err := Rollout(attn, shape(layers, heads, tokens), mask)
	if err != nil {
		t.Fatalf("rollout: %v", err)
	}
	if sal.Grid[0] != 0 {
		t.Errorf("masked-out patch has saliency %v, want 0", sal.Grid[0])
	}
	// Every remaining patch drew the same (near zero) attention, so the map
	// inside the ROI carries no signal and must render as none.
	for i, v := range sal.Grid {
		if v != 0 {
			t.Errorf("patch %d = %v; the in-ROI map is flat and must be 0", i, v)
		}
	}
}

func TestRolloutRejectsAMaskOfTheWrongSize(t *testing.T) {
	const layers, heads, tokens = 1, 1, 5
	attn := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 { return 0.2 })
	if _, err := Rollout(attn, shape(layers, heads, tokens), make([]bool, 3)); err == nil {
		t.Fatal("expected an error for a 3-cell mask over 4 patches")
	}
}

func TestConcentrationSeparatesFlatFromPeaked(t *testing.T) {
	const layers, heads, tokens = 3, 2, 17

	uniform := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		return float32(1) / float32(tokens)
	})
	flat, err := Rollout(uniform, shape(layers, heads, tokens), nil)
	if err != nil {
		t.Fatalf("rollout: %v", err)
	}
	if got := flat.Concentration; got < 0.999 || got > 1.001 {
		t.Errorf("uniform attention scored %v, want ~1.0", got)
	}

	peaked := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		if j == 4 {
			return 1
		}
		return 0
	})
	sharp, err := Rollout(peaked, shape(layers, heads, tokens), nil)
	if err != nil {
		t.Fatalf("rollout: %v", err)
	}
	if sharp.Concentration <= flat.Concentration {
		t.Errorf("peaked attention scored %v, not above uniform's %v",
			sharp.Concentration, flat.Concentration)
	}
}

func TestEffectiveAlphaRamp(t *testing.T) {
	opts := HeatmapOptions{Alpha: 0.5, MinConcentration: 2, FullConcentration: 4}
	cases := []struct {
		conc float64
		want float64
	}{
		{1.0, 0},
		{2.0, 0},
		{3.0, 0.25},
		{4.0, 0.5},
		{9.0, 0.5},
	}
	for _, tc := range cases {
		if got := opts.EffectiveAlpha(tc.conc); got != tc.want {
			t.Errorf("EffectiveAlpha(%v) = %v, want %v", tc.conc, got, tc.want)
		}
	}

	// Both zero disables the ramp entirely.
	off := HeatmapOptions{Alpha: 0.45}
	if got := off.EffectiveAlpha(0); got != 0.45 {
		t.Errorf("disabled ramp returned %v, want the full alpha", got)
	}
}

func TestPercentileClippingIgnoresASingleOutlier(t *testing.T) {
	// One runaway cell must not crush every other value to the bottom of the
	// colour ramp, which is exactly what a raw min-max does.
	v := make([]float32, 100)
	for i := range v {
		v[i] = float32(i) / 100 // 0.00 .. 0.99
	}
	v[0] = 1000
	normalizeSaliency(v, nil)

	if v[0] != 1 {
		t.Errorf("outlier normalized to %v, want 1 (clipped)", v[0])
	}
	if v[99] < 0.9 {
		t.Errorf("the top of the real range normalized to %v; the outlier swallowed the ramp", v[99])
	}
}

func TestRenderHeatmapNeverPaintsTheSurround(t *testing.T) {
	const side = 14
	src := discImage(224, 100)

	// Saliency saturated everywhere: the worst case for bleed, and what
	// upsampling a hot boundary cell effectively produces at the disc edge.
	grid := make([]float32, side*side)
	for i := range grid {
		grid[i] = 1
	}
	out := RenderHeatmap(src, &Saliency{Grid: grid, Side: side}, DefaultHeatmapOptions())

	rgba, ok := out.(*image.RGBA)
	if !ok {
		t.Fatalf("RenderHeatmap returned %T, want *image.RGBA", out)
	}
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			i := rgba.PixOffset(x, y)
			s := src.PixOffset(x, y)
			if src.Pix[s] != 0 || src.Pix[s+1] != 0 || src.Pix[s+2] != 0 {
				continue // inside the disc; painting there is the point
			}
			if rgba.Pix[i] != 0 || rgba.Pix[i+1] != 0 || rgba.Pix[i+2] != 0 {
				t.Fatalf("black surround pixel (%d,%d) was painted (%d,%d,%d)",
					x, y, rgba.Pix[i], rgba.Pix[i+1], rgba.Pix[i+2])
			}
		}
	}

	// ...and the disc itself must actually be coloured, or the guard above is
	// passing for the wrong reason.
	c := rgba.PixOffset(112, 112)
	if rgba.Pix[c] == 200 && rgba.Pix[c+1] == 120 && rgba.Pix[c+2] == 90 {
		t.Error("the disc centre was not painted at all")
	}
}

func TestRenderHeatmapWithoutAnROIFloorPaintsEverywhere(t *testing.T) {
	const side = 14
	src := discImage(224, 100)
	grid := make([]float32, side*side)
	for i := range grid {
		grid[i] = 1
	}
	opts := DefaultHeatmapOptions()
	opts.ROILumaFloor = 0

	out := RenderHeatmap(src, &Saliency{Grid: grid, Side: side}, opts).(*image.RGBA)
	i := out.PixOffset(0, 0)
	if out.Pix[i] == 0 && out.Pix[i+1] == 0 && out.Pix[i+2] == 0 {
		t.Error("with the floor disabled the corner should be painted; the test above proves nothing otherwise")
	}
}
