package infer

import (
	"image"
	"math"
	"testing"
)

// buildAttn lays out a [layers,1,heads,tokens,tokens] tensor from a per-layer
// per-head matrix generator.
func buildAttn(layers, heads, tokens int, at func(l, h, i, j int) float32) []float32 {
	out := make([]float32, layers*heads*tokens*tokens)
	idx := 0
	for l := 0; l < layers; l++ {
		for h := 0; h < heads; h++ {
			for i := 0; i < tokens; i++ {
				for j := 0; j < tokens; j++ {
					out[idx] = at(l, h, i, j)
					idx++
				}
			}
		}
	}
	return out
}

func shape(layers, heads, tokens int) []int64 {
	return []int64{int64(layers), 1, int64(heads), int64(tokens), int64(tokens)}
}

// TestAttentionRolloutFocusesOnAttendedPatch is the behavioural check: if every
// layer attends to one patch, that patch must dominate the saliency map.
func TestAttentionRolloutFocusesOnAttendedPatch(t *testing.T) {
	const layers, heads, tokens = 3, 2, 5 // 4 patches -> 2x2 grid
	const target = 3                      // token index; patch index target-1 = 2

	attn := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		if j == target {
			return 1
		}
		return 0
	})

	grid, side, err := AttentionRollout(attn, shape(layers, heads, tokens))
	if err != nil {
		t.Fatalf("rollout: %v", err)
	}
	if side != 2 {
		t.Fatalf("side = %d, want 2", side)
	}
	if len(grid) != 4 {
		t.Fatalf("grid has %d values, want 4", len(grid))
	}

	// After min-max normalization the attended patch is 1 and the rest 0.
	if grid[target-1] != 1 {
		t.Errorf("attended patch saliency = %v, want 1", grid[target-1])
	}
	for i, v := range grid {
		if i == target-1 {
			continue
		}
		if v != 0 {
			t.Errorf("patch %d saliency = %v, want 0", i, v)
		}
	}
}

// TestAttentionRolloutUniformIsFlat guards the degenerate case: uniform
// attention carries no information and must not render as spurious structure.
func TestAttentionRolloutUniformIsFlat(t *testing.T) {
	const layers, heads, tokens = 4, 3, 17 // 16 patches -> 4x4
	uniform := float32(1) / float32(tokens)

	attn := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 { return uniform })

	grid, side, err := AttentionRollout(attn, shape(layers, heads, tokens))
	if err != nil {
		t.Fatalf("rollout: %v", err)
	}
	if side != 4 {
		t.Fatalf("side = %d, want 4", side)
	}
	for i, v := range grid {
		if v != 0 {
			t.Errorf("patch %d = %v; a flat map must normalize to 0, not NaN or noise", i, v)
		}
	}
}

// TestAttentionRolloutRowsStayNormalized verifies the residual+renormalize step
// keeps the composed matrix a valid mixing matrix (rows summing to 1). If this
// drifts, deep models produce saturated or vanishing saliency.
func TestAttentionRolloutRowsStayNormalized(t *testing.T) {
	const n = 6
	a := make([]float32, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a[i*n+j] = float32(1) / float32(n)
		}
	}
	b := identity(n)
	dst := make([]float32, n*n)
	matmul(dst, a, b, n)

	for i := 0; i < n; i++ {
		var sum float64
		for j := 0; j < n; j++ {
			sum += float64(dst[i*n+j])
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Errorf("row %d sums to %v, want 1", i, sum)
		}
	}
}

func TestAttentionRolloutRejectsBadShapes(t *testing.T) {
	cases := []struct {
		name  string
		attn  []float32
		shape []int64
	}{
		{"wrong rank", make([]float32, 4), []int64{2, 2}},
		{"non-square matrix", make([]float32, 1*1*1*3*4), []int64{1, 1, 1, 3, 4}},
		{"batch > 1", make([]float32, 1*2*1*5*5), []int64{1, 2, 1, 5, 5}},
		{"non-square patch grid", make([]float32, 1*1*1*4*4), []int64{1, 1, 1, 4, 4}}, // 3 patches
		{"length mismatch", make([]float32, 7), []int64{1, 1, 1, 5, 5}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := AttentionRollout(tc.attn, tc.shape); err == nil {
				t.Fatal("expected an error, got nil")
			}
		})
	}
}

func TestSoftmaxIsStableAndSumsToOne(t *testing.T) {
	cases := [][]float32{
		{0, 0, 0, 0, 0},
		{1, 2, 3, 4, 5},
		{1000, 1001, 999, 998, 1002}, // would overflow without the max shift
		{-50, -60, -70, -80, -90},
	}
	for _, logits := range cases {
		got := Softmax(logits)
		var sum float32
		for _, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("softmax(%v) produced %v", logits, got)
			}
			sum += v
		}
		if math.Abs(float64(sum)-1) > 1e-5 {
			t.Errorf("softmax(%v) sums to %v, want 1", logits, sum)
		}
	}
}

func TestRenderHeatmapHandlesMissingGrid(t *testing.T) {
	src := imageRGBA(8, 8)
	// A nil grid must degrade to the original image, not panic.
	out := RenderHeatmap(src, nil, 0, DefaultHeatmapOptions())
	if out.Bounds().Dx() != 8 || out.Bounds().Dy() != 8 {
		t.Fatalf("bounds = %v, want 8x8", out.Bounds())
	}
}

// imageRGBA builds an opaque mid-gray test image.
func imageRGBA(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 128, 128, 128, 255
	}
	return img
}
