package infer

import (
	"fmt"
	"math"
)

// AttentionRollout turns a ViT's attention tensors into a spatial saliency grid.
//
// Grad-CAM is the usual choice for this, but it needs a backward pass, and
// ONNX Runtime is inference-only. Attention rollout (Abnar & Zuidema, 2020) is
// forward-only and native to transformers: it composes each layer's attention
// with a residual term and multiplies through the depth of the network, so the
// CLS row of the product says how much each input patch contributed.
//
//	Â_l      = 0.5 * mean_heads(A_l) + 0.5 * I     (residual stream)
//	Â_l      = rowNormalize(Â_l)
//	rollout  = Â_L @ ... @ Â_1
//	saliency = rollout[CLS, 1:]                    (drop the CLS->CLS term)
//
// attn is the flattened [layers, batch, heads, tokens, tokens] output and shape
// describes it. The returned grid is side*side values in [0,1], row-major.
func AttentionRollout(attn []float32, shape []int64) ([]float32, int, error) {
	if len(shape) != 5 {
		return nil, 0, fmt.Errorf("attentions rank = %d, want 5 [layers,batch,heads,tokens,tokens]", len(shape))
	}
	layers, batch, heads := int(shape[0]), int(shape[1]), int(shape[2])
	tokens, tokens2 := int(shape[3]), int(shape[4])

	if tokens != tokens2 {
		return nil, 0, fmt.Errorf("attention matrix is %dx%d, want square", tokens, tokens2)
	}
	if batch != 1 {
		return nil, 0, fmt.Errorf("batch = %d, want 1", batch)
	}
	if layers < 1 || heads < 1 || tokens < 2 {
		return nil, 0, fmt.Errorf("degenerate attention shape %v", shape)
	}
	if want := layers * batch * heads * tokens * tokens; len(attn) != want {
		return nil, 0, fmt.Errorf("attentions has %d values, want %d", len(attn), want)
	}

	// Patch tokens are everything but the leading CLS token, and they must form
	// a square grid (196 -> 14x14 for ViT-B/16 at 224px).
	patches := tokens - 1
	side := int(math.Round(math.Sqrt(float64(patches))))
	if side*side != patches {
		return nil, 0, fmt.Errorf("%d patch tokens is not a square grid", patches)
	}

	n := tokens
	rollout := identity(n)
	layerStride := heads * n * n
	avg := make([]float32, n*n)
	scratch := make([]float32, n*n)

	for l := 0; l < layers; l++ {
		base := l * layerStride

		// Average over heads.
		for i := range avg {
			avg[i] = 0
		}
		for h := 0; h < heads; h++ {
			off := base + h*n*n
			for i := 0; i < n*n; i++ {
				avg[i] += attn[off+i]
			}
		}
		inv := 1 / float32(heads)
		for i := range avg {
			avg[i] *= inv
		}

		// Add the residual connection, then renormalize each row to sum to 1 so
		// the product stays a valid mixing matrix.
		for i := 0; i < n; i++ {
			var sum float32
			for j := 0; j < n; j++ {
				v := 0.5 * avg[i*n+j]
				if i == j {
					v += 0.5
				}
				avg[i*n+j] = v
				sum += v
			}
			if sum > 0 {
				for j := 0; j < n; j++ {
					avg[i*n+j] /= sum
				}
			}
		}

		matmul(scratch, avg, rollout, n)
		rollout, scratch = scratch, rollout
	}

	// Row 0 is CLS; columns 1.. are the patches it drew from.
	grid := make([]float32, patches)
	copy(grid, rollout[1:tokens])

	normalizeInPlace(grid)
	return grid, side, nil
}

// identity returns an n x n identity matrix in row-major order.
func identity(n int) []float32 {
	m := make([]float32, n*n)
	for i := 0; i < n; i++ {
		m[i*n+i] = 1
	}
	return m
}

// matmul computes dst = a @ b for n x n row-major matrices. dst must not alias
// a or b.
func matmul(dst, a, b []float32, n int) {
	for i := 0; i < n; i++ {
		row := dst[i*n : (i+1)*n]
		for j := range row {
			row[j] = 0
		}
		for k := 0; k < n; k++ {
			av := a[i*n+k]
			if av == 0 {
				continue
			}
			bRow := b[k*n : (k+1)*n]
			for j := 0; j < n; j++ {
				row[j] += av * bRow[j]
			}
		}
	}
}

// flatRelTolerance is the relative spread below which a saliency map is treated
// as carrying no signal.
//
// Min-max normalization rescales whatever spread it finds to the full [0,1]
// range, so a map whose values differ only in their last floating-point bits
// would be stretched into a vivid, entirely fictitious heatmap. That is the
// worst failure mode for an explainability feature — it looks confident and
// means nothing. Comparing the spread against the magnitude of the values
// keeps genuine low-contrast maps while rejecting pure accumulation noise.
const flatRelTolerance = 1e-6

// normalizeInPlace min-max scales v into [0,1]. A map that is flat, or flat to
// within float noise, becomes all zeros so it renders as "no signal" rather
// than as invented structure.
func normalizeInPlace(v []float32) {
	if len(v) == 0 {
		return
	}
	min, max := v[0], v[0]
	for _, x := range v[1:] {
		if x < min {
			min = x
		}
		if x > max {
			max = x
		}
	}

	span := float64(max - min)
	scale := math.Max(math.Abs(float64(max)), math.Abs(float64(min)))
	if span <= flatRelTolerance*scale || span <= math.SmallestNonzeroFloat32 {
		for i := range v {
			v[i] = 0
		}
		return
	}

	for i := range v {
		v[i] = float32((float64(v[i]) - float64(min)) / span)
	}
}

// exp wraps math.Exp so engine.go's Softmax stays free of float64 conversions
// at the call site.
func exp(x float64) float64 { return math.Exp(x) }
