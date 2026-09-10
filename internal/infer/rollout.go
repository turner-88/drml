package infer

import (
	"fmt"
	"math"
	"sort"
)

// Saliency is one image's spatial explanation: a square grid of per-patch
// scores plus the statistic that says how much to trust it.
type Saliency struct {
	// Grid holds Side*Side values in [0,1], row-major. Cells outside the
	// fundus ROI, when a mask was supplied, are exactly 0.
	Grid []float32
	Side int
	// Concentration is p98/mean of the raw saliency over the ROI. Uniform
	// attention scores 1.0 and a peaked map scores higher, so it measures how
	// much structure the map actually carries.
	//
	// It has to be computed on the raw values: normalization is an affine map
	// and destroys the ratio.
	Concentration float64
}

// Rollout turns a ViT's attention tensors into a spatial saliency grid.
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
// describes it. mask, when non-nil, must have one entry per patch and marks the
// cells that lie inside the illuminated fundus disc; see FundusMask. Everything
// outside it is zeroed and excluded from the normalization statistics, because
// saliency on the black surround is not weak evidence, it is no evidence.
func Rollout(attn []float32, shape []int64, mask []bool) (*Saliency, error) {
	if len(shape) != 5 {
		return nil, fmt.Errorf("attentions rank = %d, want 5 [layers,batch,heads,tokens,tokens]", len(shape))
	}
	layers, batch, heads := int(shape[0]), int(shape[1]), int(shape[2])
	tokens, tokens2 := int(shape[3]), int(shape[4])

	if tokens != tokens2 {
		return nil, fmt.Errorf("attention matrix is %dx%d, want square", tokens, tokens2)
	}
	if batch != 1 {
		return nil, fmt.Errorf("batch = %d, want 1", batch)
	}
	if layers < 1 || heads < 1 || tokens < 2 {
		return nil, fmt.Errorf("degenerate attention shape %v", shape)
	}
	if want := layers * batch * heads * tokens * tokens; len(attn) != want {
		return nil, fmt.Errorf("attentions has %d values, want %d", len(attn), want)
	}

	// Patch tokens are everything but the leading CLS token, and they must form
	// a square grid (196 -> 14x14 for ViT-B/16 at 224px).
	patches := tokens - 1
	side := patchSide(shape)
	if side == 0 {
		return nil, fmt.Errorf("%d patch tokens is not a square grid", patches)
	}
	if mask != nil && len(mask) != patches {
		return nil, fmt.Errorf("mask has %d cells, want %d", len(mask), patches)
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

	conc := concentration(grid, mask)
	normalizeSaliency(grid, mask)
	return &Saliency{Grid: grid, Side: side, Concentration: conc}, nil
}

// AttentionRollout is Rollout without an ROI mask, returning the bare grid.
func AttentionRollout(attn []float32, shape []int64) ([]float32, int, error) {
	sal, err := Rollout(attn, shape, nil)
	if err != nil {
		return nil, 0, err
	}
	return sal.Grid, sal.Side, nil
}

// patchSide reports the grid side an attention tensor's shape implies, or 0 if
// it does not describe a square patch grid.
//
// Exposed separately from Rollout because the ROI mask has to be sized — and so
// built — before the rollout runs, so that its normalization can exclude the
// masked-out cells.
func patchSide(shape []int64) int {
	if len(shape) != 5 {
		return 0
	}
	patches := int(shape[3]) - 1
	if patches < 1 {
		return 0
	}
	side := int(math.Round(math.Sqrt(float64(patches))))
	if side*side != patches {
		return 0
	}
	return side
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
// Normalization rescales whatever spread it finds to the full [0,1] range, so a
// map whose values differ only in their last floating-point bits would be
// stretched into a vivid, entirely fictitious heatmap. That is the worst
// failure mode for an explainability feature — it looks confident and means
// nothing. Comparing the spread against the magnitude of the values keeps
// genuine low-contrast maps while rejecting pure accumulation noise.
const flatRelTolerance = 1e-6

// Percentile bounds for normalization. Clipping instead of taking the raw
// min and max stops one outlier patch — which on this checkpoint is often a
// border artefact rather than a lesion — from compressing everything else into
// the bottom of the colour ramp.
const (
	saliencyLoPct = 2.0
	saliencyHiPct = 98.0
)

// concentration reports p98/mean of v over the masked-in cells. Uniform
// saliency scores 1.0; the more the map concentrates on a few patches, the
// higher it goes. Rollout values are non-negative, so the ratio is well defined
// whenever the mean is positive.
func concentration(v []float32, mask []bool) float64 {
	in := selectMasked(v, mask)
	if len(in) == 0 {
		return 0
	}
	var sum float64
	for _, x := range in {
		sum += float64(x)
	}
	mean := sum / float64(len(in))
	if mean <= 0 {
		return 0
	}
	sort.Float64s(in)
	return percentileSorted(in, saliencyHiPct) / mean
}

// normalizeSaliency clips v to its [p2, p98] range over the masked-in cells and
// rescales that band to [0,1]. Masked-out cells become 0, as does a map that is
// flat to within float noise, so it renders as "no signal" rather than as
// invented structure.
func normalizeSaliency(v []float32, mask []bool) {
	if len(v) == 0 {
		return
	}
	in := selectMasked(v, mask)
	if len(in) == 0 {
		for i := range v {
			v[i] = 0
		}
		return
	}
	sort.Float64s(in)
	lo := percentileSorted(in, saliencyLoPct)
	hi := percentileSorted(in, saliencyHiPct)

	span := hi - lo
	scale := math.Max(math.Abs(hi), math.Abs(lo))
	if span <= flatRelTolerance*scale || span <= math.SmallestNonzeroFloat32 {
		for i := range v {
			v[i] = 0
		}
		return
	}

	for i := range v {
		if mask != nil && !mask[i] {
			v[i] = 0
			continue
		}
		v[i] = float32(clamp01((float64(v[i]) - lo) / span))
	}
}

// selectMasked copies the masked-in values of v into a new float64 slice. A nil
// mask selects everything.
func selectMasked(v []float32, mask []bool) []float64 {
	out := make([]float64, 0, len(v))
	for i, x := range v {
		if mask == nil || mask[i] {
			out = append(out, float64(x))
		}
	}
	return out
}

// percentileSorted returns the p-th percentile of an ascending slice, linearly
// interpolating between neighbours.
func percentileSorted(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	pos := p / 100 * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo < 0 {
		lo, hi = 0, 0
	}
	if hi > len(sorted)-1 {
		lo, hi = len(sorted)-1, len(sorted)-1
	}
	if lo == hi {
		return sorted[lo]
	}
	f := pos - float64(lo)
	return sorted[lo]*(1-f) + sorted[hi]*f
}

// exp wraps math.Exp so engine.go's Softmax stays free of float64 conversions
// at the call site.
func exp(x float64) float64 { return math.Exp(x) }
