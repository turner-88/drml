package infer

import "fmt"

// ExplainMethod names the algorithm that produced a Saliency.
type ExplainMethod string

const (
	// ExplainRollout is forward-only attention rollout (Abnar & Zuidema, 2020):
	// class-agnostic, and on the default checkpoint only weakly tied to the
	// image content. Used when the graph exposes attentions but no gradients.
	ExplainRollout ExplainMethod = "rollout"
	// ExplainGradRelevance is gradient-weighted attention relevance (Chefer et
	// al., ICCV 2021): class-specific, because each layer's attention is
	// weighted by the gradient of the predicted logit with respect to it.
	ExplainGradRelevance ExplainMethod = "grad-relevance"
)

// GradRelevance turns a ViT's attention probabilities and the gradient of the
// predicted-class logit with respect to them into a spatial saliency grid.
//
// This is the transformer explanation of Chefer et al. (ICCV 2021), the ViT
// analogue of Grad-CAM. Where rollout asks "which patches did the CLS token mix
// in?", this asks "which of that mixing actually pushed the winning logit up?":
//
//	Ā_l       = mean_heads( relu( ∇A_l ⊙ A_l ) )
//	R         = I;  R = R + Ā_l @ R   for l = 1..L
//	saliency  = R[CLS, 1:]
//
// The relu is applied per head before the mean, so a head that argues against
// the class cannot cancel one that argues for it. The gradient comes out of the
// ONNX graph itself: export.py bakes the backward pass in, so ORT computes it
// in the same Run as the logits.
//
// attn and grads are both flattened [layers, batch, heads, tokens, tokens] and
// must share shape. mask has the same meaning as in Rollout. When no gradient
// is positive anywhere, R stays the identity, the grid is all zero and
// Concentration is 0: the model had no positive evidence to point at, and the
// renderer draws nothing rather than a normalized fiction.
func GradRelevance(attn, grads []float32, shape []int64, mask []bool) (*Saliency, error) {
	grid, side, err := gradRelevanceRaw(attn, grads, shape)
	if err != nil {
		return nil, err
	}
	if mask != nil && len(mask) != len(grid) {
		return nil, fmt.Errorf("mask has %d cells, want %d", len(mask), len(grid))
	}
	conc := concentration(grid, mask)
	normalizeSaliency(grid, mask)
	return &Saliency{Grid: grid, Side: side, Concentration: conc, Method: ExplainGradRelevance}, nil
}

// gradRelevanceRaw is GradRelevance before masking and normalization: the raw
// CLS row of R over the patch tokens. Exposed for the parity test, which
// compares it against the Python reference in golden.json.
func gradRelevanceRaw(attn, grads []float32, shape []int64) ([]float32, int, error) {
	layers, heads, tokens, side, err := attentionDims(shape, len(attn))
	if err != nil {
		return nil, 0, err
	}
	if len(grads) != len(attn) {
		return nil, 0, fmt.Errorf("attn_grads has %d values, attentions has %d", len(grads), len(attn))
	}

	n := tokens
	r := identity(n)
	layerStride := heads * n * n
	abar := make([]float32, n*n)
	scratch := make([]float32, n*n)
	inv := 1 / float32(heads)

	for l := 0; l < layers; l++ {
		base := l * layerStride
		for i := range abar {
			abar[i] = 0
		}
		for h := 0; h < heads; h++ {
			off := base + h*n*n
			for i := 0; i < n*n; i++ {
				if v := grads[off+i] * attn[off+i]; v > 0 {
					abar[i] += v
				}
			}
		}
		for i := range abar {
			abar[i] *= inv
		}

		// R += Ā @ R: the new layer multiplies on the left, as in rollout.
		matmul(scratch, abar, r, n)
		for i := range r {
			r[i] += scratch[i]
		}
	}

	grid := make([]float32, tokens-1)
	copy(grid, r[1:tokens])
	return grid, side, nil
}

// attentionDims validates a [layers, batch, heads, tokens, tokens] shape
// against the flattened tensor length n and returns its dimensions plus the
// patch-grid side.
func attentionDims(shape []int64, n int) (layers, heads, tokens, side int, err error) {
	if len(shape) != 5 {
		return 0, 0, 0, 0, fmt.Errorf("attentions rank = %d, want 5 [layers,batch,heads,tokens,tokens]", len(shape))
	}
	layers, heads = int(shape[0]), int(shape[2])
	batch := int(shape[1])
	tokens, tokens2 := int(shape[3]), int(shape[4])

	if tokens != tokens2 {
		return 0, 0, 0, 0, fmt.Errorf("attention matrix is %dx%d, want square", tokens, tokens2)
	}
	if batch != 1 {
		return 0, 0, 0, 0, fmt.Errorf("batch = %d, want 1", batch)
	}
	if layers < 1 || heads < 1 || tokens < 2 {
		return 0, 0, 0, 0, fmt.Errorf("degenerate attention shape %v", shape)
	}
	if want := layers * batch * heads * tokens * tokens; n != want {
		return 0, 0, 0, 0, fmt.Errorf("attentions has %d values, want %d", n, want)
	}

	// Patch tokens are everything but the leading CLS token, and they must form
	// a square grid (196 -> 14x14 for ViT-B/16 at 224px).
	side = patchSide(shape)
	if side == 0 {
		return 0, 0, 0, 0, fmt.Errorf("%d patch tokens is not a square grid", tokens-1)
	}
	return layers, heads, tokens, side, nil
}
