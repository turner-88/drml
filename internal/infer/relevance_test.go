package infer

import (
	"strings"
	"testing"
)

// uniformAttn is an attention tensor where every token attends to every token
// equally, so the map is decided entirely by the gradients.
func uniformAttn(layers, heads, tokens int) []float32 {
	return buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		return 1 / float32(tokens)
	})
}

// TestGradRelevanceFocusesOnPositiveGradientPatch: with uniform attention, the
// one patch whose gradient is positive must be the whole map.
func TestGradRelevanceFocusesOnPositiveGradientPatch(t *testing.T) {
	const layers, heads, tokens = 3, 2, 5
	const target = 3 // token index; patch index target-1 = 2

	attn := uniformAttn(layers, heads, tokens)
	grads := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		if j == target {
			return 1
		}
		return 0
	})

	sal, err := GradRelevance(attn, grads, shape(layers, heads, tokens), nil)
	if err != nil {
		t.Fatalf("relevance: %v", err)
	}
	if sal.Side != 2 || len(sal.Grid) != 4 {
		t.Fatalf("side=%d len=%d, want 2 and 4", sal.Side, len(sal.Grid))
	}
	if sal.Method != ExplainGradRelevance {
		t.Errorf("method = %q, want %q", sal.Method, ExplainGradRelevance)
	}
	for i, v := range sal.Grid {
		want := float32(0)
		if i == target-1 {
			want = 1
		}
		if v != want {
			t.Errorf("patch %d = %v, want %v", i, v, want)
		}
	}
}

// TestGradRelevanceIsClassSpecificNotAttentionDriven pins the property that
// separates this method from rollout: a patch the model attends to heavily but
// whose contribution has zero gradient must score nothing, while a lightly
// attended patch with a positive gradient carries the map.
func TestGradRelevanceIsClassSpecificNotAttentionDriven(t *testing.T) {
	const layers, heads, tokens = 3, 2, 5
	const attended, useful = 1, 4 // token indices

	attn := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		switch j {
		case attended:
			return 0.9
		case useful:
			return 0.1
		}
		return 0
	})
	grads := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		if j == useful {
			return 1
		}
		return 0
	})

	sal, err := GradRelevance(attn, grads, shape(layers, heads, tokens), nil)
	if err != nil {
		t.Fatalf("relevance: %v", err)
	}
	if got := sal.Grid[useful-1]; got != 1 {
		t.Errorf("useful patch = %v, want 1", got)
	}
	if got := sal.Grid[attended-1]; got != 0 {
		t.Errorf("attended-but-useless patch = %v, want 0", got)
	}

	// Rollout, by contrast, follows the attention.
	roll, err := Rollout(attn, shape(layers, heads, tokens), nil)
	if err != nil {
		t.Fatalf("rollout: %v", err)
	}
	if roll.Grid[attended-1] <= roll.Grid[useful-1] {
		t.Errorf("rollout should favour the attended patch: attended=%v useful=%v",
			roll.Grid[attended-1], roll.Grid[useful-1])
	}
}

// TestGradRelevanceClipsNegativeGradients: evidence against the class is
// discarded, and a map with no positive evidence at all is empty, not
// normalized into a fiction.
func TestGradRelevanceClipsNegativeGradients(t *testing.T) {
	const layers, heads, tokens = 2, 2, 5
	const against, for_ = 1, 3

	attn := uniformAttn(layers, heads, tokens)
	grads := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		switch j {
		case against:
			return -5
		case for_:
			return 1
		}
		return 0
	})
	sal, err := GradRelevance(attn, grads, shape(layers, heads, tokens), nil)
	if err != nil {
		t.Fatalf("relevance: %v", err)
	}
	if sal.Grid[against-1] != 0 {
		t.Errorf("negative-gradient patch = %v, want 0", sal.Grid[against-1])
	}
	if sal.Grid[for_-1] != 1 {
		t.Errorf("positive-gradient patch = %v, want 1", sal.Grid[for_-1])
	}

	allNeg := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 { return -1 })
	sal, err = GradRelevance(attn, allNeg, shape(layers, heads, tokens), nil)
	if err != nil {
		t.Fatalf("relevance: %v", err)
	}
	for i, v := range sal.Grid {
		if v != 0 {
			t.Errorf("all-negative gradients: patch %d = %v, want 0", i, v)
		}
	}
	if sal.Concentration != 0 {
		t.Errorf("all-negative gradients: concentration = %v, want 0", sal.Concentration)
	}
}

// TestGradRelevanceClipsPerHeadBeforeAveraging pins the order of relu and the
// head mean: one head arguing for a patch survives another arguing harder
// against it. Averaging first would cancel it to nothing.
func TestGradRelevanceClipsPerHeadBeforeAveraging(t *testing.T) {
	const layers, heads, tokens = 1, 2, 5
	const target = 2

	attn := uniformAttn(layers, heads, tokens)
	grads := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		if j != target {
			return 0
		}
		if h == 0 {
			return 1
		}
		return -3
	})
	raw, _, err := gradRelevanceRaw(attn, grads, shape(layers, heads, tokens))
	if err != nil {
		t.Fatalf("relevance: %v", err)
	}
	if raw[target-1] <= 0 {
		t.Errorf("patch %d raw relevance = %v, want > 0 (relu must precede the head mean)", target-1, raw[target-1])
	}
	for i, v := range raw {
		if i != target-1 && v != 0 {
			t.Errorf("patch %d raw relevance = %v, want 0", i, v)
		}
	}
}

// TestGradRelevanceMaskZeroesTheSurround mirrors the rollout mask test: cells
// outside the ROI are zero even when their gradient is the largest.
func TestGradRelevanceMaskZeroesTheSurround(t *testing.T) {
	const layers, heads, tokens = 2, 1, 5
	attn := uniformAttn(layers, heads, tokens)
	// Patch 0 (token 1) has the largest gradient but is outside the ROI.
	grads := buildAttn(layers, heads, tokens, func(l, h, i, j int) float32 {
		switch j {
		case 1:
			return 10
		case 2:
			return 2
		case 3:
			return 1
		}
		return 0
	})
	mask := []bool{false, true, true, true}
	sal, err := GradRelevance(attn, grads, shape(layers, heads, tokens), mask)
	if err != nil {
		t.Fatalf("relevance: %v", err)
	}
	if sal.Grid[0] != 0 {
		t.Errorf("masked-out patch = %v, want 0", sal.Grid[0])
	}
	if sal.Grid[1] != 1 {
		t.Errorf("strongest in-ROI patch = %v, want 1 (normalization must ignore the surround)", sal.Grid[1])
	}
}

func TestGradRelevanceRejectsBadShapes(t *testing.T) {
	const layers, heads, tokens = 2, 2, 5
	good := shape(layers, heads, tokens)
	attn := uniformAttn(layers, heads, tokens)

	cases := []struct {
		name  string
		attn  []float32
		grads []float32
		shape []int64
		want  string
	}{
		{"rank", attn, attn, []int64{1, 2, 3}, "rank"},
		{"batch", attn, attn, []int64{layers, 2, heads, tokens, tokens}, "batch"},
		{"square", attn, attn, []int64{layers, 1, heads, tokens, tokens + 1}, "square"},
		{"length", attn[:10], attn[:10], good, "values"},
		{"grid", uniformAttn(layers, heads, 4), uniformAttn(layers, heads, 4), shape(layers, heads, 4), "square grid"},
		{"grads length", attn, attn[:len(attn)-1], good, "attn_grads"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := GradRelevance(c.attn, c.grads, c.shape, nil)
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not mention %q", err, c.want)
			}
		})
	}

	if _, err := GradRelevance(attn, attn, good, []bool{true}); err == nil {
		t.Error("expected a mask-size error")
	}
}

func TestRolloutSetsMethod(t *testing.T) {
	const layers, heads, tokens = 1, 1, 5
	sal, err := Rollout(uniformAttn(layers, heads, tokens), shape(layers, heads, tokens), nil)
	if err != nil {
		t.Fatalf("rollout: %v", err)
	}
	if sal.Method != ExplainRollout {
		t.Errorf("method = %q, want %q", sal.Method, ExplainRollout)
	}
}
