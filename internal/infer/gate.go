package infer

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math"
	"os"

	xdraw "golang.org/x/image/draw"
)

// ErrNotFundus is returned when the gate judges an upload not to be a retinal
// fundus photograph.
//
// The classifier cannot make this call itself: it is a closed-set 5-way softmax
// trained only on fundus images, so its probabilities always sum to 1 and it
// will grade a selfie as confidently as a real scan. Rejection has to happen
// outside the model.
var ErrNotFundus = errors.New("gambar bukan foto fundus retina")

// gateThumbDim is the working resolution for the image heuristics. Every
// feature below is a mean or a variance, which a 64x64 downscale preserves;
// fixing the size here makes the check cost the same for a 12 MB upload as for
// a 200 KB one, which matters because it runs before the inference semaphore.
const gateThumbDim = 64

// Gate reason codes. These are logged, never shown to the user — a rejection
// message that named the failing feature would just teach people how to defeat
// it, and means nothing to a clinician.
const (
	ReasonTooSmall  = "too_small"
	ReasonAspect    = "aspect"
	ReasonNotFundus = "not_fundus"
	ReasonOODEnergy = "ood_energy"
)

// GateConfig mirrors model/gate.json, written by model/gate.py. Thresholds are
// data rather than code for the same reason preprocess.json is: retuning after
// a recalibration run should not need a rebuild.
type GateConfig struct {
	Enabled bool `json:"enabled"`

	// MinDim rejects anything whose shorter side is below this. Fundus cameras
	// do not produce thumbnails.
	MinDim int `json:"min_dim"`
	// MaxAspect rejects long thin images: screenshots, banners, panoramas.
	// A fundus frame is square or close to it.
	MaxAspect float64 `json:"max_aspect"`

	// LumaFloor separates the illuminated disc from the black surround. Pixels
	// at or above it form the ROI that the colour and texture features measure,
	// so that a large black border cannot drag the means toward zero.
	LumaFloor float64 `json:"luma_floor"`

	// Feature weights. Red dominance carries the most because it is the single
	// most discriminative signal; the ROI vote is deliberately soft because a
	// tightly cropped fundus image has no black border to find.
	RedWeight     float64 `json:"red_weight"`
	ROIWeight     float64 `json:"roi_weight"`
	TextureWeight float64 `json:"texture_weight"`

	// ScoreMin is the calibrated pass mark for the weighted score, chosen at
	// 98-99% true-positive rate on real fundus images: rejecting a genuine scan
	// is a far worse error here than admitting a borderline one.
	ScoreMin float64 `json:"score_min"`

	// EnergyEnabled gates the second tier. It ships false until model/gate.py
	// demonstrates that the served checkpoint's logits actually separate
	// in- from out-of-distribution inputs; a dynamically quantized community
	// fine-tune is not guaranteed to.
	EnergyEnabled bool `json:"energy_enabled"`
	// EnergyMax is the rejection threshold for -logsumexp(logits). In-
	// distribution inputs score lower.
	EnergyMax float64 `json:"energy_max"`

	// Calibration records what the thresholds above were measured against.
	// Surfaced in the admin UI so switching the gate off is an informed
	// decision rather than a blind one.
	Calibration *Calibration `json:"calibration,omitempty"`
}

// Calibration is the provenance block model/gate.py writes into gate.json.
type Calibration struct {
	FundusImages      int      `json:"fundus_images"`
	NonFundusImages   int      `json:"non_fundus_images"`
	FundusSources     []string `json:"fundus_sources"`
	NonFundusSources  []string `json:"non_fundus_sources"`
	TargetTPR         float64  `json:"target_tpr"`
	FundusAdmitted    float64  `json:"measured_fundus_admitted"`
	NonFundusAdmitted float64  `json:"measured_non_fundus_admitted"`
	ScoreAUROC        float64  `json:"score_auroc"`
	EnergyAUROC       float64  `json:"energy_auroc"`
	Model             string   `json:"model"`
}

// DefaultGateConfig returns pre-calibration defaults.
//
// They are deliberately permissive: they encode the shape of the decision
// (which features, which direction) but not a tuned threshold. model/gate.py
// overwrites ScoreMin and the energy settings with measured values.
func DefaultGateConfig() *GateConfig {
	return &GateConfig{
		Enabled:       true,
		MinDim:        224,
		MaxAspect:     2.0,
		LumaFloor:     0.12,
		RedWeight:     0.45,
		ROIWeight:     0.35,
		TextureWeight: 0.20,
		ScoreMin:      0.55,
		EnergyEnabled: false,
	}
}

// GateReport is the outcome of one check, including the raw feature values so
// a rejection can be explained in the log without guesswork.
type GateReport struct {
	Passed  bool
	Reason  string
	Metrics map[string]float64
}

func pass(metrics map[string]float64) GateReport {
	return GateReport{Passed: true, Metrics: metrics}
}

func fail(reason string, metrics map[string]float64) GateReport {
	return GateReport{Passed: false, Reason: reason, Metrics: metrics}
}

// LoadGateConfig reads and validates model/gate.json.
//
// A missing file is reported as fs.ErrNotExist so the caller can decide; New
// treats it as "gate disabled" rather than a fatal error, which keeps an
// existing deployment bootable after a binary upgrade that predates its
// calibration run.
func LoadGateConfig(path string) (*GateConfig, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read gate config: %w", err)
	}
	var cfg GateConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse gate config: %w", err)
	}
	if cfg.MinDim < 0 {
		return nil, fmt.Errorf("min_dim must not be negative, got %d", cfg.MinDim)
	}
	if cfg.MaxAspect < 0 {
		return nil, fmt.Errorf("max_aspect must not be negative, got %g", cfg.MaxAspect)
	}
	if cfg.LumaFloor < 0 || cfg.LumaFloor >= 1 {
		return nil, fmt.Errorf("luma_floor must be in [0,1), got %g", cfg.LumaFloor)
	}
	if cfg.ScoreMin < 0 || cfg.ScoreMin > 1 {
		return nil, fmt.Errorf("score_min must be in [0,1], got %g", cfg.ScoreMin)
	}
	if w := cfg.RedWeight + cfg.ROIWeight + cfg.TextureWeight; w <= 0 {
		return nil, fmt.Errorf("feature weights sum to %g, expected > 0", w)
	}
	return &cfg, nil
}

// CheckImage runs the image heuristics. It is cheap enough (a 64x64 downscale
// and one pass over 4096 pixels) to run on every upload before the inference
// semaphore is acquired, so junk never occupies the single forward-pass slot.
//
// A nil or disabled config passes everything.
func (g *GateConfig) CheckImage(img image.Image) GateReport {
	if g == nil || !g.Enabled {
		return pass(nil)
	}

	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	metrics := map[string]float64{"width": float64(w), "height": float64(h)}

	// Hard prefilters. These two are unambiguous — no real fundus capture is
	// tiny or letterboxed — so they short-circuit before any pixel work.
	if short := min(w, h); short < g.MinDim {
		metrics["short_side"] = float64(short)
		return fail(ReasonTooSmall, metrics)
	}
	if w > 0 && h > 0 && g.MaxAspect > 0 {
		aspect := float64(max(w, h)) / float64(min(w, h))
		metrics["aspect"] = aspect
		if aspect > g.MaxAspect {
			return fail(ReasonAspect, metrics)
		}
	}

	f := g.features(img)
	for k, v := range f {
		metrics[k] = v
	}

	// Weighted vote rather than an AND of hard rules: each feature alone has
	// real fundus images that fail it (tight crops have no dark border, very
	// dark scans have little texture), but an image that fails all three is
	// not a retina.
	total := g.RedWeight + g.ROIWeight + g.TextureWeight
	score := (g.RedWeight*f["red_score"] +
		g.ROIWeight*f["roi_score"] +
		g.TextureWeight*f["texture_score"]) / total
	metrics["score"] = score

	if score < g.ScoreMin {
		return fail(ReasonNotFundus, metrics)
	}
	return pass(metrics)
}

// features computes the three scored heuristics plus their raw inputs.
func (g *GateConfig) features(img image.Image) map[string]float64 {
	// ApproxBiLinear, not BiLinear. A kernel interpolator widens its support
	// when downscaling, so a 2048x2048 upload costs ~16.8 ms to reduce; the
	// approximate one costs ~37 us for the same image. That 447x matters
	// because this runs on every upload ahead of the semaphore. It aliases,
	// which would be unacceptable in preprocess.go — there the kernel has to
	// match PIL's for the parity test — but here the outputs are means and
	// variances over the whole thumbnail, and measured against BiLinear the
	// colour ratios agree to 0.2%.
	thumb := image.NewRGBA(image.Rect(0, 0, gateThumbDim, gateThumbDim))
	xdraw.ApproxBiLinear.Scale(thumb, thumb.Bounds(), img, img.Bounds(), xdraw.Src, nil)

	luma := make([]float64, gateThumbDim*gateThumbDim)
	var sumR, sumG, sumB, sumL, sumLL float64
	var n float64

	for y := 0; y < gateThumbDim; y++ {
		for x := 0; x < gateThumbDim; x++ {
			// Read the 8-bit sRGB bytes directly, as preprocess.go does:
			// color.Color would widen to 16-bit alpha-premultiplied values.
			i := thumb.PixOffset(x, y)
			r := float64(thumb.Pix[i]) / 255
			gr := float64(thumb.Pix[i+1]) / 255
			bl := float64(thumb.Pix[i+2]) / 255

			l := 0.299*r + 0.587*gr + 0.114*bl
			luma[y*gateThumbDim+x] = l

			// Colour and texture are measured over the illuminated disc only.
			// Including the black surround would make every bordered image look
			// dark and desaturated regardless of what the retina looks like.
			if l >= g.LumaFloor {
				sumR += r
				sumG += gr
				sumB += bl
				sumL += l
				sumLL += l * l
				n++
			}
		}
	}

	out := map[string]float64{"roi_frac": n / float64(len(luma))}

	// An image with no pixels above the floor is a black frame, not a scan.
	if n == 0 {
		out["red_score"], out["roi_score"], out["texture_score"] = 0, 0, 0
		return out
	}

	meanR, meanG, meanB := sumR/n, sumG/n, sumB/n
	out["mean_r"], out["mean_g"], out["mean_b"] = meanR, meanG, meanB

	// 1. Red dominance. The retina is lit through blood, so R strongly leads G
	// leads B. Skin sits near 1.2, screenshots and documents near 1.0, foliage
	// and sky below 1.0. The upper arm rejects the other direction: a pure-red
	// synthetic fill is not a fundus either.
	redRatio := meanR / math.Max(meanG, 1e-6)
	out["red_ratio"] = redRatio
	redScore := plateau(redRatio, 1.15, 1.45, 5.0, 8.0)
	if meanR <= meanB {
		// The R > B ordering is not optional; without it a magenta or grey cast
		// could still land inside the ratio window.
		redScore = 0
	}
	out["red_score"] = redScore

	// 2. Circular ROI. A fundus frame is a bright disc on a black field, so the
	// corners are far darker than the centre. Soft by design: some datasets
	// ship images cropped tight to the disc, which score near zero here and
	// have to be carried by the other two features.
	//
	// Expressed as Michelson contrast rather than a centre/corner ratio: on a
	// true black border the ratio divides by the epsilon guard and reaches six
	// figures, which tells you nothing a bounded number does not and makes
	// every log line and metric dump unreadable. The two are monotonically
	// related, so the arms below are the exact images of the ratio arms 1.2
	// and 3.0 and the decision is unchanged.
	corner := (blockMean(luma, 0, 0, 12) +
		blockMean(luma, gateThumbDim-12, 0, 12) +
		blockMean(luma, 0, gateThumbDim-12, 12) +
		blockMean(luma, gateThumbDim-12, gateThumbDim-12, 12)) / 4
	centre := blockMean(luma, (gateThumbDim-20)/2, (gateThumbDim-20)/2, 20)
	contrast := (centre - corner) / math.Max(centre+corner, 1e-6)
	out["corner_luma"], out["centre_luma"], out["centre_contrast"] = corner, centre, contrast
	out["roi_score"] = plateau(contrast, 0.0909, 0.5, math.Inf(1), math.Inf(1))

	// 3. Texture, as a band rather than a floor.
	//
	// A one-sided "more texture is more fundus" ramp measured AUROC 0.46 — it
	// was voting for the wrong class. Retinal images are smooth: measured over
	// 220 of them the ROI standard deviation runs 0.037 (p1) to 0.127 (p99),
	// while 240 non-fundus photos spread from 0.022 (p5) to 0.272 (p95),
	// straddling that band on both sides. So the feature has to reject the
	// flat images it was always meant to catch AND the busy ones that a floor
	// alone rewarded.
	variance := math.Max(sumLL/n-(sumL/n)*(sumL/n), 0)
	stddev := math.Sqrt(variance)
	out["roi_stddev"] = stddev
	out["texture_score"] = plateau(stddev, 0.015, 0.030, 0.13, 0.22)

	// Blur is recorded but not scored. Genuinely out-of-focus fundus photos are
	// clinically common, and rejecting one is worse than grading it — the low
	// confidence warning already covers that case.
	out["laplacian_var"] = laplacianVar(luma)

	return out
}

// plateau maps x onto [0,1]: zero at or below lo0, rising linearly to one at
// lo1, flat until hi1, then falling back to zero at hi0. Pass +Inf for hi1 and
// hi0 for a one-sided ramp.
func plateau(x, lo0, lo1, hi1, hi0 float64) float64 {
	switch {
	case x <= lo0:
		return 0
	case x < lo1:
		return (x - lo0) / (lo1 - lo0)
	case x <= hi1:
		return 1
	case x < hi0:
		return (hi0 - x) / (hi0 - hi1)
	default:
		return 0
	}
}

// blockMean averages a size x size block of the luma plane at (x0,y0).
func blockMean(luma []float64, x0, y0, size int) float64 {
	var sum float64
	for y := y0; y < y0+size; y++ {
		for x := x0; x < x0+size; x++ {
			sum += luma[y*gateThumbDim+x]
		}
	}
	return sum / float64(size*size)
}

// laplacianVar is the variance of the 4-neighbour Laplacian, the standard
// cheap focus measure. Interior pixels only, so no edge handling is needed.
func laplacianVar(luma []float64) float64 {
	var sum, sumSq, n float64
	for y := 1; y < gateThumbDim-1; y++ {
		for x := 1; x < gateThumbDim-1; x++ {
			i := y*gateThumbDim + x
			lap := 4*luma[i] - luma[i-1] - luma[i+1] -
				luma[i-gateThumbDim] - luma[i+gateThumbDim]
			sum += lap
			sumSq += lap * lap
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return math.Max(sumSq/n-(sum/n)*(sum/n), 0)
}

// CheckLogits is the second tier: energy-based out-of-distribution detection
// (Liu et al., NeurIPS 2020). E(x) = -logsumexp(logits), where in-distribution
// inputs score lower because the model commits more total evidence to them.
//
// It rides on logits the forward pass has already produced, so it costs a
// handful of float operations and no memory. Softmax cannot substitute for it:
// normalizing away the logit magnitudes is exactly what destroys the signal.
func (g *GateConfig) CheckLogits(logits []float32) GateReport {
	if g == nil || !g.Enabled || !g.EnergyEnabled {
		return pass(nil)
	}
	energy := EnergyScore(logits)
	metrics := map[string]float64{"energy": energy}
	if energy > g.EnergyMax {
		return fail(ReasonOODEnergy, metrics)
	}
	return pass(metrics)
}

// EnergyScore returns -logsumexp(logits), computed with the max shifted out for
// the same stability reason Softmax does it.
func EnergyScore(logits []float32) float64 {
	if len(logits) == 0 {
		return 0
	}
	maxLogit := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > maxLogit {
			maxLogit = float64(v)
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v) - maxLogit)
	}
	return -(maxLogit + math.Log(sum))
}
