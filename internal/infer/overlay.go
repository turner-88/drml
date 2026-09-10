package infer

import (
	"image"

	xdraw "golang.org/x/image/draw"
)

// HeatmapOptions tunes the rendered overlay.
type HeatmapOptions struct {
	// Alpha is the heatmap's opacity over the source image, 0..1.
	Alpha float64
	// Threshold suppresses saliency below this level so the overlay highlights
	// the strongest regions instead of tinting the whole fundus.
	Threshold float64

	// MinConcentration and FullConcentration ramp Alpha by how much structure
	// the saliency map actually carries (see Saliency.Concentration). At or
	// below MinConcentration the overlay fades out entirely; at or above
	// FullConcentration the full Alpha applies.
	//
	// Without this, min-max style normalization guarantees a saturated peak on
	// every scan however diffuse the underlying attention was — the map always
	// looks equally confident, which is precisely the failure this feature
	// cannot afford. Leave both at 0 to disable the ramp.
	MinConcentration  float64
	FullConcentration float64

	// ROILumaFloor stops the overlay painting pixels darker than this, the
	// same disc/surround threshold FundusMask applies to the grid.
	//
	// Both are needed, for different reasons. The grid mask keeps the black
	// surround out of the normalization statistics; this one keeps it out of
	// the picture. Zeroing a boundary cell is not enough on its own, because
	// upsampling interpolates between that zero and its lit neighbour, and the
	// ramp between them crosses Threshold somewhere out in the black — which
	// is exactly how a 14-cell grid ends up painting hot spots in the corners.
	// Set to 0 to paint everywhere.
	ROILumaFloor float64
}

// DefaultHeatmapOptions are tuned for fundus images, where the informative
// signal is a few small lesions rather than a broad region.
//
// The concentration ramp ships disabled: the thresholds have to come from a
// measured distribution over real scans, not from a guess. See
// TestSaliencyConcentrationSweep.
func DefaultHeatmapOptions() HeatmapOptions {
	return HeatmapOptions{
		Alpha:        0.45,
		Threshold:    0.25,
		ROILumaFloor: DefaultGateConfig().LumaFloor,
	}
}

// EffectiveAlpha is Alpha scaled down for a map whose saliency is too diffuse
// to be worth drawing at full strength.
func (o HeatmapOptions) EffectiveAlpha(concentration float64) float64 {
	if o.FullConcentration <= o.MinConcentration {
		return o.Alpha // ramp disabled
	}
	t := (concentration - o.MinConcentration) / (o.FullConcentration - o.MinConcentration)
	return o.Alpha * clamp01(t)
}

// maskBlock is how many source samples back each mask cell. The two-stage
// reduction — one resample to side*maskBlock, then an exact box average — keeps
// the cheap ApproxBiLinear scaler (see gate.go's note on why) from aliasing a
// 14x14 grid onto individual pixels of a multi-megapixel upload.
const maskBlock = 8

// minROIFraction is the smallest share of cells a mask may keep and still be
// trusted. A fundus cropped tight to the disc legitimately masks nothing; a
// mask that rejects almost everything means the luma heuristic failed, and
// falling back to no mask is safer than blanking the map.
const minROIFraction = 0.15

// FundusMask marks the side x side grid cells that fall on the illuminated
// retinal disc rather than the black surround.
//
// Attention paid to the surround is not weak evidence about the retina, it is
// no evidence: those pixels carry no retinal information at all, so painting
// saliency there can only mislead. lumaFloor is the same threshold the
// non-fundus gate uses to separate disc from border (GateConfig.LumaFloor),
// calibrated on real fundus images.
//
// It returns nil when masking should not be applied, which callers treat as
// "keep every cell".
func FundusMask(img image.Image, side int, lumaFloor float64) []bool {
	if img == nil || side <= 0 || lumaFloor <= 0 {
		return nil
	}

	dim := side * maskBlock
	thumb := image.NewRGBA(image.Rect(0, 0, dim, dim))
	xdraw.ApproxBiLinear.Scale(thumb, thumb.Bounds(), img, img.Bounds(), xdraw.Src, nil)

	mask := make([]bool, side*side)
	kept := 0
	for cy := 0; cy < side; cy++ {
		for cx := 0; cx < side; cx++ {
			var sum float64
			for y := cy * maskBlock; y < (cy+1)*maskBlock; y++ {
				for x := cx * maskBlock; x < (cx+1)*maskBlock; x++ {
					// Read the 8-bit sRGB bytes directly, as gate.go does:
					// color.Color would widen to alpha-premultiplied 16-bit.
					i := thumb.PixOffset(x, y)
					r := float64(thumb.Pix[i]) / 255
					g := float64(thumb.Pix[i+1]) / 255
					b := float64(thumb.Pix[i+2]) / 255
					sum += 0.299*r + 0.587*g + 0.114*b
				}
			}
			if sum/float64(maskBlock*maskBlock) >= lumaFloor {
				mask[cy*side+cx] = true
				kept++
			}
		}
	}

	if float64(kept) < minROIFraction*float64(len(mask)) {
		return nil
	}
	return mask
}

// RenderHeatmap blends a saliency map over src and returns a new image.
//
// The grid (Side x Side, values in [0,1]) is upscaled to src's dimensions with
// a smooth filter, so the 14x14 attention resolution reads as soft regions
// rather than visible blocks.
func RenderHeatmap(src image.Image, sal *Saliency, opts HeatmapOptions) image.Image {
	b := src.Bounds()
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	xdraw.Draw(out, out.Bounds(), src, b.Min, xdraw.Src)

	if sal == nil || sal.Side <= 0 || len(sal.Grid) != sal.Side*sal.Side {
		return out // no usable saliency; return the plain image
	}
	alpha := opts.EffectiveAlpha(sal.Concentration)
	if alpha <= 0 {
		return out // too diffuse to draw honestly
	}

	// Upscale the grid as a grayscale image so the resampler does the
	// interpolation, then read it back per pixel.
	small := image.NewGray(image.Rect(0, 0, sal.Side, sal.Side))
	for i, v := range sal.Grid {
		small.Pix[i] = uint8(clamp01(float64(v)) * 255)
	}
	big := image.NewGray(out.Bounds())
	xdraw.CatmullRom.Scale(big, big.Bounds(), small, small.Bounds(), xdraw.Src, nil)

	for y := 0; y < out.Bounds().Dy(); y++ {
		for x := 0; x < out.Bounds().Dx(); x++ {
			s := float64(big.GrayAt(x, y).Y) / 255

			// Ramp in above the threshold so there is no hard edge where the
			// overlay starts.
			if s <= opts.Threshold {
				continue
			}
			w := (s - opts.Threshold) / (1 - opts.Threshold) * alpha

			// out still holds the untouched source here: every pixel is
			// visited once, and read before it is written.
			i := out.PixOffset(x, y)
			if roi := roiWeight(out.Pix[i], out.Pix[i+1], out.Pix[i+2], opts.ROILumaFloor); roi < 1 {
				if roi <= 0 {
					continue
				}
				w *= roi
			}

			hr, hg, hb := jet(s)
			out.Pix[i+0] = blend(out.Pix[i+0], hr, w)
			out.Pix[i+1] = blend(out.Pix[i+1], hg, w)
			out.Pix[i+2] = blend(out.Pix[i+2], hb, w)
		}
	}
	return out
}

// roiWeight is 0 on the black surround, 1 well inside the illuminated disc,
// and ramps between over one floor's width so the overlay fades out at the
// retina's edge instead of ending in a hard, jagged line traced by JPEG noise.
func roiWeight(r, g, b uint8, floor float64) float64 {
	if floor <= 0 {
		return 1
	}
	luma := (0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 255
	return clamp01((luma - floor) / floor)
}

// blend mixes base toward top by w in [0,1].
func blend(base, top uint8, w float64) uint8 {
	v := float64(base)*(1-w) + float64(top)*w
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return uint8(v + 0.5)
}

// jet maps 0..1 to a blue->cyan->yellow->red ramp, the convention clinicians
// expect from saliency overlays.
func jet(t float64) (r, g, b uint8) {
	t = clamp01(t)
	switch {
	case t < 0.25: // blue -> cyan
		f := t / 0.25
		return 0, uint8(255 * f), 255
	case t < 0.5: // cyan -> green
		f := (t - 0.25) / 0.25
		return 0, 255, uint8(255 * (1 - f))
	case t < 0.75: // green -> yellow
		f := (t - 0.5) / 0.25
		return uint8(255 * f), 255, 0
	default: // yellow -> red
		f := (t - 0.75) / 0.25
		return 255, uint8(255 * (1 - f)), 0
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
