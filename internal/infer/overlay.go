package infer

import (
	"image"
	"image/color"

	xdraw "golang.org/x/image/draw"
)

// HeatmapOptions tunes the rendered overlay.
type HeatmapOptions struct {
	// Alpha is the heatmap's opacity over the source image, 0..1.
	Alpha float64
	// Threshold suppresses saliency below this level so the overlay highlights
	// the strongest regions instead of tinting the whole fundus.
	Threshold float64
}

// DefaultHeatmapOptions are tuned for fundus images, where the informative
// signal is a few small lesions rather than a broad region.
func DefaultHeatmapOptions() HeatmapOptions {
	return HeatmapOptions{Alpha: 0.45, Threshold: 0.25}
}

// RenderHeatmap blends a saliency grid over src and returns a new image.
//
// The grid (side x side, values in [0,1]) is upscaled to src's dimensions with
// a smooth filter, so the 14x14 attention resolution reads as soft regions
// rather than visible blocks.
func RenderHeatmap(src image.Image, grid []float32, side int, opts HeatmapOptions) image.Image {
	b := src.Bounds()
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	xdraw.Draw(out, out.Bounds(), src, b.Min, xdraw.Src)

	if side <= 0 || len(grid) != side*side {
		return out // no usable saliency; return the plain image
	}

	// Upscale the grid as a grayscale image so the resampler does the
	// interpolation, then read it back per pixel.
	small := image.NewGray(image.Rect(0, 0, side, side))
	for i, v := range grid {
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
			w := (s - opts.Threshold) / (1 - opts.Threshold) * opts.Alpha

			hr, hg, hb := jet(s)
			i := out.PixOffset(x, y)
			out.Pix[i+0] = blend(out.Pix[i+0], hr, w)
			out.Pix[i+1] = blend(out.Pix[i+1], hg, w)
			out.Pix[i+2] = blend(out.Pix[i+2], hb, w)
		}
	}
	return out
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

// ensure color is referenced for the Gray type's documented behaviour
var _ = color.Gray{}
