package infer

import (
	"encoding/json"
	"fmt"
	"image"
	"os"

	xdraw "golang.org/x/image/draw"
)

// PreprocessConfig mirrors model/preprocess.json, which export.py writes from
// the checkpoint's own image processor. Reading it at runtime means a model
// swap that changes input size or normalization needs no Go changes.
type PreprocessConfig struct {
	Height        int       `json:"height"`
	Width         int       `json:"width"`
	ImageMean     []float32 `json:"image_mean"`
	ImageStd      []float32 `json:"image_std"`
	RescaleFactor float32   `json:"rescale_factor"`
	DoNormalize   bool      `json:"do_normalize"`
	DoRescale     bool      `json:"do_rescale"`
	ResampleCode  int       `json:"resample_code"`
	Resample      string    `json:"resample"`
}

// LoadPreprocessConfig reads and validates the exported preprocessing constants.
func LoadPreprocessConfig(path string) (*PreprocessConfig, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read preprocess config: %w", err)
	}
	var cfg PreprocessConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse preprocess config: %w", err)
	}
	if cfg.Height <= 0 || cfg.Width <= 0 {
		return nil, fmt.Errorf("invalid input size %dx%d", cfg.Width, cfg.Height)
	}
	if len(cfg.ImageMean) != 3 || len(cfg.ImageStd) != 3 {
		return nil, fmt.Errorf("image_mean/image_std must have 3 entries, got %d/%d",
			len(cfg.ImageMean), len(cfg.ImageStd))
	}
	for i, s := range cfg.ImageStd {
		if s == 0 {
			return nil, fmt.Errorf("image_std[%d] is zero", i)
		}
	}
	return &cfg, nil
}

// scaler picks the Go resampler matching the PIL filter recorded at export
// time. PIL widens a filter's support when downscaling, so these must be
// x/image/draw Kernels (which do the same) rather than naive samplers —
// otherwise Go and Python disagree and the golden parity test fails.
func (p *PreprocessConfig) scaler() xdraw.Interpolator {
	switch p.ResampleCode {
	case 0: // PIL NEAREST
		return xdraw.NearestNeighbor
	case 3: // PIL BICUBIC
		return xdraw.CatmullRom
	case 1: // PIL LANCZOS — no exact equivalent; CatmullRom is the closest kernel
		return xdraw.CatmullRom
	default: // 2 = PIL BILINEAR, the ViT default
		return xdraw.BiLinear
	}
}

// Resize scales img to the model's input dimensions using the exported filter.
func (p *PreprocessConfig) Resize(img image.Image) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, p.Width, p.Height))
	p.scaler().Scale(dst, dst.Bounds(), img, img.Bounds(), xdraw.Src, nil)
	return dst
}

// Tensor converts an image to the flat CHW float32 buffer the ONNX graph wants.
//
// It performs resize -> rescale -> normalize, matching the HF processor. It
// deliberately does NOT crop: cropping is a separate pre-step so the parity
// test can compare this function against Python like for like.
func (p *PreprocessConfig) Tensor(img image.Image) []float32 {
	rgba := p.Resize(img)

	out := make([]float32, 3*p.Height*p.Width)
	planeSize := p.Height * p.Width

	rescale := p.RescaleFactor
	if !p.DoRescale {
		rescale = 1
	}

	for y := 0; y < p.Height; y++ {
		for x := 0; x < p.Width; x++ {
			// Read the 8-bit sRGB bytes directly. Go's color.Color would widen
			// to 16-bit alpha-premultiplied values, which is not what PIL sees.
			i := rgba.PixOffset(x, y)
			idx := y*p.Width + x
			for c := 0; c < 3; c++ {
				v := float32(rgba.Pix[i+c]) * rescale
				if p.DoNormalize {
					v = (v - p.ImageMean[c]) / p.ImageStd[c]
				}
				out[c*planeSize+idx] = v
			}
		}
	}
	return out
}
