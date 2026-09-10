package pdfdoc

import (
	"bufio"
	"fmt"
	"image"
	"io"
)

const (
	// analysisMaxDim bounds the raster used to *locate* the discs.
	//
	// Finding a bounding box does not need detail, so this pass is deliberately
	// coarse: at 1024 the raster costs ~2.7 MB instead of the 40-50 MB a fundus
	// photo would occupy at full size. The crop itself is taken at native
	// resolution in a second pass - see cropNative.
	analysisMaxDim = 1024

	// maxPixels rejects absurd rasters before any allocation.
	maxPixels = 64 << 20

	// minRegionPx is the smallest square a fundus disc can plausibly occupy in
	// the decimated raster.
	minRegionPx = 64

	// contentLevel is the channel value above which a pixel counts as image
	// content. The area outside the disc is exact black, so this only has to
	// clear the ringing introduced by box-averaging.
	contentLevel = 16
)

// raster is a decimated RGB image held as tightly packed bytes.
type raster struct {
	pix  []byte
	w, h int
}

func (r *raster) at(x, y int) (byte, byte, byte) {
	i := (y*r.w + x) * 3
	return r.pix[i], r.pix[i+1], r.pix[i+2]
}

func (r *raster) hasContent(x, y int) bool {
	cr, cg, cb := r.at(x, y)
	return cr > contentLevel || cg > contentLevel || cb > contentLevel
}

// components returns how many colour samples per pixel the image stores.
func (d *Doc) components(dict Dict) (int, bool) {
	switch cs := d.Resolve(dict["ColorSpace"]).(type) {
	case Name:
		switch cs {
		case "DeviceRGB":
			return 3, true
		case "DeviceGray":
			return 1, true
		case "DeviceCMYK":
			return 4, true
		}
	case Array:
		if len(cs) >= 2 {
			if n, _ := d.NameOf(cs[0]); n == "ICCBased" {
				if sd, ok := d.DictOf(cs[1]); ok {
					if n, ok := d.IntOf(sd["N"]); ok {
						return n, true
					}
				}
			}
		}
	}
	return 0, false
}

// decimate streams an image XObject into a bounded raster.
//
// The stream is inflated once, one source row at a time, and box-averaged
// straight into the output. The full-size raster - 40 to 50 MB for a fundus
// photo - is never materialised. The decimation factor is returned so a caller
// can map regions found here back onto source coordinates.
func (d *Doc) decimate(s *Stream) (*raster, int, error) {
	w, h, comps, err := d.imageGeometry(s)
	if err != nil {
		return nil, 0, err
	}

	k := 1
	for maxInt(w, h)/k > analysisMaxDim {
		k++
	}
	ow, oh := w/k, h/k
	if ow < 1 || oh < 1 {
		return nil, 0, fmt.Errorf("pdfdoc: image too small after decimation")
	}

	rc, err := d.DecodedReader(s)
	if err != nil {
		return nil, 0, err
	}
	defer rc.Close()
	br := bufio.NewReaderSize(rc, 64<<10)

	out := &raster{pix: make([]byte, ow*oh*3), w: ow, h: oh}
	rowBuf := make([]byte, w*comps)
	acc := make([]uint32, ow*3)
	div := uint32(k * k)

	for oy := 0; oy < oh; oy++ {
		for i := range acc {
			acc[i] = 0
		}
		for sub := 0; sub < k; sub++ {
			if _, err := io.ReadFull(br, rowBuf); err != nil {
				return nil, 0, fmt.Errorf("pdfdoc: read image row: %w", err)
			}
			for ox := 0; ox < ow; ox++ {
				base := ox * k * comps
				var r, g, b uint32
				for j := 0; j < k; j++ {
					p := base + j*comps
					r += uint32(rowBuf[p])
					g += uint32(rowBuf[p+1])
					b += uint32(rowBuf[p+2])
				}
				acc[ox*3] += r
				acc[ox*3+1] += g
				acc[ox*3+2] += b
			}
		}
		dst := out.pix[oy*ow*3:]
		for i := 0; i < ow*3; i++ {
			dst[i] = byte(acc[i] / div)
		}
	}
	return out, k, nil
}

// imageGeometry validates an image XObject and returns its dimensions.
func (d *Doc) imageGeometry(s *Stream) (w, h, comps int, err error) {
	w, ok1 := d.IntOf(s.Dict["Width"])
	h, ok2 := d.IntOf(s.Dict["Height"])
	if !ok1 || !ok2 || w <= 0 || h <= 0 {
		return 0, 0, 0, fmt.Errorf("pdfdoc: image has no usable dimensions")
	}
	if w*h > maxPixels {
		return 0, 0, 0, fmt.Errorf("pdfdoc: image %dx%d exceeds the pixel budget", w, h)
	}
	if bpc, ok := d.IntOf(s.Dict["BitsPerComponent"]); !ok || bpc != 8 {
		return 0, 0, 0, fmt.Errorf("pdfdoc: only 8-bit images are supported")
	}
	comps, ok := d.components(s.Dict)
	if !ok || comps != 3 {
		return 0, 0, 0, fmt.Errorf("pdfdoc: only 3-component colour is supported")
	}
	return w, h, comps, nil
}

// discRegions finds the bounding boxes of the fundus discs in a raster.
//
// A report may paint one eye per image or both eyes into a single wide raster,
// so this splits on runs of empty columns rather than assuming one disc.
func (r *raster) discRegions() []image.Rectangle {
	colHits := make([]int, r.w)
	for y := 0; y < r.h; y++ {
		for x := 0; x < r.w; x++ {
			if r.hasContent(x, y) {
				colHits[x]++
			}
		}
	}

	// A column counts as occupied only if a meaningful share of it is content,
	// so a stray bright pixel cannot bridge two discs into one region.
	minCol := maxInt(2, r.h/64)

	var out []image.Rectangle
	x := 0
	for x < r.w {
		if colHits[x] < minCol {
			x++
			continue
		}
		start := x
		for x < r.w && colHits[x] >= minCol {
			x++
		}
		end := x // exclusive

		yMin, yMax := -1, -1
		for y := 0; y < r.h; y++ {
			row := false
			for xx := start; xx < end; xx++ {
				if r.hasContent(xx, y) {
					row = true
					break
				}
			}
			if row {
				if yMin < 0 {
					yMin = y
				}
				yMax = y
			}
		}
		if yMin < 0 {
			continue
		}
		rect := image.Rect(start, yMin, end, yMax+1)
		if rect.Dx() < minRegionPx || rect.Dy() < minRegionPx {
			continue
		}
		// A fundus disc is round. Anything far from square is a header strip or
		// a rule, not an eye.
		if aspect(rect) > 2.0 {
			continue
		}
		out = append(out, rect)
	}
	return out
}

func aspect(r image.Rectangle) float64 {
	w, h := float64(r.Dx()), float64(r.Dy())
	if w < h {
		w, h = h, w
	}
	if h == 0 {
		return 0
	}
	return w / h
}

// cropNative streams an image XObject a second time and copies the given
// regions out of it at full resolution.
//
// The regions arrive in decimated coordinates, so they are scaled back up by k
// and padded by the same amount before clamping: decimation rounds a boundary
// down by up to k-1 source pixels, and clipping the rim off a fundus disc would
// be a real loss. Overshooting only admits a few pixels of the black surround,
// which neither the gate nor the model is sensitive to.
//
// Every region is filled during this one pass, so a report holding both eyes in
// a single raster costs one inflate rather than one per eye.
func (d *Doc) cropNative(s *Stream, regions []image.Rectangle, k int) ([]*image.NRGBA, error) {
	w, h, comps, err := d.imageGeometry(s)
	if err != nil {
		return nil, err
	}

	bounds := image.Rect(0, 0, w, h)
	rects := make([]image.Rectangle, len(regions))
	out := make([]*image.NRGBA, len(regions))
	for i, r := range regions {
		scaled := image.Rect(r.Min.X*k-k, r.Min.Y*k-k, r.Max.X*k+k, r.Max.Y*k+k)
		rects[i] = scaled.Intersect(bounds)
		if rects[i].Empty() {
			return nil, fmt.Errorf("pdfdoc: region %v falls outside the image", r)
		}
		out[i] = image.NewNRGBA(image.Rect(0, 0, rects[i].Dx(), rects[i].Dy()))
	}

	rc, err := d.DecodedReader(s)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	br := bufio.NewReaderSize(rc, 64<<10)

	// Stop reading once the last row any region needs has gone by.
	lastRow := 0
	for _, r := range rects {
		lastRow = maxInt(lastRow, r.Max.Y)
	}

	rowBuf := make([]byte, w*comps)
	for y := 0; y < lastRow; y++ {
		if _, err := io.ReadFull(br, rowBuf); err != nil {
			return nil, fmt.Errorf("pdfdoc: read image row: %w", err)
		}
		for i, r := range rects {
			if y < r.Min.Y || y >= r.Max.Y {
				continue
			}
			img := out[i]
			dst := (y - r.Min.Y) * img.Stride
			src := r.Min.X * comps
			for x := 0; x < r.Dx(); x++ {
				img.Pix[dst] = rowBuf[src]
				img.Pix[dst+1] = rowBuf[src+1]
				img.Pix[dst+2] = rowBuf[src+2]
				img.Pix[dst+3] = 255
				src += comps
				dst += 4
			}
		}
	}
	return out, nil
}

// crop copies a region out of the raster as an NRGBA image.
func (r *raster) crop(rect image.Rectangle) *image.NRGBA {
	rect = rect.Intersect(image.Rect(0, 0, r.w, r.h))
	out := image.NewNRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	for y := 0; y < rect.Dy(); y++ {
		src := ((y+rect.Min.Y)*r.w + rect.Min.X) * 3
		dst := y * out.Stride
		for x := 0; x < rect.Dx(); x++ {
			out.Pix[dst] = r.pix[src]
			out.Pix[dst+1] = r.pix[src+1]
			out.Pix[dst+2] = r.pix[src+2]
			out.Pix[dst+3] = 255
			src += 3
			dst += 4
		}
	}
	return out
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
