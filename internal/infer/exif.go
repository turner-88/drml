package infer

import (
	"encoding/binary"
	"image"
)

// Orientation is the EXIF orientation tag (0x0112), 1..8. OrientationNormal
// means the stored pixels are already upright.
type Orientation int

const OrientationNormal Orientation = 1

// JPEGOrientation reads the EXIF orientation tag out of a JPEG's APP1 segment.
//
// Go's image/jpeg decoder ignores EXIF entirely, while browsers honour it by
// default (CSS image-orientation: from-image). Left alone, that splits the
// pipeline in two: the model grades — and the overlay is rendered on — sideways
// pixels, while the clinician sees the upright ones, so the heatmap lands in
// the wrong place by a rotation. Reading the tag ourselves is the whole fix,
// and it costs a scan of a few hundred bytes of header.
//
// Anything unreadable or absent returns OrientationNormal: this must never be
// the reason an upload is rejected.
func JPEGOrientation(raw []byte) Orientation {
	// SOI, then a chain of marker segments until the compressed scan begins.
	if len(raw) < 4 || raw[0] != 0xFF || raw[1] != 0xD8 {
		return OrientationNormal
	}
	for i := 2; i+4 <= len(raw); {
		if raw[i] != 0xFF {
			return OrientationNormal // out of sync with the marker chain
		}
		marker := raw[i+1]
		switch {
		case marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7):
			i += 2 // standalone marker, no length field
			continue
		case marker == 0xDA || marker == 0xD9:
			return OrientationNormal // start of scan / end of image
		}
		size := int(binary.BigEndian.Uint16(raw[i+2 : i+4]))
		if size < 2 || i+2+size > len(raw) {
			return OrientationNormal
		}
		if marker == 0xE1 {
			payload := raw[i+4 : i+2+size]
			if o, ok := orientationFromAPP1(payload); ok {
				return o
			}
		}
		i += 2 + size
	}
	return OrientationNormal
}

// orientationFromAPP1 parses one APP1 payload as an Exif TIFF header and looks
// for tag 0x0112 in IFD0.
func orientationFromAPP1(p []byte) (Orientation, bool) {
	const header = "Exif\x00\x00"
	if len(p) < len(header)+8 || string(p[:len(header)]) != header {
		return OrientationNormal, false
	}
	tiff := p[len(header):]

	var bo binary.ByteOrder
	switch string(tiff[0:2]) {
	case "II":
		bo = binary.LittleEndian
	case "MM":
		bo = binary.BigEndian
	default:
		return OrientationNormal, false
	}
	if bo.Uint16(tiff[2:4]) != 42 {
		return OrientationNormal, false
	}

	off := int(bo.Uint32(tiff[4:8]))
	if off < 8 || off+2 > len(tiff) {
		return OrientationNormal, false
	}
	count := int(bo.Uint16(tiff[off : off+2]))
	off += 2

	// Each IFD entry is 12 bytes: tag, type, count, value/offset.
	for n := 0; n < count && off+12 <= len(tiff); n, off = n+1, off+12 {
		if bo.Uint16(tiff[off:off+2]) != 0x0112 {
			continue
		}
		// Type 3 (SHORT), count 1: the value sits inline in the first two
		// bytes of the value field.
		if bo.Uint16(tiff[off+2:off+4]) != 3 || bo.Uint32(tiff[off+4:off+8]) != 1 {
			return OrientationNormal, false
		}
		v := Orientation(bo.Uint16(tiff[off+8 : off+10]))
		if v < 1 || v > 8 {
			return OrientationNormal, false
		}
		return v, true
	}
	return OrientationNormal, false
}

// Reorient returns img rotated and flipped so its pixels are upright, applying
// the EXIF orientation the decoder dropped.
//
// OrientationNormal returns img itself: the common case must not pay for a
// full copy.
func Reorient(img image.Image, o Orientation) image.Image {
	if img == nil || o == OrientationNormal || o < 1 || o > 8 {
		return img
	}

	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	// 5..8 exchange the axes, so the output is transposed.
	dw, dh := w, h
	if o >= 5 {
		dw, dh = h, w
	}
	out := image.NewRGBA(image.Rect(0, 0, dw, dh))

	// src maps a destination pixel back to its source pixel. Writing it in this
	// direction means every output pixel is written exactly once, with no gaps.
	var src func(dx, dy int) (int, int)
	switch o {
	case 2: // mirror horizontally
		src = func(dx, dy int) (int, int) { return w - 1 - dx, dy }
	case 3: // rotate 180
		src = func(dx, dy int) (int, int) { return w - 1 - dx, h - 1 - dy }
	case 4: // mirror vertically
		src = func(dx, dy int) (int, int) { return dx, h - 1 - dy }
	case 5: // transpose
		src = func(dx, dy int) (int, int) { return dy, dx }
	case 6: // rotate 90 clockwise
		src = func(dx, dy int) (int, int) { return dy, h - 1 - dx }
	case 7: // transverse
		src = func(dx, dy int) (int, int) { return w - 1 - dy, h - 1 - dx }
	case 8: // rotate 90 counter-clockwise
		src = func(dx, dy int) (int, int) { return w - 1 - dy, dx }
	}

	for dy := 0; dy < dh; dy++ {
		for dx := 0; dx < dw; dx++ {
			sx, sy := src(dx, dy)
			out.Set(dx, dy, img.At(b.Min.X+sx, b.Min.Y+sy))
		}
	}
	return out
}

// Upright applies the EXIF orientation carried by img's original bytes, so the
// gate, the model and the overlay all work on the same pixels the browser will
// show.
func Upright(img image.Image, raw []byte) image.Image {
	return Reorient(img, JPEGOrientation(raw))
}
