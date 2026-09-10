package infer

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/jpeg"
	"testing"
)

// jpegWithOrientation encodes img and splices an EXIF APP1 segment carrying the
// given orientation directly after the SOI marker, which is where a camera
// writes it.
func jpegWithOrientation(t *testing.T, img image.Image, o Orientation) []byte {
	t.Helper()

	var body bytes.Buffer
	if err := jpeg.Encode(&body, img, nil); err != nil {
		t.Fatalf("encode: %v", err)
	}
	raw := body.Bytes()

	// Exif header + little-endian TIFF header + a one-entry IFD0.
	var tiff bytes.Buffer
	tiff.WriteString("Exif\x00\x00")
	tiff.WriteString("II")
	binary.Write(&tiff, binary.LittleEndian, uint16(42))
	binary.Write(&tiff, binary.LittleEndian, uint32(8)) // IFD0 offset
	binary.Write(&tiff, binary.LittleEndian, uint16(1)) // entry count
	binary.Write(&tiff, binary.LittleEndian, uint16(0x0112))
	binary.Write(&tiff, binary.LittleEndian, uint16(3)) // SHORT
	binary.Write(&tiff, binary.LittleEndian, uint32(1)) // count
	binary.Write(&tiff, binary.LittleEndian, uint16(o))
	binary.Write(&tiff, binary.LittleEndian, uint16(0)) // value field padding
	binary.Write(&tiff, binary.LittleEndian, uint32(0)) // next IFD

	var out bytes.Buffer
	out.Write(raw[:2]) // SOI
	out.Write([]byte{0xFF, 0xE1})
	binary.Write(&out, binary.BigEndian, uint16(tiff.Len()+2))
	out.Write(tiff.Bytes())
	out.Write(raw[2:])
	return out.Bytes()
}

func TestJPEGOrientationRoundTrips(t *testing.T) {
	img := imageRGBA(8, 8)
	for o := Orientation(1); o <= 8; o++ {
		raw := jpegWithOrientation(t, img, o)
		if got := JPEGOrientation(raw); got != o {
			t.Errorf("orientation %d read back as %d", o, got)
		}
	}
}

func TestJPEGOrientationToleratesJunk(t *testing.T) {
	img := imageRGBA(8, 8)
	var plain bytes.Buffer
	if err := jpeg.Encode(&plain, img, nil); err != nil {
		t.Fatalf("encode: %v", err)
	}

	cases := map[string][]byte{
		"no exif":  plain.Bytes(),
		"empty":    nil,
		"not jpeg": []byte("this is a PNG, honest"),
		"truncated header": func() []byte {
			b := jpegWithOrientation(t, img, 6)
			return b[:6]
		}(),
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			// Unreadable EXIF must never be a reason to reject an upload.
			if got := JPEGOrientation(raw); got != OrientationNormal {
				t.Errorf("got orientation %d, want %d", got, OrientationNormal)
			}
		})
	}
}

// cornerImage paints each corner a distinct colour so a transform can be
// identified from where the corners end up.
func cornerImage(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i] = uint8(255 * x / max(w-1, 1))   // red rises to the right
			img.Pix[i+1] = uint8(255 * y / max(h-1, 1)) // green rises downward
			img.Pix[i+3] = 255
		}
	}
	return img
}

func TestReorient(t *testing.T) {
	const w, h = 4, 6
	src := cornerImage(w, h)

	// Where the top-left pixel of the upright image must come from, per the
	// EXIF spec's definition of each orientation.
	topLeftSource := map[Orientation][2]int{
		2: {w - 1, 0},
		3: {w - 1, h - 1},
		4: {0, h - 1},
		5: {0, 0},
		6: {0, h - 1},
		7: {w - 1, h - 1},
		8: {w - 1, 0},
	}

	for o := Orientation(2); o <= 8; o++ {
		out := Reorient(src, o)
		wantW, wantH := w, h
		if o >= 5 {
			wantW, wantH = h, w
		}
		if got := out.Bounds(); got.Dx() != wantW || got.Dy() != wantH {
			t.Errorf("orientation %d: bounds %v, want %dx%d", o, got, wantW, wantH)
		}

		s := topLeftSource[o]
		wr, wg, _, _ := src.At(s[0], s[1]).RGBA()
		gr, gg, _, _ := out.At(0, 0).RGBA()
		if gr != wr || gg != wg {
			t.Errorf("orientation %d: top-left is (%d,%d), want the pixel from (%d,%d) = (%d,%d)",
				o, gr>>8, gg>>8, s[0], s[1], wr>>8, wg>>8)
		}
	}

	// Every transform must be reversible, so no pixels are lost.
	for o := Orientation(1); o <= 8; o++ {
		out := Reorient(src, o)
		if out.Bounds().Dx()*out.Bounds().Dy() != w*h {
			t.Errorf("orientation %d changed the pixel count", o)
		}
	}
}

func TestReorientNormalIsTheSameImage(t *testing.T) {
	src := cornerImage(4, 4)
	if got := Reorient(src, OrientationNormal); got != image.Image(src) {
		t.Error("OrientationNormal must return the source untouched, not a copy")
	}
}
