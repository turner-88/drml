package pdfdoc

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"
	"testing"
)

func extract(t *testing.T, pdf []byte) ([]Fundus, error) {
	t.Helper()
	return Extract(bytes.NewReader(pdf), int64(len(pdf)))
}

func mustExtract(t *testing.T, pdf []byte) []Fundus {
	t.Helper()
	out, err := extract(t, pdf)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	return out
}

func TestExtractTwoEyes(t *testing.T) {
	got := mustExtract(t, twoEyeReport())
	if len(got) != 2 {
		t.Fatalf("got %d eyes, want 2", len(got))
	}
	// Sorted left to right, which for this layout is OD then OS.
	if got[0].Eye != EyeOD || got[1].Eye != EyeOS {
		t.Errorf("eyes = %s,%s; want OD,OS", got[0].Eye, got[1].Eye)
	}
	for _, f := range got {
		b := f.Img.Bounds()
		// The disc has radius 100, so the crop should be about 200 square.
		if b.Dx() < 190 || b.Dx() > 210 || b.Dy() < 190 || b.Dy() > 210 {
			t.Errorf("%s crop = %dx%d, want ~200x200", f.Eye, b.Dx(), b.Dy())
		}
		// The surround must stay black: the model and the gate are both
		// calibrated on fundus images framed that way.
		r, g, bl, _ := f.Img.At(1, 1).RGBA()
		if r>>8 > 16 || g>>8 > 16 || bl>>8 > 16 {
			t.Errorf("%s corner = (%d,%d,%d), want near-black", f.Eye, r>>8, g>>8, bl>>8)
		}
	}
}

// TestExtractSingleEyeUsesLabel is the regression that matters most.
//
// A report carrying one eye places it dead centre whichever eye it is, so these
// two documents are geometrically identical and differ only in their label. Any
// shortcut that infers laterality from position fails here.
func TestExtractSingleEyeUsesLabel(t *testing.T) {
	od, os := oneEyeReport("OD(R)"), oneEyeReport("OS(L)")

	// Guard the premise: if the fixtures ever stop being geometrically
	// identical, this test would pass for the wrong reason.
	if len(od) != len(os) {
		t.Fatalf("fixtures differ in size (%d vs %d); the layouts must match", len(od), len(os))
	}

	for _, tc := range []struct {
		name string
		pdf  []byte
		want Eye
	}{
		{"right eye", od, EyeOD},
		{"left eye", os, EyeOS},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := mustExtract(t, tc.pdf)
			if len(got) != 1 {
				t.Fatalf("got %d eyes, want 1", len(got))
			}
			if got[0].Eye != tc.want {
				t.Errorf("eye = %s, want %s", got[0].Eye, tc.want)
			}
		})
	}
}

// TestExtractSkipsNonFundusImages covers the clinic logo and the alpha masks,
// which sit in the same resource dictionary as the photographs.
func TestExtractSkipsNonFundusImages(t *testing.T) {
	pdf := buildReport(
		[]fixtureImage{
			// A DCTDecode logo: right subtype, filter we do not decode.
			{num: 9, w: 32, h: 32, xName: "XL", pageX: 700, pageW: 60,
				pix: bytes.Repeat([]byte{0xFF}, 64),
				dict: "/Type/XObject/Subtype/Image/Width 32/Height 32" +
					"/ColorSpace/DeviceRGB/BitsPerComponent 8/Filter/DCTDecode"},
			// A DeviceGray plane, the shape an /SMask has.
			{num: 10, w: 300, h: 300, pix: grayRaster(300, 300), xName: "XM", pageX: 60, pageW: 300,
				dict: "/Type/XObject/Subtype/Image/Width 300/Height 300" +
					"/ColorSpace/DeviceGray/BitsPerComponent 8"},
			{num: 5, w: 300, h: 300, pix: discRaster(300, 300, 150, 150, 100), xName: "X1", pageX: 60, pageW: 300},
		},
		[]fixtureLabel{{text: "OD(R)", pageX: 70}},
		"")

	got := mustExtract(t, pdf)
	if len(got) != 1 {
		t.Fatalf("got %d eyes, want 1 (logo and mask must be skipped)", len(got))
	}
	if got[0].Eye != EyeOD {
		t.Errorf("eye = %s, want OD", got[0].Eye)
	}
}

func TestExtractAmbiguousLabelsRefused(t *testing.T) {
	// Two photographs but only one label: laterality cannot be established, and
	// filing a scan against a guessed eye is worse than refusing.
	pdf := buildReport(
		[]fixtureImage{
			{num: 5, w: 300, h: 300, pix: discRaster(300, 300, 150, 150, 100), xName: "X1", pageX: 60, pageW: 300},
			{num: 6, w: 300, h: 300, pix: discRaster(300, 300, 150, 150, 100), xName: "X2", pageX: 460, pageW: 300},
		},
		[]fixtureLabel{{text: "OD(R)", pageX: 70}},
		"")

	_, err := extract(t, pdf)
	if err == nil || !strings.Contains(err.Error(), "labels do not match") {
		t.Fatalf("err = %v, want ErrAmbiguousEye", err)
	}
}

func TestExtractDuplicateEyeRefused(t *testing.T) {
	pdf := buildReport(
		[]fixtureImage{
			{num: 5, w: 300, h: 300, pix: discRaster(300, 300, 150, 150, 100), xName: "X1", pageX: 60, pageW: 300},
			{num: 6, w: 300, h: 300, pix: discRaster(300, 300, 150, 150, 100), xName: "X2", pageX: 460, pageW: 300},
		},
		[]fixtureLabel{{text: "OD(R)", pageX: 70}, {text: "OD(R)", pageX: 470}},
		"")

	if _, err := extract(t, pdf); err == nil {
		t.Fatal("want an error when both images claim the same eye")
	}
}

func TestExtractNoFundus(t *testing.T) {
	pdf := buildReport(nil, []fixtureLabel{{text: "OD(R)", pageX: 70}}, "")
	if _, err := extract(t, pdf); err == nil {
		t.Fatal("want an error for a report with no images")
	}
}

func TestExtractRejectsEncrypted(t *testing.T) {
	pdf := buildReport(
		[]fixtureImage{
			{num: 5, w: 300, h: 300, pix: discRaster(300, 300, 150, 150, 100), xName: "X1", pageX: 60, pageW: 300},
		},
		[]fixtureLabel{{text: "OD(R)", pageX: 70}},
		"/Encrypt 99 0 R")

	_, err := extract(t, pdf)
	if err == nil || !strings.Contains(err.Error(), "unsupported PDF") {
		t.Fatalf("err = %v, want ErrNotSupportedPDF", err)
	}
}

// TestExtractToleratesJunk asserts the parser fails rather than panicking on
// input that is not a report. Uploads are user-controlled.
func TestExtractToleratesJunk(t *testing.T) {
	good := twoEyeReport()
	cases := map[string][]byte{
		"empty":     {},
		"tiny":      []byte("%PDF"),
		"random":    bytes.Repeat([]byte{0x9e, 0x01, 0xff, 0x7f}, 512),
		"truncated": good[:len(good)/2],
	}
	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panic on %s input: %v", name, r)
				}
			}()
			if _, err := extract(t, in); err == nil {
				t.Errorf("want an error for %s input", name)
			}
		})
	}
}

// TestExtractRecoversFromDamagedXref exercises the object-scan fallback.
//
// The scan is what lets a report survive a mangled cross-reference table, which
// is worth having: the alternative is refusing a document whose images are
// perfectly readable.
func TestExtractRecoversFromDamagedXref(t *testing.T) {
	good := twoEyeReport()

	i := bytes.Index(good, []byte("\nxref\n"))
	if i < 0 {
		t.Fatal("fixture has no xref table")
	}
	// Mark every entry free, so the table yields no usable objects.
	freed := append(append([]byte{}, good[:i]...),
		bytes.ReplaceAll(good[i:], []byte(" 00000 n "), []byte(" 00000 f "))...)

	cases := map[string][]byte{
		"entries marked free": freed,
		// Prepending bytes shifts every recorded offset out of true.
		"offsets shifted": append([]byte("%PDF-1.4\n% padding padding padding\n"), good...),
		"no startxref":    bytes.ReplaceAll(good, []byte("startxref"), []byte("startxrfe")),
	}
	for name, pdf := range cases {
		t.Run(name, func(t *testing.T) {
			got, err := extract(t, pdf)
			if err != nil {
				t.Fatalf("Extract: %v", err)
			}
			if len(got) != 2 || got[0].Eye != EyeOD || got[1].Eye != EyeOS {
				t.Fatalf("got %v, want OD and OS", got)
			}
		})
	}
}

// TestExtractStreamsLargeRasters is the guard against anyone replacing the
// row-at-a-time decode with a single zlib.Decompress of the whole image.
//
// The raster below is 48 MB raw. Materialising it - even once - would show up
// here immediately.
func TestExtractStreamsLargeRasters(t *testing.T) {
	const dim = 4000
	rawSize := dim * dim * 3

	pdf := buildReport(
		[]fixtureImage{
			{num: 5, w: dim, h: dim, pix: discRaster(dim, dim, dim/2, dim/2, dim/3),
				xName: "X1", pageX: 60, pageW: 300},
		},
		[]fixtureLabel{{text: "OD(R)", pageX: 70}},
		"")

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	got := mustExtract(t, pdf)

	runtime.ReadMemStats(&after)
	allocated := after.TotalAlloc - before.TotalAlloc

	if len(got) != 1 || got[0].Eye != EyeOD {
		t.Fatalf("got %d eyes (%v), want 1 OD", len(got), got)
	}

	// The crop keeps the resolution the report stores, because the source PDF
	// is not retained: the disc has radius dim/3, so it spans about 2*dim/3.
	// Catching a regression to a decimated crop matters as much as the memory
	// bound below - it would silently halve what the clinician can see.
	b := got[0].Img.Bounds()
	if want := 2 * dim / 3; b.Dx() < want-16 || b.Dx() > want+16 {
		t.Errorf("crop width = %d, want ~%d (native resolution)", b.Dx(), want)
	}

	// The budget is the crop itself (~28 MB here) plus the coarse raster used
	// to locate it (~3 MB). Inflating the whole image would add rawSize on top
	// of both, so rawSize is the honest ceiling: it cannot be reached while the
	// decode streams, and cannot be avoided once it does not.
	ceiling := uint64(rawSize)
	if allocated > ceiling {
		t.Errorf("allocated %.1f MB decoding a %.1f MB raster; want under %.1f MB "+
			"(the raster is being materialised instead of streamed)",
			float64(allocated)/(1<<20), float64(rawSize)/(1<<20), float64(ceiling)/(1<<20))
	}
	t.Logf("allocated %.1f MB for a %.1f MB raster", float64(allocated)/(1<<20), float64(rawSize)/(1<<20))
}

func TestEyeFromText(t *testing.T) {
	cases := []struct {
		in   string
		want Eye
		ok   bool
	}{
		{"OD(R)", EyeOD, true},
		{"OS(L)", EyeOS, true},
		{" OD ", EyeOD, true},
		{"od(r)", EyeOD, true},
		{"ODONTOLOGI", "", false},
		{"OSTEOPOROSIS", "", false},
		{"Patient Name", "", false},
		{"", "", false},
		{strings.Repeat("OD", 40), "", false},
	}
	for _, tc := range cases {
		got, ok := eyeFromText(tc.in)
		if ok != tc.ok || got != tc.want {
			t.Errorf("eyeFromText(%q) = %q,%v; want %q,%v", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}

func TestCMapDecodesBFRange(t *testing.T) {
	cm := parseCMap([]byte(fmt.Sprintf(
		"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"+
			"1 beginbfrange\n<0010> <0012> <%s>\nendbfrange\n", utf16beHex("A"))))
	got := cm.decode(String{0x00, 0x10, 0x00, 0x11, 0x00, 0x12})
	if got != "ABC" {
		t.Errorf("decode = %q, want %q", got, "ABC")
	}
}
