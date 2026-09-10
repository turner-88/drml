package pdfdoc

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"unicode/utf16"
)

// A minimal PDF writer for tests.
//
// The real reports are patient records and cannot be committed, so the fixtures
// are built here instead: a classic xref table, FlateDecode RGB images with
// black surrounds, and a ToUnicode CMap carrying the eye labels — the same
// shape IMAGEnet's Chrome print-to-PDF output has.
type pdfBuilder struct {
	buf  bytes.Buffer
	offs map[int]int64
	max  int
}

func newPDF() *pdfBuilder {
	b := &pdfBuilder{offs: map[int]int64{}}
	b.buf.WriteString("%PDF-1.4\n")
	return b
}

func (b *pdfBuilder) track(num int) {
	b.offs[num] = int64(b.buf.Len())
	if num > b.max {
		b.max = num
	}
}

func (b *pdfBuilder) obj(num int, body string) {
	b.track(num)
	fmt.Fprintf(&b.buf, "%d 0 obj\n%s\nendobj\n", num, body)
}

func (b *pdfBuilder) stream(num int, dict string, data []byte) {
	b.track(num)
	fmt.Fprintf(&b.buf, "%d 0 obj\n<<%s/Length %d>>\nstream\n", num, dict, len(data))
	b.buf.Write(data)
	b.buf.WriteString("\nendstream\nendobj\n")
}

// flateStream writes a stream whose data is zlib-compressed.
func (b *pdfBuilder) flateStream(num int, dict string, raw []byte) {
	var z bytes.Buffer
	w := zlib.NewWriter(&z)
	w.Write(raw)
	w.Close()
	b.stream(num, dict+"/Filter/FlateDecode", z.Bytes())
}

func (b *pdfBuilder) finish(trailerExtra string) []byte {
	xref := int64(b.buf.Len())
	n := b.max + 1
	fmt.Fprintf(&b.buf, "xref\n0 %d\n", n)
	b.buf.WriteString("0000000000 65535 f \n")
	for i := 1; i < n; i++ {
		fmt.Fprintf(&b.buf, "%010d 00000 n \n", b.offs[i])
	}
	fmt.Fprintf(&b.buf, "trailer\n<</Size %d/Root 1 0 R%s>>\nstartxref\n%d\n%%%%EOF\n",
		n, trailerExtra, xref)
	return b.buf.Bytes()
}

// utf16beHex encodes a string the way a ToUnicode destination is written.
func utf16beHex(s string) string {
	var out bytes.Buffer
	for _, u := range utf16.Encode([]rune(s)) {
		fmt.Fprintf(&out, "%04X", u)
	}
	return out.String()
}

// toUnicodeCMap maps 2-byte codes to text, one entry per label.
func toUnicodeCMap(labels []string) []byte {
	var b bytes.Buffer
	b.WriteString("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n")
	b.WriteString("1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n")
	fmt.Fprintf(&b, "%d beginbfchar\n", len(labels))
	for i, l := range labels {
		fmt.Fprintf(&b, "<%04X> <%s>\n", i+1, utf16beHex(l))
	}
	b.WriteString("endbfchar\nendcmap\nend\nend\n")
	return b.Bytes()
}

// discRaster builds an RGB raster: black everywhere, with a red-dominant disc.
func discRaster(w, h, cx, cy, r int) []byte {
	pix := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx, dy := x-cx, y-cy
			if dx*dx+dy*dy <= r*r {
				i := (y*w + x) * 3
				pix[i], pix[i+1], pix[i+2] = 200, 90, 60
			}
		}
	}
	return pix
}

// grayRaster stands in for an SMask: single channel, so it must be skipped.
func grayRaster(w, h int) []byte { return make([]byte, w*h) }

type fixtureImage struct {
	num        int
	w, h       int
	pix        []byte
	dict       string // overrides the default RGB image dict when set
	xName      string
	pageX      float64
	pageW      float64
	skipInPage bool
}

type fixtureLabel struct {
	text  string
	pageX float64
}

// buildReport assembles a one-page report from images and labels.
func buildReport(imgs []fixtureImage, labels []fixtureLabel, trailerExtra string) []byte {
	b := newPDF()

	var xobjEntries, content bytes.Buffer
	for _, im := range imgs {
		fmt.Fprintf(&xobjEntries, "/%s %d 0 R", im.xName, im.num)
		if im.skipInPage {
			continue
		}
		// Each image is painted through its own q/cm/Q block, which is what
		// forces the extractor to track the graphics state.
		fmt.Fprintf(&content, "q %g 0 0 300 %g 150 cm /%s Do Q\n", im.pageW, im.pageX, im.xName)
	}

	labelTexts := make([]string, len(labels))
	for i, l := range labels {
		labelTexts[i] = l.text
		// Labels sit in their own text blocks with a translated CTM, mirroring
		// how the real reports position them.
		fmt.Fprintf(&content, "q 1 0 0 1 %g 0 cm BT /F1 12 Tf 1 0 0 1 0 480 Tm <%04X> Tj ET Q\n",
			l.pageX, i+1)
	}

	b.obj(1, "<</Type/Catalog/Pages 2 0 R>>")
	b.obj(2, "<</Type/Pages/Kids[3 0 R]/Count 1>>")
	b.obj(3, fmt.Sprintf(
		"<</Type/Page/Parent 2 0 R/MediaBox[0 0 842 595]"+
			"/Resources<</XObject<<%s>>/Font<</F1 7 0 R>>>>/Contents 4 0 R>>",
		xobjEntries.String()))
	b.flateStream(4, "", content.Bytes())
	b.obj(7, "<</Type/Font/Subtype/Type0/BaseFont/Test/ToUnicode 8 0 R>>")
	b.flateStream(8, "", toUnicodeCMap(labelTexts))

	for _, im := range imgs {
		dict := im.dict
		if dict == "" {
			dict = fmt.Sprintf(
				"/Type/XObject/Subtype/Image/Width %d/Height %d"+
					"/ColorSpace/DeviceRGB/BitsPerComponent 8", im.w, im.h)
		}
		b.flateStream(im.num, dict, im.pix)
	}
	return b.finish(trailerExtra)
}

// twoEyeReport is the common layout: OD on the left, OS on the right.
func twoEyeReport() []byte {
	return buildReport(
		[]fixtureImage{
			{num: 5, w: 300, h: 300, pix: discRaster(300, 300, 150, 150, 100), xName: "X1", pageX: 60, pageW: 300},
			{num: 6, w: 300, h: 300, pix: discRaster(300, 300, 150, 150, 100), xName: "X2", pageX: 460, pageW: 300},
		},
		[]fixtureLabel{{text: "OD(R)", pageX: 70}, {text: "OS(L)", pageX: 470}},
		"")
}

// oneEyeReport is the layout that defeats geometry: a single wide raster with
// the disc dead centre, identical whichever eye it is.
func oneEyeReport(label string) []byte {
	return buildReport(
		[]fixtureImage{
			{num: 5, w: 800, h: 300, pix: discRaster(800, 300, 400, 150, 100), xName: "X1", pageX: 60, pageW: 700},
		},
		[]fixtureLabel{{text: label, pageX: 400}},
		"")
}
