// Package pdfdoc extracts fundus photographs from single-page IMAGEnet
// screening reports.
//
// The reports are Chrome print-to-PDF output: one page, a classic xref table,
// and the fundus photos embedded as FlateDecode RGB rasters with black
// surrounds. That last detail is what makes this cheap - the photos are plain
// zlib-compressed pixels, so no PDF rasterizer, no cgo, and no extra system
// packages are needed, and the images come out at their original framing rather
// than re-rendered at print DPI.
//
// Laterality is read from the report's "OD(R)" / "OS(L)" text, never inferred
// from where an image sits on the page. A report carrying a single eye places
// it dead centre whether it is the left or the right one, so position genuinely
// cannot tell them apart, and labelling a patient's eye wrongly is not a
// mistake worth risking to save a text parse.
package pdfdoc

import (
	"errors"
	"fmt"
	"image"
	"io"
	"sort"
	"strings"
)

// Eye is the laterality read from the report label.
type Eye string

const (
	// EyeOD is the right eye, oculus dexter.
	EyeOD Eye = "OD"
	// EyeOS is the left eye, oculus sinister.
	EyeOS Eye = "OS"
)

// Errors callers distinguish. Everything else is an unexpected parse failure.
var (
	// ErrNotSupportedPDF means the file is a PDF this extractor cannot read -
	// encrypted, built from object streams, or structurally broken.
	ErrNotSupportedPDF = errors.New("pdfdoc: unsupported PDF")
	// ErrNoFundus means no fundus photograph was found in the document.
	ErrNoFundus = errors.New("pdfdoc: no fundus image found")
	// ErrAmbiguousEye means the OD/OS labels could not be matched to the
	// images one-for-one, so laterality cannot be established.
	ErrAmbiguousEye = errors.New("pdfdoc: eye labels do not match the images")
)

// Fundus is one extracted eye.
type Fundus struct {
	Eye Eye
	// Img is the fundus disc at the resolution the report stores it, cropped
	// from the page raster. The surround is black, as it is in the source.
	Img image.Image
}

// disc is a candidate fundus found in one image XObject, with the page-space
// span it occupies.
type disc struct {
	img        *image.NRGBA
	xMin, xMax float64
}

// Extract returns one Fundus per eye in a single-page IMAGEnet report.
//
// ra is read lazily, so passing the multipart temp file keeps the PDF off the
// heap entirely.
func Extract(ra io.ReaderAt, size int64) ([]Fundus, error) {
	doc, err := Open(ra, size)
	if err != nil {
		return nil, err
	}
	page, err := doc.Page()
	if err != nil {
		return nil, err
	}
	content, err := doc.walkContent(page)
	if err != nil {
		return nil, err
	}

	res, _ := doc.DictOf(doc.Inherited(page, "Resources"))
	xobjs, _ := doc.DictOf(res["XObject"])

	var discs []disc
	for _, p := range content.images {
		s, ok := doc.StreamOf(xobjs[p.name])
		if !ok {
			continue
		}
		if sub, _ := doc.NameOf(s.Dict["Subtype"]); sub != "Image" {
			continue
		}
		// The clinic logo is a small DCTDecode image and the alpha channels are
		// DeviceGray; both fail here, which is exactly what we want. A decode
		// failure is never fatal - the next image may still be a fundus.
		r, k, err := doc.decimate(s)
		if err != nil {
			continue
		}
		regions := r.discRegions()
		if len(regions) == 0 {
			continue
		}
		// The coarse pass above only located the discs. Read the stream again to
		// lift them out at full resolution, which is what the scan is kept as:
		// the source report is not retained by default.
		imgs, err := doc.cropNative(s, regions, k)
		if err != nil {
			continue
		}
		for i, rect := range regions {
			// Map the region back into page space so it can be matched against
			// the labels, which are positioned on the page rather than on the
			// raster.
			span := p.xMax - p.xMin
			discs = append(discs, disc{
				img:  imgs[i],
				xMin: p.xMin + float64(rect.Min.X)/float64(r.w)*span,
				xMax: p.xMin + float64(rect.Max.X)/float64(r.w)*span,
			})
		}
	}
	if len(discs) == 0 {
		return nil, ErrNoFundus
	}

	labels := eyeLabels(content.runs)
	out, err := assign(discs, labels)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// eyeLabel is an OD/OS marker found on the page.
type eyeLabel struct {
	eye Eye
	x   float64
}

// eyeLabels picks the eye markers out of the page text.
func eyeLabels(runs []textRun) []eyeLabel {
	var out []eyeLabel
	for _, r := range runs {
		if eye, ok := eyeFromText(r.text); ok {
			out = append(out, eyeLabel{eye: eye, x: r.x})
		}
	}
	return out
}

// eyeFromText recognises a standalone "OD"/"OS" marker.
//
// The length bound and the letter check keep it from firing on ordinary prose
// or a patient name that happens to begin with those letters.
func eyeFromText(text string) (Eye, bool) {
	t := strings.TrimSpace(text)
	if t == "" || len(t) > 24 {
		return "", false
	}
	up := strings.ToUpper(t)
	var eye Eye
	switch {
	case strings.HasPrefix(up, "OD"):
		eye = EyeOD
	case strings.HasPrefix(up, "OS"):
		eye = EyeOS
	default:
		return "", false
	}
	if len(up) > 2 {
		if c := up[2]; c >= 'A' && c <= 'Z' {
			return "", false
		}
	}
	return eye, true
}

// assign pairs each disc with an eye label.
//
// Matching is by page position - a label belongs to the disc whose horizontal
// span contains it - with a left-to-right fallback when nothing contains it.
// The single-disc case is decided by the label alone, because that is the one
// layout where position carries no information at all.
func assign(discs []disc, labels []eyeLabel) ([]Fundus, error) {
	sort.SliceStable(discs, func(i, j int) bool { return discs[i].xMin < discs[j].xMin })
	sort.SliceStable(labels, func(i, j int) bool { return labels[i].x < labels[j].x })

	if len(labels) != len(discs) {
		return nil, fmt.Errorf("%w: found %d image(s) but %d label(s)",
			ErrAmbiguousEye, len(discs), len(labels))
	}

	assigned := make([]Eye, len(discs))
	used := make([]bool, len(labels))
	for i, d := range discs {
		best := -1
		for j, l := range labels {
			if used[j] {
				continue
			}
			if l.x >= d.xMin && l.x <= d.xMax {
				best = j
				break
			}
		}
		if best < 0 {
			// Nothing sits inside this disc: fall back to reading order, which
			// is only ever reached when both counts already agree.
			for j := range labels {
				if !used[j] {
					best = j
					break
				}
			}
		}
		if best < 0 {
			return nil, ErrAmbiguousEye
		}
		used[best] = true
		assigned[i] = labels[best].eye
	}

	// Two images of the same eye means the labels were misread. Refuse rather
	// than record a scan against the wrong eye.
	if len(assigned) > 1 {
		seen := map[Eye]bool{}
		for _, e := range assigned {
			if seen[e] {
				return nil, fmt.Errorf("%w: %s matched more than once", ErrAmbiguousEye, e)
			}
			seen[e] = true
		}
	}

	out := make([]Fundus, len(discs))
	for i := range discs {
		out[i] = Fundus{Eye: assigned[i], Img: discs[i].img}
	}
	return out, nil
}
