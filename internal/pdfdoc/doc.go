package pdfdoc

import (
	"bytes"
	"compress/zlib"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
)

// Stream is an indirect object whose dictionary is followed by raw data.
//
// The data is addressed by offset and length rather than copied: a fundus
// raster inflates to 40-50 MB, so it must be read through the document's
// io.ReaderAt one row at a time, never buffered whole.
type Stream struct {
	Dict   Dict
	Offset int64
	Length int64
}

// Doc is a parsed PDF file.
type Doc struct {
	ra    io.ReaderAt
	size  int64
	xref  map[int]int64
	trail Dict
	cache map[int]any
}

// objWindow is the initial read window for one indirect object. Report
// dictionaries are a few hundred bytes; the retry below covers the rest.
const (
	objWindow    = 64 << 10
	objWindowMax = 1 << 20
	xrefWindow   = 1 << 20
)

// Open indexes the objects in a PDF without reading its bulk into memory.
func Open(ra io.ReaderAt, size int64) (*Doc, error) {
	if size < 8 {
		return nil, fmt.Errorf("%w: file too small", ErrNotSupportedPDF)
	}
	d := &Doc{ra: ra, size: size, xref: map[int]int64{}, cache: map[int]any{}}

	// The cross-reference table is authoritative, and matters here for a
	// specific reason: scanning the file for "N 0 obj" can match those bytes
	// inside a 40 MB compressed raster, and a false positive that shadows a
	// real object number would silently corrupt the extraction. The scan is
	// only a fallback for a file whose xref is damaged.
	if err := d.loadXref(); err != nil || len(d.xref) == 0 {
		d.scanObjects()
	}
	if len(d.xref) == 0 {
		return nil, fmt.Errorf("%w: no objects found", ErrNotSupportedPDF)
	}
	if d.trail != nil {
		if _, encrypted := d.trail["Encrypt"]; encrypted {
			return nil, fmt.Errorf("%w: encrypted documents are not supported", ErrNotSupportedPDF)
		}
	}
	return d, nil
}

func (d *Doc) window(off, n int64) ([]byte, error) {
	if off < 0 || off >= d.size {
		return nil, io.EOF
	}
	if off+n > d.size {
		n = d.size - off
	}
	b := make([]byte, n)
	got, err := d.ra.ReadAt(b, off)
	if got == 0 && err != nil {
		return nil, err
	}
	return b[:got], nil
}

var startxrefRe = regexp.MustCompile(`startxref\s+(\d+)`)

func (d *Doc) loadXref() error {
	tailLen := int64(2048)
	if tailLen > d.size {
		tailLen = d.size
	}
	tail, err := d.window(d.size-tailLen, tailLen)
	if err != nil {
		return err
	}
	m := startxrefRe.FindAllSubmatch(tail, -1)
	if len(m) == 0 {
		return errors.New("pdfdoc: no startxref")
	}
	off, err := strconv.ParseInt(string(m[len(m)-1][1]), 10, 64)
	if err != nil {
		return err
	}

	// Incremental updates chain through /Prev. Bound the walk and remember
	// visited offsets so a self-referential chain cannot loop forever.
	seen := map[int64]bool{}
	for i := 0; i < 64 && off > 0 && !seen[off]; i++ {
		seen[off] = true
		prev, err := d.parseXrefSection(off)
		if err != nil {
			return err
		}
		off = prev
	}
	return nil
}

// parseXrefSection reads one classic xref table and its trailer, returning the
// /Prev offset (0 when there is none).
func (d *Doc) parseXrefSection(off int64) (int64, error) {
	buf, err := d.window(off, xrefWindow)
	if err != nil {
		return 0, err
	}
	l := &lexer{b: buf}
	if l.keyword() != "xref" {
		// A cross-reference stream, which implies object streams we do not
		// decode. Fall back to scanning.
		return 0, errors.New("pdfdoc: not a classic xref table")
	}

	for {
		l.skipSpace()
		if l.pos >= len(l.b) {
			return 0, errSyntax
		}
		if l.peekKeyword() == "trailer" {
			l.keyword()
			obj, err := l.object()
			if err != nil {
				return 0, err
			}
			t, ok := obj.(Dict)
			if !ok {
				return 0, errSyntax
			}
			// Earlier sections in the chain must not overwrite the newest
			// trailer's /Root.
			if d.trail == nil {
				d.trail = t
			}
			if prev, ok := toInt(t["Prev"]); ok {
				return int64(prev), nil
			}
			return 0, nil
		}

		start, err := strconv.Atoi(l.keyword())
		if err != nil {
			return 0, errSyntax
		}
		count, err := strconv.Atoi(l.keyword())
		if err != nil {
			return 0, errSyntax
		}
		if count < 0 || count > 1<<22 {
			return 0, errSyntax
		}
		for i := 0; i < count; i++ {
			entryOff, err := strconv.ParseInt(l.keyword(), 10, 64)
			if err != nil {
				return 0, errSyntax
			}
			l.keyword() // generation
			kind := l.keyword()
			num := start + i
			// Only in-use entries, and never let an older section override a
			// newer one already recorded.
			if kind == "n" && entryOff > 0 && entryOff < d.size {
				if _, exists := d.xref[num]; !exists {
					d.xref[num] = entryOff
				}
			}
		}
	}
}

var objHeaderRe = regexp.MustCompile(`(?s)(\d{1,10})\s+(\d{1,5})\s+obj\b`)

// scanObjects rebuilds the index by sweeping the file for object headers. It is
// the recovery path for a damaged xref; later definitions win, matching how an
// incrementally updated file supersedes earlier revisions.
func (d *Doc) scanObjects() {
	const chunk = 1 << 20
	const overlap = 64
	for off := int64(0); off < d.size; off += chunk - overlap {
		buf, err := d.window(off, chunk)
		if err != nil {
			return
		}
		for _, m := range objHeaderRe.FindAllSubmatchIndex(buf, -1) {
			num, err := strconv.Atoi(string(buf[m[2]:m[3]]))
			if err != nil {
				continue
			}
			d.xref[num] = off + int64(m[0])
		}
		if int64(len(buf)) < chunk {
			return
		}
	}
}

// Object loads and caches one indirect object.
func (d *Doc) Object(num int) (any, error) {
	if v, ok := d.cache[num]; ok {
		return v, nil
	}
	off, ok := d.xref[num]
	if !ok {
		return nil, fmt.Errorf("pdfdoc: object %d not found", num)
	}

	var obj any
	var err error
	for _, size := range []int64{objWindow, objWindowMax} {
		obj, err = d.parseObjectAt(off, num, size)
		if err == nil {
			break
		}
	}
	if err != nil {
		return nil, err
	}
	d.cache[num] = obj
	return obj, nil
}

func (d *Doc) parseObjectAt(off int64, want int, winSize int64) (any, error) {
	buf, err := d.window(off, winSize)
	if err != nil {
		return nil, err
	}
	l := &lexer{b: buf}
	got, err := strconv.Atoi(l.keyword())
	if err != nil {
		return nil, errSyntax
	}
	l.keyword() // generation
	if l.keyword() != "obj" {
		return nil, errSyntax
	}
	if got != want {
		return nil, fmt.Errorf("pdfdoc: object %d header says %d", want, got)
	}

	obj, err := l.object()
	if err != nil {
		return nil, err
	}
	if l.peekKeyword() != "stream" {
		return obj, nil
	}

	dict, ok := obj.(Dict)
	if !ok {
		return nil, errSyntax
	}
	l.keyword()
	// The data begins after the EOL that must follow the "stream" keyword.
	if l.pos < len(l.b) && l.b[l.pos] == '\r' {
		l.pos++
	}
	if l.pos < len(l.b) && l.b[l.pos] == '\n' {
		l.pos++
	}
	length, ok := toInt(d.Resolve(dict["Length"]))
	if !ok || length < 0 {
		return nil, fmt.Errorf("pdfdoc: stream object %d has no usable /Length", want)
	}
	dataOff := off + int64(l.pos)
	if dataOff+int64(length) > d.size {
		return nil, fmt.Errorf("pdfdoc: stream object %d runs past end of file", want)
	}
	return &Stream{Dict: dict, Offset: dataOff, Length: int64(length)}, nil
}

// Resolve follows indirect references. A reference that cannot be loaded
// resolves to nil, which every caller already treats as "absent".
func (d *Doc) Resolve(v any) any {
	for i := 0; i < 16; i++ {
		ref, ok := v.(Ref)
		if !ok {
			return v
		}
		obj, err := d.Object(ref.Num)
		if err != nil {
			return nil
		}
		v = obj
	}
	return nil
}

// Trailer exposes the document trailer for catalog lookup.
func (d *Doc) Trailer() Dict { return d.trail }

// DictOf resolves v and returns it as a dictionary, including a stream's own
// dictionary.
func (d *Doc) DictOf(v any) (Dict, bool) {
	switch t := d.Resolve(v).(type) {
	case Dict:
		return t, true
	case *Stream:
		return t.Dict, true
	}
	return nil, false
}

// StreamOf resolves v and returns it as a stream.
func (d *Doc) StreamOf(v any) (*Stream, bool) {
	s, ok := d.Resolve(v).(*Stream)
	return s, ok
}

// ArrayOf resolves v and returns it as an array.
func (d *Doc) ArrayOf(v any) (Array, bool) {
	a, ok := d.Resolve(v).(Array)
	return a, ok
}

// IntOf resolves v and returns it as an int.
func (d *Doc) IntOf(v any) (int, bool) { return toInt(d.Resolve(v)) }

// NameOf resolves v and returns it as a name.
func (d *Doc) NameOf(v any) (Name, bool) {
	n, ok := d.Resolve(v).(Name)
	return n, ok
}

func toInt(v any) (int, bool) {
	f, ok := v.(float64)
	if !ok {
		return 0, false
	}
	return int(f), true
}

// filters returns the stream's filter chain as names.
func (d *Doc) filters(s *Stream) []Name {
	switch f := d.Resolve(s.Dict["Filter"]).(type) {
	case Name:
		return []Name{f}
	case Array:
		out := make([]Name, 0, len(f))
		for _, v := range f {
			if n, ok := d.NameOf(v); ok {
				out = append(out, n)
			}
		}
		return out
	}
	return nil
}

// RawReader streams the undecoded stream bytes.
func (d *Doc) RawReader(s *Stream) io.Reader {
	return io.NewSectionReader(d.ra, s.Offset, s.Length)
}

// DecodedReader streams the stream through its filter chain.
//
// Only FlateDecode and "no filter" are handled, which covers every content
// stream, CMap and fundus raster an IMAGEnet report contains. An image using
// any other filter is skipped by the caller rather than mis-decoded.
func (d *Doc) DecodedReader(s *Stream) (io.ReadCloser, error) {
	r := d.RawReader(s)
	fs := d.filters(s)
	switch len(fs) {
	case 0:
		return io.NopCloser(r), nil
	case 1:
		if fs[0] == "FlateDecode" {
			zr, err := zlib.NewReader(r)
			if err != nil {
				return nil, fmt.Errorf("pdfdoc: inflate: %w", err)
			}
			return zr, nil
		}
	}
	return nil, fmt.Errorf("pdfdoc: unsupported filter chain %v", fs)
}

// decodeAll reads a small stream fully. Used for content streams and CMaps,
// never for image data.
func (d *Doc) decodeAll(s *Stream, limit int64) ([]byte, error) {
	rc, err := d.DecodedReader(s)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, io.LimitReader(rc, limit)); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Page returns the first page dictionary.
//
// Reports are single-page by construction, so the catalog walk stops at the
// first leaf rather than building a page tree.
func (d *Doc) Page() (Dict, error) {
	if d.trail != nil {
		if root, ok := d.DictOf(d.trail["Root"]); ok {
			if pages, ok := d.DictOf(root["Pages"]); ok {
				if pg := d.firstLeaf(pages, 0); pg != nil {
					return pg, nil
				}
			}
		}
	}
	// Fall back to the lowest-numbered object that calls itself a page. This
	// also covers a file whose catalog we could not follow.
	best := -1
	for num := range d.xref {
		if best >= 0 && num >= best {
			continue
		}
		dict, ok := d.DictOf(Ref{Num: num})
		if !ok {
			continue
		}
		if n, _ := d.NameOf(dict["Type"]); n == "Page" {
			best = num
		}
	}
	if best >= 0 {
		dict, _ := d.DictOf(Ref{Num: best})
		return dict, nil
	}
	return nil, fmt.Errorf("%w: no page found", ErrNotSupportedPDF)
}

func (d *Doc) firstLeaf(node Dict, depth int) Dict {
	if depth > maxDepth {
		return nil
	}
	if n, _ := d.NameOf(node["Type"]); n == "Page" {
		return node
	}
	kids, ok := d.ArrayOf(node["Kids"])
	if !ok {
		return nil
	}
	for _, kid := range kids {
		kd, ok := d.DictOf(kid)
		if !ok {
			continue
		}
		if leaf := d.firstLeaf(kd, depth+1); leaf != nil {
			return leaf
		}
	}
	return nil
}

// Inherited looks a key up on the page, walking /Parent for the attributes a
// page may inherit from its ancestors.
func (d *Doc) Inherited(page Dict, key Name) any {
	node := page
	for i := 0; i < maxDepth; i++ {
		if v, ok := node[key]; ok {
			return v
		}
		parent, ok := d.DictOf(node["Parent"])
		if !ok {
			return nil
		}
		node = parent
	}
	return nil
}
