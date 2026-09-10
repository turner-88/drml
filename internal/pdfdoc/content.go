package pdfdoc

import (
	"strings"
)

// matrix is a PDF transformation matrix [a b c d e f], mapping
// (x,y) to (a*x + c*y + e, b*x + d*y + f).
type matrix [6]float64

var identity = matrix{1, 0, 0, 1, 0, 0}

// mul returns m applied first, then n.
func mul(m, n matrix) matrix {
	return matrix{
		m[0]*n[0] + m[1]*n[2],
		m[0]*n[1] + m[1]*n[3],
		m[2]*n[0] + m[3]*n[2],
		m[2]*n[1] + m[3]*n[3],
		m[4]*n[0] + m[5]*n[2] + n[4],
		m[4]*n[1] + m[5]*n[3] + n[5],
	}
}

func (m matrix) apply(x, y float64) (float64, float64) {
	return m[0]*x + m[2]*y + m[4], m[1]*x + m[3]*y + m[5]
}

// placement is one image drawn on the page, in absolute page coordinates.
//
// An image XObject is painted into the unit square, so the CTM in force at the
// "Do" carries its whole position and size.
type placement struct {
	name       Name
	xMin, xMax float64
	yMin, yMax float64
}

func (p placement) contains(x float64) bool { return x >= p.xMin && x <= p.xMax }

// textRun is the text of one BT/ET block with the position of its first glyph.
type textRun struct {
	text string
	x, y float64
}

// pageContent is what the content-stream walk recovers.
type pageContent struct {
	images []placement
	runs   []textRun
}

// maxContentBytes bounds a content stream. Report content streams are a few
// kilobytes; this stops a hostile file from inflating one into memory.
const maxContentBytes = 8 << 20

// walkContent interprets the page's content stream, recording where each image
// is painted and what text sits where.
//
// Tracking the graphics state is not optional here. In a two-eye report both
// eye labels carry a nearly identical local text matrix and only separate into
// left and right once the enclosing q/cm/Q blocks are applied.
func (d *Doc) walkContent(page Dict) (*pageContent, error) {
	data, err := d.pageContentBytes(page)
	if err != nil {
		return nil, err
	}
	res, _ := d.DictOf(d.Inherited(page, "Resources"))
	fonts := d.fontCMaps(res)

	pc := &pageContent{}
	l := &lexer{b: data}

	ctm := identity
	var stack []matrix
	var operands []any

	// Text state, valid between BT and ET.
	inText := false
	tm, tlm := identity, identity
	var curFont *cmap
	var runText strings.Builder
	var runX, runY float64
	runStarted := false

	flushRun := func() {
		if runStarted && runText.Len() > 0 {
			pc.runs = append(pc.runs, textRun{text: runText.String(), x: runX, y: runY})
		}
		runText.Reset()
		runStarted = false
	}

	numArg := func(i int) float64 {
		// Operands are read positionally from the end, which is how PDF
		// operators take them.
		if i >= len(operands) {
			return 0
		}
		f, _ := operands[len(operands)-1-i].(float64)
		return f
	}

	showText := func(raw String) {
		if !inText {
			return
		}
		if !runStarted {
			runX, runY = mul(tm, ctm).apply(0, 0)
			runStarted = true
		}
		runText.WriteString(curFont.decode(raw))
	}

	for {
		l.skipSpace()
		if l.pos >= len(l.b) {
			break
		}
		c := l.b[l.pos]

		// Operands parse as ordinary objects; anything else is an operator.
		if c == '/' || c == '[' || c == '(' || c == '<' || c == '+' || c == '-' || c == '.' ||
			(c >= '0' && c <= '9') {
			obj, err := l.object()
			if err != nil {
				// A malformed operand should not abandon the page: skip the
				// byte and resynchronise on the next token.
				l.pos++
				operands = operands[:0]
				continue
			}
			operands = append(operands, obj)
			if len(operands) > 32 {
				operands = operands[len(operands)-32:]
			}
			continue
		}

		op := l.keyword()
		if op == "" {
			l.pos++
			continue
		}

		switch op {
		case "q":
			stack = append(stack, ctm)
		case "Q":
			if n := len(stack); n > 0 {
				ctm = stack[n-1]
				stack = stack[:n-1]
			}
		case "cm":
			m := matrix{numArg(5), numArg(4), numArg(3), numArg(2), numArg(1), numArg(0)}
			ctm = mul(m, ctm)

		case "BT":
			inText = true
			tm, tlm = identity, identity
			flushRun()
		case "ET":
			flushRun()
			inText = false
		case "Tf":
			if len(operands) >= 2 {
				if n, ok := operands[len(operands)-2].(Name); ok {
					curFont = fonts[n]
				}
			}
		case "Tm":
			tm = matrix{numArg(5), numArg(4), numArg(3), numArg(2), numArg(1), numArg(0)}
			tlm = tm
		case "Td":
			tlm = mul(matrix{1, 0, 0, 1, numArg(1), numArg(0)}, tlm)
			tm = tlm
		case "TD":
			tlm = mul(matrix{1, 0, 0, 1, numArg(1), numArg(0)}, tlm)
			tm = tlm
		case "T*":
			tlm = mul(matrix{1, 0, 0, 1, 0, 0}, tlm)
			tm = tlm
		case "Tj", "'", "\"":
			if len(operands) > 0 {
				if s, ok := operands[len(operands)-1].(String); ok {
					showText(s)
				}
			}
		case "TJ":
			if len(operands) > 0 {
				if arr, ok := operands[len(operands)-1].(Array); ok {
					for _, el := range arr {
						if s, ok := el.(String); ok {
							showText(s)
						}
					}
				}
			}

		case "Do":
			if len(operands) > 0 {
				if n, ok := operands[len(operands)-1].(Name); ok {
					pc.images = append(pc.images, placementFor(n, ctm))
				}
			}

		case "BI":
			// Inline image: its binary payload is not PDF syntax, so skip to EI
			// rather than letting the tokenizer wander into it.
			if i := indexEI(l.b[l.pos:]); i >= 0 {
				l.pos += i
			} else {
				l.pos = len(l.b)
			}
		}
		operands = operands[:0]
	}
	flushRun()
	return pc, nil
}

// placementFor maps the unit square through the CTM and takes its bounds.
func placementFor(name Name, ctm matrix) placement {
	xs := make([]float64, 0, 4)
	ys := make([]float64, 0, 4)
	for _, pt := range [][2]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}} {
		x, y := ctm.apply(pt[0], pt[1])
		xs = append(xs, x)
		ys = append(ys, y)
	}
	p := placement{name: name, xMin: xs[0], xMax: xs[0], yMin: ys[0], yMax: ys[0]}
	for i := 1; i < 4; i++ {
		p.xMin = min(p.xMin, xs[i])
		p.xMax = max(p.xMax, xs[i])
		p.yMin = min(p.yMin, ys[i])
		p.yMax = max(p.yMax, ys[i])
	}
	return p
}

func indexEI(b []byte) int {
	for i := 0; i+1 < len(b); i++ {
		if b[i] == 'E' && b[i+1] == 'I' && (i == 0 || isSpace(b[i-1])) {
			return i + 2
		}
	}
	return -1
}

// pageContentBytes concatenates the page's content streams.
func (d *Doc) pageContentBytes(page Dict) ([]byte, error) {
	var out []byte
	appendStream := func(v any) {
		s, ok := d.StreamOf(v)
		if !ok {
			return
		}
		b, err := d.decodeAll(s, maxContentBytes)
		if err != nil {
			return
		}
		out = append(out, b...)
		out = append(out, '\n')
	}

	switch c := d.Resolve(page["Contents"]).(type) {
	case *Stream:
		appendStream(c)
	case Array:
		for _, el := range c {
			appendStream(el)
		}
	}
	if len(out) == 0 {
		return nil, ErrNoFundus
	}
	return out, nil
}

// cmap maps glyph codes to text via a font's ToUnicode CMap.
type cmap struct {
	single  map[uint32]string
	codeLen int
}

// decode turns a show-text string into readable text. A font without a
// ToUnicode CMap yields nothing rather than mojibake: the caller only looks for
// the ASCII eye markers, and a wrong guess there is worse than no answer.
func (c *cmap) decode(raw String) string {
	if c == nil || len(c.single) == 0 {
		return ""
	}
	var sb strings.Builder
	n := c.codeLen
	if n < 1 || n > 4 {
		n = 2
	}
	for i := 0; i+n <= len(raw); i += n {
		var code uint32
		for j := 0; j < n; j++ {
			code = code<<8 | uint32(raw[i+j])
		}
		sb.WriteString(c.single[code])
	}
	return sb.String()
}

// fontCMaps builds a ToUnicode CMap per font resource name.
func (d *Doc) fontCMaps(res Dict) map[Name]*cmap {
	out := map[Name]*cmap{}
	if res == nil {
		return out
	}
	fonts, ok := d.DictOf(res["Font"])
	if !ok {
		return out
	}
	for name, ref := range fonts {
		fd, ok := d.DictOf(ref)
		if !ok {
			continue
		}
		s, ok := d.StreamOf(fd["ToUnicode"])
		if !ok {
			continue
		}
		b, err := d.decodeAll(s, 1<<20)
		if err != nil {
			continue
		}
		out[name] = parseCMap(b)
	}
	return out
}

// parseCMap reads the bfchar/bfrange sections of a ToUnicode CMap.
func parseCMap(b []byte) *cmap {
	c := &cmap{single: map[uint32]string{}, codeLen: 2}
	l := &lexer{b: b}

	var operands []any
	mode := ""
	sawCodespace := false

	beCode := func(v any) (uint32, int, bool) {
		s, ok := v.(String)
		if !ok || len(s) == 0 || len(s) > 4 {
			return 0, 0, false
		}
		var out uint32
		for _, by := range s {
			out = out<<8 | uint32(by)
		}
		return out, len(s), true
	}
	beText := func(v any) (string, bool) {
		s, ok := v.(String)
		if !ok || len(s) == 0 {
			return "", false
		}
		// ToUnicode destinations are UTF-16BE.
		var sb strings.Builder
		for i := 0; i+1 < len(s); i += 2 {
			u := rune(uint16(s[i])<<8 | uint16(s[i+1]))
			if u >= 0xD800 && u <= 0xDFFF {
				continue // surrogate halves are irrelevant to ASCII markers
			}
			sb.WriteRune(u)
		}
		return sb.String(), true
	}

	for {
		l.skipSpace()
		if l.pos >= len(l.b) {
			break
		}
		ch := l.b[l.pos]
		if ch == '<' || ch == '[' || ch == '(' || ch == '/' || ch == '+' || ch == '-' || ch == '.' ||
			(ch >= '0' && ch <= '9') {
			obj, err := l.object()
			if err != nil {
				l.pos++
				continue
			}
			operands = append(operands, obj)
			continue
		}

		switch kw := l.keyword(); kw {
		case "":
			l.pos++
		case "begincodespacerange", "beginbfchar", "beginbfrange":
			mode = kw
			operands = operands[:0]
		case "endcodespacerange":
			for i := 0; i+1 < len(operands); i += 2 {
				if _, n, ok := beCode(operands[i]); ok && !sawCodespace {
					c.codeLen = n
					sawCodespace = true
				}
			}
			mode, operands = "", operands[:0]
		case "endbfchar":
			for i := 0; i+1 < len(operands); i += 2 {
				src, n, ok := beCode(operands[i])
				if !ok {
					continue
				}
				if !sawCodespace {
					c.codeLen = n
				}
				if txt, ok := beText(operands[i+1]); ok {
					c.single[src] = txt
				}
			}
			mode, operands = "", operands[:0]
		case "endbfrange":
			for i := 0; i+2 < len(operands); i += 3 {
				lo, n, ok1 := beCode(operands[i])
				hi, _, ok2 := beCode(operands[i+1])
				if !ok1 || !ok2 || hi < lo || hi-lo > 65535 {
					continue
				}
				if !sawCodespace {
					c.codeLen = n
				}
				switch dst := operands[i+2].(type) {
				case String:
					base, ok := beText(dst)
					if !ok || base == "" {
						continue
					}
					r := []rune(base)
					for code := lo; code <= hi; code++ {
						shifted := make([]rune, len(r))
						copy(shifted, r)
						shifted[len(shifted)-1] += rune(code - lo)
						c.single[code] = string(shifted)
					}
				case Array:
					for k, el := range dst {
						if uint32(k) > hi-lo {
							break
						}
						if txt, ok := beText(el); ok {
							c.single[lo+uint32(k)] = txt
						}
					}
				}
			}
			mode, operands = "", operands[:0]
		default:
			if mode == "" {
				operands = operands[:0]
			}
		}
	}
	return c
}
