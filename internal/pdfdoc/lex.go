package pdfdoc

import (
	"errors"
	"strconv"
)

// The PDF object model, cut down to what an IMAGEnet report actually uses.
// Anything richer (functions, patterns, shadings) is irrelevant here: the
// extractor only ever walks a page dictionary, its resources and one content
// stream.
type (
	// Name is a PDF name object, stored without its leading slash.
	Name string
	// Ref is an indirect reference, "12 0 R".
	Ref struct{ Num, Gen int }
	// Dict is a dictionary keyed by name.
	Dict map[Name]any
	// Array is a PDF array.
	Array []any
	// String is a PDF string. Reports use these only inside ToUnicode CMaps.
	String []byte
)

var errSyntax = errors.New("pdfdoc: malformed PDF syntax")

// maxDepth bounds nesting so a hostile file cannot drive the parser into a
// stack overflow with a few kilobytes of "[[[[[".
const maxDepth = 32

// lexer parses PDF objects out of a byte window. Working over a slice rather
// than a stream is deliberate: object dictionaries are small and always read
// into a bounded window, and a slice makes the one-token backtrack needed to
// tell "1 0 R" from two consecutive integers trivial.
type lexer struct {
	b   []byte
	pos int
}

func isSpace(c byte) bool {
	return c == 0 || c == '\t' || c == '\n' || c == '\f' || c == '\r' || c == ' '
}

func isDelim(c byte) bool {
	switch c {
	case '(', ')', '<', '>', '[', ']', '{', '}', '/', '%':
		return true
	}
	return false
}

func isRegular(c byte) bool { return !isSpace(c) && !isDelim(c) }

// skipSpace advances past whitespace and comments.
func (l *lexer) skipSpace() {
	for l.pos < len(l.b) {
		c := l.b[l.pos]
		switch {
		case isSpace(c):
			l.pos++
		case c == '%':
			for l.pos < len(l.b) && l.b[l.pos] != '\n' && l.b[l.pos] != '\r' {
				l.pos++
			}
		default:
			return
		}
	}
}

// keyword reads a bare regular-character run such as "obj", "stream" or "R".
func (l *lexer) keyword() string {
	l.skipSpace()
	start := l.pos
	for l.pos < len(l.b) && isRegular(l.b[l.pos]) {
		l.pos++
	}
	return string(l.b[start:l.pos])
}

// peekKeyword reads a keyword without consuming it.
func (l *lexer) peekKeyword() string {
	save := l.pos
	kw := l.keyword()
	l.pos = save
	return kw
}

// object parses one object at the current position.
func (l *lexer) object() (any, error) { return l.objectAt(0) }

func (l *lexer) objectAt(depth int) (any, error) {
	if depth > maxDepth {
		return nil, errSyntax
	}
	l.skipSpace()
	if l.pos >= len(l.b) {
		return nil, errSyntax
	}

	switch c := l.b[l.pos]; {
	case c == '<':
		if l.pos+1 < len(l.b) && l.b[l.pos+1] == '<' {
			return l.dict(depth)
		}
		return l.hexString()
	case c == '(':
		return l.literalString()
	case c == '[':
		return l.array(depth)
	case c == '/':
		return l.name()
	case c == '+' || c == '-' || c == '.' || (c >= '0' && c <= '9'):
		return l.numberOrRef()
	default:
		switch kw := l.keyword(); kw {
		case "true":
			return true, nil
		case "false":
			return false, nil
		case "null":
			return nil, nil
		case "":
			return nil, errSyntax
		default:
			// An unexpected bare keyword ends the object cleanly rather than
			// derailing the whole document.
			return nil, errSyntax
		}
	}
}

func (l *lexer) dict(depth int) (any, error) {
	l.pos += 2 // "<<"
	d := Dict{}
	for {
		l.skipSpace()
		if l.pos >= len(l.b) {
			return nil, errSyntax
		}
		if l.b[l.pos] == '>' {
			if l.pos+1 < len(l.b) && l.b[l.pos+1] == '>' {
				l.pos += 2
				return d, nil
			}
			return nil, errSyntax
		}
		if l.b[l.pos] != '/' {
			return nil, errSyntax
		}
		key, err := l.name()
		if err != nil {
			return nil, err
		}
		val, err := l.objectAt(depth + 1)
		if err != nil {
			return nil, err
		}
		d[key.(Name)] = val
	}
}

func (l *lexer) array(depth int) (any, error) {
	l.pos++ // "["
	a := Array{}
	for {
		l.skipSpace()
		if l.pos >= len(l.b) {
			return nil, errSyntax
		}
		if l.b[l.pos] == ']' {
			l.pos++
			return a, nil
		}
		v, err := l.objectAt(depth + 1)
		if err != nil {
			return nil, err
		}
		a = append(a, v)
	}
}

func (l *lexer) name() (any, error) {
	l.pos++ // "/"
	var out []byte
	for l.pos < len(l.b) && isRegular(l.b[l.pos]) {
		c := l.b[l.pos]
		if c == '#' && l.pos+2 < len(l.b) {
			if hi, ok1 := hexVal(l.b[l.pos+1]); ok1 {
				if lo, ok2 := hexVal(l.b[l.pos+2]); ok2 {
					out = append(out, hi<<4|lo)
					l.pos += 3
					continue
				}
			}
		}
		out = append(out, c)
		l.pos++
	}
	return Name(out), nil
}

// numberOrRef parses a number, promoting "N G R" to a Ref.
func (l *lexer) numberOrRef() (any, error) {
	save := l.pos
	tok := l.keyword()
	n, err := strconv.ParseFloat(tok, 64)
	if err != nil {
		return nil, errSyntax
	}
	isInt := true
	for _, c := range tok {
		if c < '0' || c > '9' {
			isInt = false
			break
		}
	}
	if isInt {
		// Look ahead for "G R"; rewind if this is just two adjacent numbers.
		mark := l.pos
		genTok := l.keyword()
		if gen, gerr := strconv.Atoi(genTok); gerr == nil {
			if l.keyword() == "R" {
				return Ref{Num: int(n), Gen: gen}, nil
			}
		}
		l.pos = mark
	}
	_ = save
	return n, nil
}

func hexVal(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}

func (l *lexer) hexString() (any, error) {
	l.pos++ // "<"
	var out []byte
	var cur byte
	half := false
	for l.pos < len(l.b) {
		c := l.b[l.pos]
		l.pos++
		if c == '>' {
			if half {
				out = append(out, cur<<4)
			}
			return String(out), nil
		}
		v, ok := hexVal(c)
		if !ok {
			continue // whitespace inside hex strings is legal
		}
		if half {
			out = append(out, cur<<4|v)
			half = false
		} else {
			cur = v
			half = true
		}
	}
	return nil, errSyntax
}

func (l *lexer) literalString() (any, error) {
	l.pos++ // "("
	var out []byte
	depth := 1
	for l.pos < len(l.b) {
		c := l.b[l.pos]
		l.pos++
		switch c {
		case '\\':
			if l.pos >= len(l.b) {
				return nil, errSyntax
			}
			e := l.b[l.pos]
			l.pos++
			switch e {
			case 'n':
				out = append(out, '\n')
			case 'r':
				out = append(out, '\r')
			case 't':
				out = append(out, '\t')
			case 'b':
				out = append(out, '\b')
			case 'f':
				out = append(out, '\f')
			case '\n':
				// line continuation, emits nothing
			case '\r':
				if l.pos < len(l.b) && l.b[l.pos] == '\n' {
					l.pos++
				}
			default:
				if e >= '0' && e <= '7' {
					v := int(e - '0')
					for i := 0; i < 2 && l.pos < len(l.b) && l.b[l.pos] >= '0' && l.b[l.pos] <= '7'; i++ {
						v = v*8 + int(l.b[l.pos]-'0')
						l.pos++
					}
					out = append(out, byte(v))
				} else {
					out = append(out, e)
				}
			}
		case '(':
			depth++
			out = append(out, c)
		case ')':
			depth--
			if depth == 0 {
				return String(out), nil
			}
			out = append(out, c)
		default:
			out = append(out, c)
		}
	}
	return nil, errSyntax
}
