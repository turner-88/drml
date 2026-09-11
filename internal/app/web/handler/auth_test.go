package handler

import (
	"os"
	"testing"

	"github.com/remorac/drml/internal/shared/config"
)

func TestValidateUsername(t *testing.T) {
	cases := []struct {
		in string
		ok bool
	}{
		{"dokter1", true},
		{"ABC", true},
		{"ab", false},                     // too short
		{"123", false},                    // no letter
		{"dr.budi", false},                // symbol
		{"dr budi", false},                // space
		{"dokterñ", false},                // non-ASCII letter
		{string(make([]byte, 51)), false}, // too long
	}
	for _, c := range cases {
		if got := validateUsername(c.in) == ""; got != c.ok {
			t.Errorf("validateUsername(%q) ok = %v, want %v", c.in, got, c.ok)
		}
	}
}

// TestTemplatesParse catches a template syntax error, including in the guest
// pages, without starting the server.
func TestTemplatesParse(t *testing.T) {
	h, err := New(&config.Config{}, nil, nil, nil, nil, nil, os.DirFS("../../../.."))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"register", "forgot_password", "reset_password"} {
		if _, ok := h.guest[name]; !ok {
			t.Fatalf("%s guest template not parsed", name)
		}
	}
}
