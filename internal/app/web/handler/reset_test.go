package handler

import (
	"strings"
	"testing"
	"time"

	db "github.com/remorac/drml/internal/database/sqlc"
)

const testSecret = "test-secret"

func testUser() *db.User {
	return &db.User{ID: 7, Email: "budi@klinik.test", PasswordHash: "$2a$10$oldhash"}
}

func TestResetTokenRoundTrip(t *testing.T) {
	u := testUser()
	now := time.Now()
	tok := resetToken(testSecret, u, now.Add(resetTokenTTL))
	if !verifyResetToken(testSecret, tok, u, now) {
		t.Fatal("fresh token rejected")
	}
	if id, _, _, ok := parseResetToken(tok); !ok || id != u.ID {
		t.Fatalf("parseResetToken = %d, %v", id, ok)
	}
}

func TestResetTokenExpires(t *testing.T) {
	u := testUser()
	now := time.Now()
	tok := resetToken(testSecret, u, now.Add(resetTokenTTL))
	if verifyResetToken(testSecret, tok, u, now.Add(resetTokenTTL+time.Second)) {
		t.Fatal("expired token accepted")
	}
}

// Single use: the MAC covers the password hash, so once a new password is set
// the link that set it, and every other outstanding link, stops working.
func TestResetTokenDiesWithPasswordChange(t *testing.T) {
	u := testUser()
	now := time.Now()
	tok := resetToken(testSecret, u, now.Add(resetTokenTTL))
	u.PasswordHash = "$2a$10$newhash"
	if verifyResetToken(testSecret, tok, u, now) {
		t.Fatal("token still valid after the password changed")
	}
}

func TestResetTokenDiesWithEmailChange(t *testing.T) {
	u := testUser()
	now := time.Now()
	tok := resetToken(testSecret, u, now.Add(resetTokenTTL))
	u.Email = "other@klinik.test"
	if verifyResetToken(testSecret, tok, u, now) {
		t.Fatal("token still valid after the email changed")
	}
}

func TestResetTokenRejectsTampering(t *testing.T) {
	u := testUser()
	now := time.Now()
	tok := resetToken(testSecret, u, now.Add(resetTokenTTL))
	parts := strings.Split(tok, ".")

	other := *u
	other.ID = 8
	cases := map[string]struct {
		tok string
		u   *db.User
	}{
		"other secret":    {resetToken("another-secret", u, now.Add(resetTokenTTL)), u},
		"id swapped":      {"8." + parts[1] + "." + parts[2], &other},
		"expiry extended": {parts[0] + ".9999999999." + parts[2], u},
		"mac flipped":     {parts[0] + "." + parts[1] + "." + strings.Repeat("A", len(parts[2])), u},
		"wrong account":   {tok, &other},
		"empty":           {"", u},
		"two parts":       {parts[0] + "." + parts[1], u},
		"bad base64":      {parts[0] + "." + parts[1] + ".!!!", u},
		"negative id":     {"-7." + parts[1] + "." + parts[2], u},
	}
	for name, c := range cases {
		if verifyResetToken(testSecret, c.tok, c.u, now) {
			t.Errorf("%s: tampered token accepted", name)
		}
	}
}

func TestValidateEmail(t *testing.T) {
	cases := []struct {
		in string
		ok bool
	}{
		{"budi@klinik.test", true},
		{"dr.budi+drml@klinik.co.id", true},
		{"", false},
		{"budi", false},
		{"Budi <budi@klinik.test>", false},
		{"budi@klinik.test\r\nBcc: x@y.test", false},
		{strings.Repeat("a", 94) + "@x.test", false}, // 101 characters
	}
	for _, c := range cases {
		if got := validateEmail(c.in) == ""; got != c.ok {
			t.Errorf("validateEmail(%q) ok = %v, want %v", c.in, got, c.ok)
		}
	}
}
