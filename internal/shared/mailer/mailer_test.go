package mailer

import (
	"io"
	"mime/quotedprintable"
	"net/mail"
	"strings"
	"testing"
	"time"
)

var testFrom = mail.Address{Name: "DRML", Address: "noreply@drml.test"}

func TestBuildMessageRejectsHeaderInjection(t *testing.T) {
	cases := []struct{ to, subject string }{
		{"a@b.test", "Reset\r\nBcc: victim@evil.test"},
		{"a@b.test\r\nBcc: victim@evil.test", "Reset"},
		{"a@b.test", "Reset\nX: y"},
	}
	for _, c := range cases {
		if _, err := buildMessage(testFrom, c.to, c.subject, "body", time.Now()); err == nil {
			t.Errorf("buildMessage(to=%q, subject=%q) accepted a line break", c.to, c.subject)
		}
	}
}

func TestBuildMessageRejectsBadRecipient(t *testing.T) {
	for _, to := range []string{"", "not-an-address", "Budi <budi@b.test>"} {
		if _, err := buildMessage(testFrom, to, "s", "b", time.Now()); err == nil {
			t.Errorf("buildMessage accepted recipient %q", to)
		}
	}
}

func TestBuildMessageEncoding(t *testing.T) {
	from := mail.Address{Name: "Klinik Mata Ä", Address: "noreply@drml.test"}
	body := "Halo Budi,\n\nTautan: https://drml.test/reset-password?token=1.2.abc=def\n" +
		strings.Repeat("x", 120) + "\n"
	raw, err := buildMessage(from, "budi@b.test", "Atur ulang password — DRML", body, time.Now())
	if err != nil {
		t.Fatal(err)
	}

	msg, err := mail.ReadMessage(strings.NewReader(string(raw)))
	if err != nil {
		t.Fatalf("not a parseable message: %v", err)
	}
	if got := msg.Header.Get("Subject"); !strings.HasPrefix(got, "=?utf-8?q?") {
		t.Errorf("non-ASCII subject not Q-encoded: %q", got)
	}
	if got, err := msg.Header.AddressList("From"); err != nil || got[0].Name != from.Name {
		t.Errorf("From = %v (%v), want name %q", got, err, from.Name)
	}
	if msg.Header.Get("Message-ID") == "" || msg.Header.Get("Date") == "" {
		t.Error("missing Message-ID or Date")
	}
	if strings.Contains(string(raw), "\n") && strings.Count(string(raw), "\r\n") != strings.Count(string(raw), "\n") {
		t.Error("message contains a bare LF; SMTP requires CRLF")
	}

	decoded, err := io.ReadAll(quotedprintable.NewReader(msg.Body))
	if err != nil {
		t.Fatal(err)
	}
	want := strings.ReplaceAll(body, "\n", "\r\n")
	if string(decoded) != want {
		t.Errorf("body round-trip:\n got %q\nwant %q", decoded, want)
	}
}

func TestNewSMTPValidatesFrom(t *testing.T) {
	if _, err := NewSMTP(Config{Host: "smtp.test", Port: 587, From: "not an address"}); err == nil {
		t.Error("accepted an invalid from address")
	}
	if _, err := NewSMTP(Config{Port: 587, From: "a@b.test"}); err == nil {
		t.Error("accepted an empty host")
	}
	s, err := NewSMTP(Config{Host: "smtp.test", Port: 587, From: "a@b.test", FromName: "DRML"})
	if err != nil {
		t.Fatal(err)
	}
	if s.from.Name != "DRML" {
		t.Errorf("from name = %q, want DRML", s.from.Name)
	}
}
