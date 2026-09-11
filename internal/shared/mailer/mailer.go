// Package mailer sends plain-text transactional email.
//
// Standard library only: one reset link per request does not justify a
// dependency, and net/smtp covers both implicit TLS and STARTTLS.
package mailer

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"mime"
	"mime/quotedprintable"
	"net"
	"net/mail"
	"net/smtp"
	"strconv"
	"strings"
	"time"
)

// Mailer delivers one message to one recipient.
type Mailer interface {
	Send(ctx context.Context, to, subject, body string) error
}

// Config configures the SMTP relay.
type Config struct {
	Host     string
	Port     int
	Username string
	Password string
	From     string
	FromName string
}

// defaultTimeout bounds a send whose context carries no deadline, so a relay
// that accepts the connection and then stalls cannot pin a goroutine forever.
const defaultTimeout = 30 * time.Second

// SMTP sends through an SMTP relay.
type SMTP struct {
	cfg  Config
	from mail.Address
}

// NewSMTP validates the sender address up front, so a typo in SMTP_FROM fails
// at startup rather than on the first password reset.
func NewSMTP(cfg Config) (*SMTP, error) {
	if cfg.Host == "" {
		return nil, errors.New("smtp: host is required")
	}
	addr, err := mail.ParseAddress(cfg.From)
	if err != nil {
		return nil, fmt.Errorf("smtp: invalid from address %q: %w", cfg.From, err)
	}
	if cfg.FromName != "" {
		addr.Name = cfg.FromName
	}
	return &SMTP{cfg: cfg, from: *addr}, nil
}

// Send delivers the message. Port 465 is implicit TLS; any other port is
// upgraded with STARTTLS when the server offers it, and credentials are never
// sent over a connection that was not upgraded.
func (s *SMTP) Send(ctx context.Context, to, subject, body string) error {
	msg, err := buildMessage(s.from, to, subject, body, time.Now())
	if err != nil {
		return err
	}

	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(defaultTimeout)
	}
	addr := net.JoinHostPort(s.cfg.Host, strconv.Itoa(s.cfg.Port))
	tlsCfg := &tls.Config{ServerName: s.cfg.Host, MinVersion: tls.VersionTLS12}
	dialer := &net.Dialer{Deadline: deadline}

	var conn net.Conn
	if s.cfg.Port == 465 {
		conn, err = (&tls.Dialer{NetDialer: dialer, Config: tlsCfg}).DialContext(ctx, "tcp", addr)
	} else {
		conn, err = dialer.DialContext(ctx, "tcp", addr)
	}
	if err != nil {
		return fmt.Errorf("smtp: dial %s: %w", addr, err)
	}
	// net/smtp has no context support; the deadline covers the whole exchange.
	if err := conn.SetDeadline(deadline); err != nil {
		conn.Close()
		return fmt.Errorf("smtp: set deadline: %w", err)
	}

	c, err := smtp.NewClient(conn, s.cfg.Host)
	if err != nil {
		conn.Close()
		return fmt.Errorf("smtp: handshake: %w", err)
	}
	defer c.Close()

	if s.cfg.Port != 465 {
		if ok, _ := c.Extension("STARTTLS"); ok {
			if err := c.StartTLS(tlsCfg); err != nil {
				return fmt.Errorf("smtp: starttls: %w", err)
			}
		} else if s.cfg.Username != "" {
			return errors.New("smtp: server does not offer STARTTLS; refusing to send credentials in clear")
		}
	}
	if s.cfg.Username != "" {
		if err := c.Auth(smtp.PlainAuth("", s.cfg.Username, s.cfg.Password, s.cfg.Host)); err != nil {
			return fmt.Errorf("smtp: auth: %w", err)
		}
	}

	if err := c.Mail(s.from.Address); err != nil {
		return fmt.Errorf("smtp: mail from: %w", err)
	}
	if err := c.Rcpt(to); err != nil {
		return fmt.Errorf("smtp: rcpt to: %w", err)
	}
	w, err := c.Data()
	if err != nil {
		return fmt.Errorf("smtp: data: %w", err)
	}
	if _, err := w.Write(msg); err != nil {
		return fmt.Errorf("smtp: write: %w", err)
	}
	if err := w.Close(); err != nil {
		return fmt.Errorf("smtp: end data: %w", err)
	}
	return c.Quit()
}

// Log writes messages to the server log instead of sending them. It exists so
// the reset flow can be exercised in development without a relay; the server
// never selects it in production, since the log would then hold live links.
type Log struct{}

func (Log) Send(_ context.Context, to, subject, body string) error {
	log.Printf("mailer: SMTP_HOST unset, logging instead of sending\nTo: %s\nSubject: %s\n\n%s", to, subject, body)
	return nil
}

var errHeaderInjection = errors.New("mailer: line break in header value")

// buildMessage renders an RFC 5322 message with a UTF-8 quoted-printable body.
func buildMessage(from mail.Address, to, subject, body string, now time.Time) ([]byte, error) {
	for _, v := range []string{from.Name, from.Address, to, subject} {
		if strings.ContainsAny(v, "\r\n") {
			return nil, errHeaderInjection
		}
	}
	rcpt, err := mail.ParseAddress(to)
	if err != nil || rcpt.Address != to {
		return nil, fmt.Errorf("mailer: invalid recipient %q", to)
	}

	var b bytes.Buffer
	header := func(k, v string) { fmt.Fprintf(&b, "%s: %s\r\n", k, v) }
	// Address.String Q-encodes a non-ASCII display name itself.
	header("From", from.String())
	header("To", rcpt.String())
	header("Subject", mime.QEncoding.Encode("utf-8", subject))
	header("Date", now.Format(time.RFC1123Z))
	header("Message-ID", messageID(from.Address))
	header("MIME-Version", "1.0")
	header("Content-Type", "text/plain; charset=utf-8")
	header("Content-Transfer-Encoding", "quoted-printable")
	b.WriteString("\r\n")

	// Text mode writes every line break as CRLF, as SMTP requires.
	qp := quotedprintable.NewWriter(&b)
	if _, err := qp.Write([]byte(body)); err != nil {
		return nil, err
	}
	if err := qp.Close(); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// messageID is unique per message and scoped to the sender's domain, so
// receiving servers do not mark the mail as suspect for lacking one.
func messageID(fromAddr string) string {
	domain := "localhost"
	if i := strings.LastIndexByte(fromAddr, '@'); i >= 0 {
		domain = fromAddr[i+1:]
	}
	var buf [16]byte
	_, _ = rand.Read(buf[:])
	return "<" + hex.EncodeToString(buf[:]) + "@" + domain + ">"
}
