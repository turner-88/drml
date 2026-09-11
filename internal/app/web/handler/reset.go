package handler

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"golang.org/x/crypto/bcrypt"

	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/shared/util"
)

// resetTokenTTL bounds how long an emailed link works. Kept short because the
// mailbox it sits in may be readable by more people than the account holder.
const resetTokenTTL = time.Hour

// resetSendTimeout bounds the lookup and SMTP exchange, which run detached
// from the request that triggered them.
const resetSendTimeout = 30 * time.Second

const resetMailSubject = "Atur ulang password akun DRML"

const resetMailBody = `Halo %s,

Kami menerima permintaan untuk mengatur ulang password akun DRML Anda
(username: %s). Buka tautan berikut untuk membuat password baru:

%s

Tautan ini berlaku selama %d menit dan hanya dapat digunakan satu kali.

Jika Anda tidak meminta pengaturan ulang password, abaikan email ini.
Password Anda tidak akan berubah.

— DRML
`

// ForgotPasswordPage renders the request-a-link form.
func (h *Handler) ForgotPasswordPage(w http.ResponseWriter, r *http.Request) {
	if h.mailer == nil {
		http.NotFound(w, r)
		return
	}
	data := map[string]any{
		"Title":      "Lupa Password",
		"Sent":       r.URL.Query().Get("sent") == "1",
		"TTLMinutes": int(resetTokenTTL.Minutes()),
	}
	if r.URL.Query().Get("blocked") == "1" {
		data["Error"] = "Terlalu banyak percobaan reset password. Silakan coba lagi dalam 1 jam."
	}
	h.renderGuest(w, r, "forgot_password", data)
}

// ForgotPassword emails a reset link if the address belongs to an account.
//
// The reply is identical either way, and the lookup and send happen after the
// response is written, so neither the wording nor the timing tells a visitor
// which addresses are registered.
func (h *Handler) ForgotPassword(w http.ResponseWriter, r *http.Request) {
	if h.mailer == nil {
		http.NotFound(w, r)
		return
	}
	email := strings.TrimSpace(r.FormValue("email"))
	if msg := validateEmail(email); msg != "" {
		h.renderGuest(w, r, "forgot_password", map[string]any{
			"Title":      "Lupa Password",
			"Error":      msg,
			"Email":      email,
			"TTLMinutes": int(resetTokenTTL.Minutes()),
		})
		return
	}

	go h.sendResetLink(email)

	// Redirect so a refresh does not send a second email.
	http.Redirect(w, r, "/forgot-password?sent=1", http.StatusSeeOther)
}

// sendResetLink looks up the account and mails it a link. It runs on its own
// goroutine, outside chi's Recoverer, so it recovers its own panics rather than
// taking the server down.
func (h *Handler) sendResetLink(email string) {
	defer func() {
		if p := recover(); p != nil {
			log.Printf("password reset: panic: %v", p)
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), resetSendTimeout)
	defer cancel()

	u, err := h.store.GetUserByEmail(ctx, email)
	if err != nil {
		if !errors.Is(err, sql.ErrNoRows) {
			log.Printf("password reset: lookup: %v", err)
		}
		return
	}
	if u.SuspendedAt.Valid {
		log.Printf("password reset: not sent, user id %d is suspended", u.ID)
		return
	}

	tok := resetToken(h.cfg.JWT.SecretKey, &u, time.Now().Add(resetTokenTTL))
	link := strings.TrimRight(h.cfg.AppURL, "/") + "/reset-password?token=" + url.QueryEscape(tok)
	name := u.Name.String
	if name == "" {
		name = u.Username
	}
	body := fmt.Sprintf(resetMailBody, name, u.Username, link, int(resetTokenTTL.Minutes()))

	// The stored address, not the typed one: they can differ in case.
	if err := h.mailer.Send(ctx, u.Email, resetMailSubject, body); err != nil {
		log.Printf("password reset: send to user id %d: %v", u.ID, err)
		return
	}
	log.Printf("password reset: link sent to user id %d", u.ID)
}

// ResetPasswordPage renders the new-password form for a valid link.
func (h *Handler) ResetPasswordPage(w http.ResponseWriter, r *http.Request) {
	if h.mailer == nil {
		http.NotFound(w, r)
		return
	}
	// The token is in this page's URL; keep it out of the Referer sent to the
	// font CDN the guest layout loads.
	w.Header().Set("Referrer-Policy", "no-referrer")

	tok := r.URL.Query().Get("token")
	u, ok := h.userForResetToken(r.Context(), tok)
	if !ok {
		h.renderGuest(w, r, "reset_password", map[string]any{"Title": "Atur Ulang Password", "Invalid": true})
		return
	}
	h.renderGuest(w, r, "reset_password", map[string]any{
		"Title":    "Atur Ulang Password",
		"Token":    tok,
		"Username": u.Username,
	})
}

// ResetPassword sets the new password. The user is sent to the login page
// rather than signed in, so the new password is exercised once straight away.
func (h *Handler) ResetPassword(w http.ResponseWriter, r *http.Request) {
	if h.mailer == nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Referrer-Policy", "no-referrer")

	tok := r.FormValue("token")
	password := r.FormValue("password")

	u, ok := h.userForResetToken(r.Context(), tok)
	if !ok {
		h.renderGuest(w, r, "reset_password", map[string]any{"Title": "Atur Ulang Password", "Invalid": true})
		return
	}

	form := map[string]any{"Title": "Atur Ulang Password", "Token": tok, "Username": u.Username}
	fail := func(msg string) {
		form["Error"] = msg
		h.renderGuest(w, r, "reset_password", form)
	}

	if password != r.FormValue("password_confirm") {
		fail("Konfirmasi password tidak sama.")
		return
	}
	if err := util.ValidatePassword(password, &h.cfg.PasswordPolicy); err != nil {
		fail(err.Error())
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		log.Printf("password reset hash: %v", err)
		http.Error(w, "Gagal menyimpan password baru", http.StatusInternalServerError)
		return
	}

	now := int32(time.Now().Unix())
	n, err := h.store.ResetUserPassword(r.Context(), db.ResetUserPasswordParams{
		NewHash:   string(hash),
		UpdatedAt: store.NullInt32(now),
		UpdatedBy: store.NullInt32(u.ID),
		ID:        u.ID,
		OldHash:   u.PasswordHash,
	})
	if err != nil {
		log.Printf("password reset: update user id %d: %v", u.ID, err)
		fail("Gagal menyimpan password baru. Silakan coba lagi.")
		return
	}
	if n == 0 {
		// The password changed between the token check and the update: the
		// same link was submitted twice, and the other submission won.
		h.renderGuest(w, r, "reset_password", map[string]any{"Title": "Atur Ulang Password", "Invalid": true})
		return
	}

	log.Printf("password reset: user %q (id %d) set a new password", u.Username, u.ID)
	http.Redirect(w, r, "/login?reset=1", http.StatusSeeOther)
}

// userForResetToken resolves a token to the live, unsuspended account it was
// issued for.
func (h *Handler) userForResetToken(ctx context.Context, tok string) (*db.User, bool) {
	id, _, _, ok := parseResetToken(tok)
	if !ok {
		return nil, false
	}
	u, err := h.store.GetUserByID(ctx, id)
	if err != nil {
		if !errors.Is(err, sql.ErrNoRows) {
			log.Printf("password reset: lookup user id %d: %v", id, err)
		}
		return nil, false
	}
	if u.SuspendedAt.Valid || !verifyResetToken(h.cfg.JWT.SecretKey, tok, &u, time.Now()) {
		return nil, false
	}
	return &u, true
}

// resetToken builds a stateless password-reset token: <id>.<expiry>.<mac>.
//
// Nothing is stored. The MAC binds the token to the account's current password
// hash and email, so setting a new password (through this link or any other
// way) kills every link issued before it, and so does changing the email.
func resetToken(secret string, u *db.User, exp time.Time) string {
	e := exp.Unix()
	mac := resetMAC(secret, u.ID, e, u.PasswordHash, u.Email)
	return fmt.Sprintf("%d.%d.%s", u.ID, e, base64.RawURLEncoding.EncodeToString(mac))
}

// verifyResetToken reports whether tok is an unexpired token for u as u is now.
func verifyResetToken(secret, tok string, u *db.User, now time.Time) bool {
	id, exp, mac, ok := parseResetToken(tok)
	if !ok || id != u.ID || now.Unix() > exp {
		return false
	}
	return hmac.Equal(mac, resetMAC(secret, u.ID, exp, u.PasswordHash, u.Email))
}

// resetMAC is keyed by the JWT secret; the label keeps it from ever matching
// anything else signed under that key.
func resetMAC(secret string, id int32, exp int64, passwordHash, email string) []byte {
	m := hmac.New(sha256.New, []byte(secret))
	fmt.Fprintf(m, "drml-pwreset|%d|%d|%s|%s", id, exp, passwordHash, email)
	return m.Sum(nil)
}

// parseResetToken splits a token without judging it. A token that parses is
// still worthless until verifyResetToken has checked it against the account.
func parseResetToken(tok string) (id int32, exp int64, mac []byte, ok bool) {
	parts := strings.Split(tok, ".")
	if len(parts) != 3 {
		return 0, 0, nil, false
	}
	id64, err := strconv.ParseInt(parts[0], 10, 32)
	if err != nil || id64 <= 0 {
		return 0, 0, nil, false
	}
	exp, err = strconv.ParseInt(parts[1], 10, 64)
	if err != nil {
		return 0, 0, nil, false
	}
	mac, err = base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return 0, 0, nil, false
	}
	return int32(id64), exp, mac, true
}
