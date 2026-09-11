package handler

import (
	"context"
	"database/sql"
	"errors"
	"log"
	"net/http"
	"net/mail"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/go-sql-driver/mysql"
	"golang.org/x/crypto/bcrypt"

	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/shared/model"
	"github.com/remorac/drml/internal/shared/util"
)

const authCookie = "drml_token"

// LoginPage renders the sign-in form.
func (h *Handler) LoginPage(w http.ResponseWriter, r *http.Request) {
	data := map[string]any{
		"Title":            "Masuk",
		"RegistrationOpen": h.registrationOpen(r.Context()),
		"ForgotEnabled":    h.mailer != nil,
	}
	if r.URL.Query().Get("blocked") == "1" {
		data["Error"] = "Terlalu banyak percobaan login yang gagal. Silakan coba lagi dalam 15 menit."
	}
	if r.URL.Query().Get("reset") == "1" {
		data["Notice"] = "Password berhasil diubah. Silakan masuk menggunakan password baru Anda."
	}
	h.renderGuest(w, r, "login", data)
}

// Login authenticates and issues the session cookie.
func (h *Handler) Login(w http.ResponseWriter, r *http.Request) {
	username := r.FormValue("username")
	password := r.FormValue("password")
	open := h.registrationOpen(r.Context())

	fail := func() {
		// One message for every failure mode: distinguishing "no such user"
		// from "wrong password" would let an attacker enumerate accounts.
		h.renderGuest(w, r, "login", map[string]any{
			"Title":            "Masuk",
			"Error":            "Username atau password salah.",
			"Username":         username,
			"RegistrationOpen": open,
			"ForgotEnabled":    h.mailer != nil,
		})
	}

	user, err := h.store.GetUserByUsername(r.Context(), username)
	if err != nil {
		if !errors.Is(err, sql.ErrNoRows) {
			log.Printf("login: lookup %q: %v", username, err)
		}
		// Spend comparable time hashing so a missing user is not detectably
		// faster than a wrong password.
		_ = bcrypt.CompareHashAndPassword(
			[]byte("$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"),
			[]byte(password),
		)
		fail()
		return
	}

	if user.SuspendedAt.Valid {
		h.renderGuest(w, r, "login", map[string]any{
			"Title":            "Masuk",
			"Error":            "Akun ini sedang ditangguhkan. Silakan hubungi administrator.",
			"RegistrationOpen": open,
			"ForgotEnabled":    h.mailer != nil,
		})
		return
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)); err != nil {
		fail()
		return
	}

	subrole := user.Subrole
	if err := h.startSession(w, &model.User{
		ID:       user.ID,
		Username: user.Username,
		Role:     model.UserRole(user.Role),
		Subrole:  &subrole,
	}); err != nil {
		log.Printf("login: token for %q: %v", username, err)
		http.Error(w, "Gagal membuat sesi login", http.StatusInternalServerError)
		return
	}

	http.Redirect(w, r, "/", http.StatusFound)
}

// RegisterPage renders the public clinician sign-up form.
func (h *Handler) RegisterPage(w http.ResponseWriter, r *http.Request) {
	if !h.registrationOpen(r.Context()) {
		http.NotFound(w, r)
		return
	}
	data := map[string]any{"Title": "Daftar"}
	if r.URL.Query().Get("blocked") == "1" {
		data["Error"] = "Terlalu banyak percobaan pendaftaran. Silakan coba lagi dalam 1 jam."
	}
	h.renderGuest(w, r, "register", data)
}

// Register creates a clinician account for a member of the public and signs
// them in. The role is fixed here, never read from the form, so the endpoint
// cannot be used to mint an administrator.
func (h *Handler) Register(w http.ResponseWriter, r *http.Request) {
	if !h.registrationOpen(r.Context()) {
		http.NotFound(w, r)
		return
	}

	name := strings.TrimSpace(r.FormValue("name"))
	email := strings.TrimSpace(r.FormValue("email"))
	username := strings.TrimSpace(r.FormValue("username"))
	password := r.FormValue("password")

	// Passwords are never echoed back into the re-rendered form.
	form := map[string]any{
		"Title": "Daftar",
		"Name":  name, "Email": email, "Username": username,
	}
	fail := func(msg string) {
		form["Error"] = msg
		h.renderGuest(w, r, "register", form)
	}

	if name == "" {
		fail("Nama lengkap wajib diisi.")
		return
	}
	if utf8.RuneCountInString(name) > 100 {
		fail("Nama lengkap maksimal 100 karakter.")
		return
	}
	if msg := validateEmail(email); msg != "" {
		fail(msg)
		return
	}
	if msg := validateUsername(username); msg != "" {
		fail(msg)
		return
	}
	if password != r.FormValue("password_confirm") {
		fail("Konfirmasi password tidak sama.")
		return
	}
	if err := util.ValidatePassword(password, &h.cfg.PasswordPolicy); err != nil {
		fail(err.Error())
		return
	}

	if msg := h.identityTaken(r.Context(), username, email, 0); msg != "" {
		fail(msg)
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		log.Printf("register hash: %v", err)
		http.Error(w, "Gagal membuat akun", http.StatusInternalServerError)
		return
	}

	now := int32(time.Now().Unix())
	// created_by stays NULL: that is what marks an account as self-registered.
	// must_change_password stays NULL too, since the user chose this password.
	res, err := h.store.CreateUser(r.Context(), db.CreateUserParams{
		Name:         store.NullString(name),
		Email:        email,
		Username:     username,
		PasswordHash: string(hash),
		Role:         int32(model.UserRoleClinician),
		Subrole:      int32(model.UserSubroleStaff),
		CreatedAt:    store.NullInt32(now),
		UpdatedAt:    store.NullInt32(now),
	})
	if err != nil {
		log.Printf("register %q: %v", username, err)
		// The lookup above passed, so a duplicate here is most likely a
		// concurrent sign-up for the same name or address.
		if msg := duplicateMessage(err); msg != "" {
			fail(msg)
			return
		}
		fail("Gagal membuat akun. Silakan coba lagi.")
		return
	}
	id, err := res.LastInsertId()
	if err != nil {
		// The account exists; only the automatic sign-in is lost.
		log.Printf("register %q: new id: %v", username, err)
		http.Redirect(w, r, "/login", http.StatusFound)
		return
	}

	log.Printf("register: self-registered clinician %q (id %d)", username, id)

	subrole := int32(model.UserSubroleStaff)
	if err := h.startSession(w, &model.User{
		ID:       int32(id),
		Username: username,
		Role:     model.UserRoleClinician,
		Subrole:  &subrole,
	}); err != nil {
		log.Printf("register: token for %q: %v", username, err)
		http.Redirect(w, r, "/login", http.StatusFound)
		return
	}

	http.Redirect(w, r, "/", http.StatusFound)
}

// Logout clears the session cookie.
func (h *Handler) Logout(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name:     authCookie,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		Secure:   h.cfg.IsProduction(),
		SameSite: http.SameSiteLaxMode,
		MaxAge:   -1,
	})
	http.Redirect(w, r, "/login", http.StatusFound)
}

// startSession issues the JWT cookie for an authenticated user.
func (h *Handler) startSession(w http.ResponseWriter, u *model.User) error {
	token, err := util.GenerateToken(u, &util.JWTConfig{
		SecretKey:       h.cfg.JWT.SecretKey,
		ExpirationHours: h.cfg.JWT.ExpirationHours,
	})
	if err != nil {
		return err
	}
	http.SetCookie(w, &http.Cookie{
		Name:     authCookie,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		Secure:   h.cfg.IsProduction(),
		SameSite: http.SameSiteLaxMode,
		MaxAge:   h.cfg.JWT.ExpirationHours * 3600,
	})
	return nil
}

// registrationOpen reports whether public sign-up is enabled. A failed read
// keeps the form closed rather than opening it by accident.
func (h *Handler) registrationOpen(ctx context.Context) bool {
	open, err := h.store.GetBoolSetting(ctx, store.SettingRegistrationEnabled, false)
	if err != nil {
		log.Printf("registration setting: %v", err)
		return false
	}
	return open
}

// identityTaken returns a user-facing message when the username or email
// already belongs to an account other than except (0 for a new account). The
// unique keys are the real guarantee; this turns the common case into a clear
// message before bcrypt runs.
func (h *Handler) identityTaken(ctx context.Context, username, email string, except int32) string {
	if u, err := h.store.GetUserByUsername(ctx, username); err == nil {
		if u.ID != except {
			return "Username sudah digunakan oleh akun lain."
		}
	} else if !errors.Is(err, sql.ErrNoRows) {
		log.Printf("identity lookup username: %v", err)
		return "Gagal memeriksa ketersediaan username."
	}
	if u, err := h.store.GetUserByEmail(ctx, email); err == nil {
		if u.ID != except {
			return "Email sudah digunakan oleh akun lain."
		}
	} else if !errors.Is(err, sql.ErrNoRows) {
		log.Printf("identity lookup email: %v", err)
		return "Gagal memeriksa ketersediaan email."
	}
	return ""
}

// duplicateMessage names the field an insert collided on, or returns "" when
// err is not a duplicate-entry error.
func duplicateMessage(err error) string {
	var myErr *mysql.MySQLError
	if !errors.As(err, &myErr) || myErr.Number != 1062 { // ER_DUP_ENTRY
		return ""
	}
	if strings.Contains(myErr.Message, "uq_user_email") {
		return "Email sudah digunakan oleh akun lain."
	}
	return "Username sudah digunakan oleh akun lain."
}

// validateEmail requires a bare address that fits the user.email column.
func validateEmail(e string) string {
	if e == "" {
		return "Email wajib diisi."
	}
	if len(e) > 100 {
		return "Email maksimal 100 karakter."
	}
	// ParseAddress also accepts "Name <addr>"; require the bare address.
	if addr, err := mail.ParseAddress(e); err != nil || addr.Address != e {
		return "Format email tidak valid."
	}
	return ""
}

// validateUsername applies the rule recorded on the user.username column:
// letters required, digits optional, symbols forbidden.
func validateUsername(u string) string {
	if len(u) < 3 || len(u) > 50 {
		return "Username harus 3–50 karakter."
	}
	hasLetter := false
	for _, c := range u {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z':
			hasLetter = true
		case c >= '0' && c <= '9':
		default:
			return "Username hanya boleh berisi huruf dan angka, tanpa spasi atau simbol."
		}
	}
	if !hasLetter {
		return "Username harus mengandung setidaknya satu huruf."
	}
	return ""
}
