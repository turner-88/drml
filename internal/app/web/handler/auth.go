package handler

import (
	"database/sql"
	"errors"
	"log"
	"net/http"

	"golang.org/x/crypto/bcrypt"

	"github.com/remorac/drml/internal/shared/model"
	"github.com/remorac/drml/internal/shared/util"
)

const authCookie = "drml_token"

// LoginPage renders the sign-in form.
func (h *Handler) LoginPage(w http.ResponseWriter, r *http.Request) {
	data := map[string]any{"Title": "Masuk"}
	if r.URL.Query().Get("blocked") == "1" {
		data["Error"] = "Terlalu banyak percobaan masuk. Coba lagi dalam 15 menit."
	}
	h.renderGuest(w, r, "login", data)
}

// Login authenticates and issues the session cookie.
func (h *Handler) Login(w http.ResponseWriter, r *http.Request) {
	username := r.FormValue("username")
	password := r.FormValue("password")

	fail := func() {
		// One message for every failure mode: distinguishing "no such user"
		// from "wrong password" would let an attacker enumerate accounts.
		h.renderGuest(w, r, "login", map[string]any{
			"Title":    "Masuk",
			"Error":    "Nama pengguna atau kata sandi salah.",
			"Username": username,
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
			"Title": "Masuk",
			"Error": "Akun ini ditangguhkan. Hubungi administrator.",
		})
		return
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)); err != nil {
		fail()
		return
	}

	subrole := user.Subrole
	token, err := util.GenerateToken(&model.User{
		ID:       user.ID,
		Username: user.Username,
		Role:     model.UserRole(user.Role),
		Subrole:  &subrole,
	}, &util.JWTConfig{
		SecretKey:       h.cfg.JWT.SecretKey,
		ExpirationHours: h.cfg.JWT.ExpirationHours,
	})
	if err != nil {
		log.Printf("login: token for %q: %v", username, err)
		http.Error(w, "Gagal membuat sesi", http.StatusInternalServerError)
		return
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
