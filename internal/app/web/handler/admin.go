package handler

import (
	"database/sql"
	"errors"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"golang.org/x/crypto/bcrypt"

	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/shared/model"
	"github.com/remorac/drml/internal/shared/pagination"
	"github.com/remorac/drml/internal/shared/util"
)

// UserList shows all accounts.
func (h *Handler) UserList(w http.ResponseWriter, r *http.Request) {
	page := atoiDefault(r.URL.Query().Get("page"), 1)
	if page < 1 {
		page = 1
	}
	size := h.cfg.PageSize

	var roleFilter sql.NullInt32
	if v := r.URL.Query().Get("role"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			roleFilter = sql.NullInt32{Int32: int32(n), Valid: true}
		}
	}

	total, err := h.store.CountUsers(r.Context(), db.CountUsersParams{Role: roleFilter})
	if err != nil {
		log.Printf("user list count: %v", err)
		http.Error(w, "Gagal memuat data pengguna", http.StatusInternalServerError)
		return
	}
	rows, err := h.store.ListUsers(r.Context(), db.ListUsersParams{
		Role:   roleFilter,
		Limit:  int32(size),
		Offset: int32((page - 1) * size),
	})
	if err != nil {
		log.Printf("user list: %v", err)
		http.Error(w, "Gagal memuat data pengguna", http.StatusInternalServerError)
		return
	}

	data := map[string]any{
		"Title":      "Manajemen Pengguna",
		"Users":      rows,
		"Pagination": pagination.New(page, total, size, "/admin/users?", true),
		"RoleFilter": r.URL.Query().Get("role"),
	}
	switch r.URL.Query().Get("error") {
	case "self":
		data["Error"] = "Anda tidak dapat menangguhkan akun Anda sendiri."
	case "last_admin":
		data["Error"] = "Tidak dapat menangguhkan akun administrator terakhir dalam sistem."
	}

	h.render(w, r, "users", data)
}

// NewUserPage renders the create-account form.
func (h *Handler) NewUserPage(w http.ResponseWriter, r *http.Request) {
	h.render(w, r, "user_form", map[string]any{
		"Title": "Tambah Pengguna",
		"Role":  int(model.UserRoleClinician),
	})
}

// CreateUser adds an account.
func (h *Handler) CreateUser(w http.ResponseWriter, r *http.Request) {
	username := strings.TrimSpace(r.FormValue("username"))
	password := r.FormValue("password")
	name := strings.TrimSpace(r.FormValue("name"))
	email := strings.TrimSpace(r.FormValue("email"))
	role, _ := strconv.Atoi(r.FormValue("role"))

	// Role is echoed back so a failed submission re-renders with the operator's
	// choice still selected rather than silently resetting to Klinisi.
	form := map[string]any{
		"Title":    "Tambah Pengguna",
		"Username": username, "Name": name, "Email": email, "Role": role,
	}

	if len(username) < 3 {
		form["Error"] = "Username minimal 3 karakter."
		h.render(w, r, "user_form", form)
		return
	}
	if err := util.ValidatePassword(password, &h.cfg.PasswordPolicy); err != nil {
		form["Error"] = err.Error()
		h.render(w, r, "user_form", form)
		return
	}
	if role != int(model.UserRoleClinician) && role != int(model.UserRoleAdmin) {
		form["Error"] = "Pilihan peran (role) tidak valid."
		h.render(w, r, "user_form", form)
		return
	}

	if _, err := h.store.GetUserByUsername(r.Context(), username); err == nil {
		form["Error"] = "Username sudah digunakan oleh akun lain."
		h.render(w, r, "user_form", form)
		return
	} else if !errors.Is(err, sql.ErrNoRows) {
		log.Printf("create user lookup: %v", err)
		form["Error"] = "Gagal memeriksa ketersediaan username."
		h.render(w, r, "user_form", form)
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		log.Printf("create user hash: %v", err)
		http.Error(w, "Gagal membuat akun pengguna", http.StatusInternalServerError)
		return
	}

	me := actor(r)
	now := int32(time.Now().Unix())
	if _, err := h.store.CreateUser(r.Context(), db.CreateUserParams{
		Name:               store.NullString(name),
		Email:              store.NullString(email),
		Username:           username,
		PasswordHash:       string(hash),
		Role:               int32(role),
		Subrole:            int32(model.UserSubroleStaff),
		MustChangePassword: store.NullInt32(1),
		CreatedAt:          store.NullInt32(now),
		UpdatedAt:          store.NullInt32(now),
		CreatedBy:          store.NullInt32(me.ID),
		UpdatedBy:          store.NullInt32(me.ID),
	}); err != nil {
		log.Printf("create user: %v", err)
		form["Error"] = "Gagal membuat akun pengguna."
		h.render(w, r, "user_form", form)
		return
	}

	http.Redirect(w, r, "/admin/users", http.StatusFound)
}

// SuspendUser toggles an account's suspension.
func (h *Handler) SuspendUser(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(chi.URLParam(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	me := actor(r)
	if int32(id) == me.ID {
		// Suspending yourself would lock the last admin out of the system.
		http.Redirect(w, r, "/admin/users?error=self", http.StatusSeeOther)
		return
	}

	target, err := h.store.GetUserByID(r.Context(), int32(id))
	if err != nil {
		http.NotFound(w, r)
		return
	}

	// Refuse to suspend the last active administrator.
	if model.UserRole(target.Role) == model.UserRoleAdmin && !target.SuspendedAt.Valid {
		admins, err := h.store.CountUsersByRole(r.Context(), int32(model.UserRoleAdmin))
		if err != nil {
			log.Printf("suspend: count admins: %v", err)
			http.Error(w, "Gagal memverifikasi akun administrator", http.StatusInternalServerError)
			return
		}
		if admins <= 1 {
			http.Redirect(w, r, "/admin/users?error=last_admin", http.StatusSeeOther)
			return
		}
	}

	now := int32(time.Now().Unix())
	suspended := sql.NullInt32{}
	if !target.SuspendedAt.Valid {
		suspended = store.NullInt32(now)
	}

	if err := h.store.SetUserSuspended(r.Context(), db.SetUserSuspendedParams{
		SuspendedAt: suspended,
		UpdatedAt:   store.NullInt32(now),
		UpdatedBy:   store.NullInt32(me.ID),
		ID:          int32(id),
	}); err != nil {
		log.Printf("suspend user %d: %v", id, err)
		http.Error(w, "Gagal memperbarui data pengguna", http.StatusInternalServerError)
		return
	}

	http.Redirect(w, r, "/admin/users", http.StatusSeeOther)
}
