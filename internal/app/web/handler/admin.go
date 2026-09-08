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
		http.Error(w, "Gagal memuat pengguna", http.StatusInternalServerError)
		return
	}
	rows, err := h.store.ListUsers(r.Context(), db.ListUsersParams{
		Role:   roleFilter,
		Limit:  int32(size),
		Offset: int32((page - 1) * size),
	})
	if err != nil {
		log.Printf("user list: %v", err)
		http.Error(w, "Gagal memuat pengguna", http.StatusInternalServerError)
		return
	}

	h.render(w, r, "users", map[string]any{
		"Title":      "Manajemen Pengguna",
		"Users":      rows,
		"Pagination": pagination.New(page, total, size, "/admin/users?", true),
		"RoleFilter": r.URL.Query().Get("role"),
	})
}

// NewUserPage renders the create-account form.
func (h *Handler) NewUserPage(w http.ResponseWriter, r *http.Request) {
	h.render(w, r, "user_form", map[string]any{
		"Title": "Tambah Pengguna",
		"IsNew": true,
	})
}

// CreateUser adds an account.
func (h *Handler) CreateUser(w http.ResponseWriter, r *http.Request) {
	username := strings.TrimSpace(r.FormValue("username"))
	password := r.FormValue("password")
	name := strings.TrimSpace(r.FormValue("name"))
	email := strings.TrimSpace(r.FormValue("email"))
	role, _ := strconv.Atoi(r.FormValue("role"))

	form := map[string]any{
		"Title": "Tambah Pengguna", "IsNew": true,
		"Username": username, "Name": name, "Email": email, "Role": role,
	}

	if len(username) < 3 {
		form["Error"] = "Nama pengguna minimal 3 karakter."
		h.render(w, r, "user_form", form)
		return
	}
	if err := util.ValidatePassword(password, &h.cfg.PasswordPolicy); err != nil {
		form["Error"] = err.Error()
		h.render(w, r, "user_form", form)
		return
	}
	if role != int(model.UserRoleClinician) && role != int(model.UserRoleAdmin) {
		form["Error"] = "Peran tidak valid."
		h.render(w, r, "user_form", form)
		return
	}

	if _, err := h.store.GetUserByUsername(r.Context(), username); err == nil {
		form["Error"] = "Nama pengguna sudah digunakan."
		h.render(w, r, "user_form", form)
		return
	} else if !errors.Is(err, sql.ErrNoRows) {
		log.Printf("create user lookup: %v", err)
		form["Error"] = "Gagal memeriksa nama pengguna."
		h.render(w, r, "user_form", form)
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		log.Printf("create user hash: %v", err)
		http.Error(w, "Gagal membuat pengguna", http.StatusInternalServerError)
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
		form["Error"] = "Gagal membuat pengguna."
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
		http.Error(w, "Tidak dapat menangguhkan akun sendiri", http.StatusBadRequest)
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
			http.Error(w, "Gagal memeriksa administrator", http.StatusInternalServerError)
			return
		}
		if admins <= 1 {
			http.Error(w, "Tidak dapat menangguhkan administrator terakhir", http.StatusBadRequest)
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
		http.Error(w, "Gagal memperbarui pengguna", http.StatusInternalServerError)
		return
	}

	if r.Header.Get("HX-Request") != "" {
		w.Header().Set("HX-Redirect", "/admin/users")
		w.WriteHeader(http.StatusOK)
		return
	}
	http.Redirect(w, r, "/admin/users", http.StatusSeeOther)
}
