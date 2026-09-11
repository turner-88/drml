package handler

import (
	"database/sql"
	"errors"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

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
	case "self_delete":
		data["Error"] = "Anda tidak dapat menghapus akun Anda sendiri."
	case "has_scans":
		data["Error"] = "Akun yang sudah memiliki data pemeriksaan tidak dapat dihapus. Tangguhkan akun tersebut sebagai gantinya."
	}
	switch r.URL.Query().Get("notice") {
	case "updated":
		data["Notice"] = "Data pengguna berhasil diperbarui."
	case "password":
		data["Notice"] = "Password pengguna berhasil diatur ulang."
	case "deleted":
		data["Notice"] = "Akun pengguna berhasil dihapus."
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
	// choice still selected rather than silently resetting to User.
	form := map[string]any{
		"Title":    "Tambah Pengguna",
		"Username": username, "Name": name, "Email": email, "Role": role,
	}

	if len(username) < 3 {
		form["Error"] = "Username minimal 3 karakter."
		h.render(w, r, "user_form", form)
		return
	}
	if msg := validateEmail(email); msg != "" {
		form["Error"] = msg
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

	if msg := h.identityTaken(r.Context(), username, email, 0); msg != "" {
		form["Error"] = msg
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
		Email:              email,
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
		if msg := duplicateMessage(err); msg != "" {
			form["Error"] = msg
		}
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

// selfRegistered reports whether the account came through /register. Only
// registration leaves created_by NULL on a staff account; cmd/seed leaves it
// NULL too, but seeds a superuser.
func selfRegistered(u *db.User) bool {
	return !u.CreatedBy.Valid && u.Subrole != int32(model.UserSubroleSuperuser)
}

// canEditUser reports whether an administrator may change the account's data.
// While public registration is open, accounts people created for themselves
// are theirs to keep as entered; once it is closed, every account is managed
// by the administrators.
func canEditUser(u *db.User, registrationOpen bool) bool {
	return !registrationOpen || !selfRegistered(u)
}

// userFromURL loads the account named by the {id} route parameter, writing the
// error response itself when there is none.
func (h *Handler) userFromURL(w http.ResponseWriter, r *http.Request) (db.User, bool) {
	id, err := strconv.Atoi(chi.URLParam(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return db.User{}, false
	}
	u, err := h.store.GetUserByID(r.Context(), int32(id))
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			http.NotFound(w, r)
		} else {
			log.Printf("user lookup id %d: %v", id, err)
			http.Error(w, "Gagal memuat data pengguna", http.StatusInternalServerError)
		}
		return db.User{}, false
	}
	return u, true
}

// userEditData builds the manage-account view model from the stored row.
// Callers overwrite the field values to echo back a rejected submission.
func (h *Handler) userEditData(r *http.Request, u *db.User) map[string]any {
	editable := canEditUser(u, h.registrationOpen(r.Context()))
	self := u.ID == actor(r).ID
	return map[string]any{
		"Title":    "Kelola Pengguna",
		"UserID":   u.ID,
		"Account":  u.Username,
		"Editable": editable,
		"IsSelf":   self,
		// Locked roles are shown but not offered; UpdateUserData enforces the
		// same rule for a hand-crafted request.
		"RoleLocked": self || !editable,
		"Username":   u.Username,
		"Name":       u.Name.String,
		"Email":      u.Email,
		"Role":       int(u.Role),
	}
}

// EditUserPage renders the manage-account page: the account's data, and a
// password reset that is available whatever the data's edit state.
func (h *Handler) EditUserPage(w http.ResponseWriter, r *http.Request) {
	u, ok := h.userFromURL(w, r)
	if !ok {
		return
	}
	h.render(w, r, "user_edit", h.userEditData(r, &u))
}

// UpdateUserData saves an administrator's changes to an account's name, email,
// username and role.
func (h *Handler) UpdateUserData(w http.ResponseWriter, r *http.Request) {
	target, ok := h.userFromURL(w, r)
	if !ok {
		return
	}
	data := h.userEditData(r, &target)
	fail := func(msg string) {
		data["Error"] = msg
		h.render(w, r, "user_edit", data)
	}

	// Re-checked here rather than trusted from the page: registration may have
	// been opened since the form was loaded.
	if !data["Editable"].(bool) {
		fail("Akun ini dibuat melalui pendaftaran mandiri. Datanya hanya dapat diubah saat pendaftaran publik dinonaktifkan.")
		return
	}

	username := strings.TrimSpace(r.FormValue("username"))
	name := strings.TrimSpace(r.FormValue("name"))
	email := strings.TrimSpace(r.FormValue("email"))
	role, _ := strconv.Atoi(r.FormValue("role"))
	data["Username"], data["Name"], data["Email"], data["Role"] = username, name, email, role

	if utf8.RuneCountInString(name) > 100 {
		fail("Nama lengkap maksimal 100 karakter.")
		return
	}
	if msg := validateEmail(email); msg != "" {
		fail(msg)
		return
	}
	// Only a changed username is held to the current rule, so an account named
	// before the rule existed can still have its other fields saved.
	if username != target.Username {
		if msg := validateUsername(username); msg != "" {
			fail(msg)
			return
		}
	}
	if role != int(model.UserRoleClinician) && role != int(model.UserRoleAdmin) {
		fail("Pilihan peran (role) tidak valid.")
		return
	}

	me := actor(r)
	if int32(role) != target.Role {
		if target.ID == me.ID {
			// Demoting yourself would drop you out of this page mid-session.
			fail("Anda tidak dapat mengubah peran akun Anda sendiri.")
			return
		}
		if model.UserRole(target.Role) == model.UserRoleAdmin {
			admins, err := h.store.CountUsersByRole(r.Context(), int32(model.UserRoleAdmin))
			if err != nil {
				log.Printf("update user: count admins: %v", err)
				fail("Gagal memverifikasi akun administrator.")
				return
			}
			if admins <= 1 {
				fail("Tidak dapat mengubah peran administrator terakhir dalam sistem.")
				return
			}
		}
	}

	if msg := h.identityTaken(r.Context(), username, email, target.ID); msg != "" {
		fail(msg)
		return
	}

	if err := h.store.UpdateUser(r.Context(), db.UpdateUserParams{
		Name:      store.NullString(name),
		Email:     email,
		Username:  username,
		Role:      int32(role),
		UpdatedAt: store.NullInt32(int32(time.Now().Unix())),
		UpdatedBy: store.NullInt32(me.ID),
		ID:        target.ID,
	}); err != nil {
		log.Printf("update user id %d: %v", target.ID, err)
		if msg := duplicateMessage(err); msg != "" {
			fail(msg)
			return
		}
		fail("Gagal menyimpan data pengguna.")
		return
	}

	// The session carries the username for display; reissue it so an admin who
	// renamed their own account does not keep seeing the old name.
	if target.ID == me.ID && username != me.Username {
		if err := h.startSession(w, &model.User{
			ID: me.ID, Username: username, Role: me.Role, Subrole: me.Subrole,
		}); err != nil {
			log.Printf("update user: reissue session for id %d: %v", me.ID, err)
		}
	}

	log.Printf("admin: user %q (id %d) updated by %s (id %d)", username, target.ID, me.Username, me.ID)
	http.Redirect(w, r, "/admin/users?notice=updated", http.StatusSeeOther)
}

// SetUserPassword replaces an account's password with one the administrator
// chose, for users locked out while outgoing mail is not configured. It also
// voids any reset link already emailed, since those are bound to the old hash.
func (h *Handler) SetUserPassword(w http.ResponseWriter, r *http.Request) {
	target, ok := h.userFromURL(w, r)
	if !ok {
		return
	}
	data := h.userEditData(r, &target)
	fail := func(msg string) {
		data["PasswordError"] = msg
		h.render(w, r, "user_edit", data)
	}

	password := r.FormValue("password")
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
		log.Printf("set password hash: %v", err)
		http.Error(w, "Gagal menyimpan password baru", http.StatusInternalServerError)
		return
	}

	me := actor(r)
	if err := h.store.AdminSetUserPassword(r.Context(), db.AdminSetUserPasswordParams{
		PasswordHash: string(hash),
		UpdatedAt:    store.NullInt32(int32(time.Now().Unix())),
		UpdatedBy:    store.NullInt32(me.ID),
		ID:           target.ID,
	}); err != nil {
		log.Printf("set password for user id %d: %v", target.ID, err)
		fail("Gagal menyimpan password baru. Silakan coba lagi.")
		return
	}

	log.Printf("admin: password of user %q (id %d) reset by %s (id %d)", target.Username, target.ID, me.Username, me.ID)
	http.Redirect(w, r, "/admin/users?notice=password", http.StatusSeeOther)
}

// DeleteUser removes an account that has never recorded a scan. Accounts with
// scans are suspended instead, so the clinical record keeps its author.
func (h *Handler) DeleteUser(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(chi.URLParam(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	me := actor(r)
	if int32(id) == me.ID {
		http.Redirect(w, r, "/admin/users?error=self_delete", http.StatusSeeOther)
		return
	}

	// The scan check is part of the DELETE itself, so a scan recorded after the
	// list was rendered still blocks it.
	n, err := h.store.DeleteUser(r.Context(), int32(id))
	if err != nil {
		log.Printf("delete user id %d: %v", id, err)
		http.Error(w, "Gagal menghapus akun pengguna", http.StatusInternalServerError)
		return
	}
	if n == 0 {
		if _, ok := h.userFromURL(w, r); ok {
			http.Redirect(w, r, "/admin/users?error=has_scans", http.StatusSeeOther)
		}
		return
	}

	log.Printf("admin: user id %d deleted by %s (id %d)", id, me.Username, me.ID)
	http.Redirect(w, r, "/admin/users?notice=deleted", http.StatusSeeOther)
}
