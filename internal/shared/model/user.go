package model

// UserRole is the access level stored in user.role.
type UserRole int32

const (
	// UserRoleClinician may upload scans and see only their own history.
	UserRoleClinician UserRole = 1
	// UserRoleAdmin sees every scan and manages users.
	UserRoleAdmin UserRole = 2
)

// UserSubrole is stored in user.subrole.
type UserSubrole int32

const (
	UserSubroleStaff     UserSubrole = 1
	UserSubroleSuperuser UserSubrole = 9
)

// RoleLabel returns the Indonesian display name for a role.
func RoleLabel(r UserRole) string {
	switch r {
	case UserRoleAdmin:
		return "Administrator"
	case UserRoleClinician:
		return "User"
	default:
		return "Tidak diketahui"
	}
}

// User is placed in the request context by the auth middleware after the JWT
// is validated. It carries only what authorization needs — never secrets.
type User struct {
	ID       int32    `json:"id"`
	Role     UserRole `json:"role"`
	Subrole  *int32   `json:"subrole,omitempty"`
	Username string   `json:"username,omitempty"`
}

func (u *User) IsAdmin() bool     { return u.Role == UserRoleAdmin }
func (u *User) IsClinician() bool { return u.Role == UserRoleClinician }

func (u *User) IsSuperuser() bool {
	return u.Subrole != nil && *u.Subrole == int32(UserSubroleSuperuser)
}

// ScanScope reports how scan queries should be filtered for this user.
//
// Every scan query takes (allScans, createdBy) rather than filtering in Go, so
// a clinician physically cannot receive another clinician's rows.
func (u *User) ScanScope() (allScans bool, createdBy int32) {
	if u == nil {
		return false, 0
	}
	return u.IsAdmin(), u.ID
}

// --- Request types ---

type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type CreateUserRequest struct {
	Name     string   `json:"name"`
	Email    string   `json:"email"`
	Username string   `json:"username"`
	Password string   `json:"password"`
	Role     UserRole `json:"role"`
	Subrole  int32    `json:"subrole"`
}
