package handler

import (
	"database/sql"
	"testing"

	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/shared/model"
)

func TestCanEditUser(t *testing.T) {
	selfReg := db.User{Role: int32(model.UserRoleClinician), Subrole: int32(model.UserSubroleStaff)}
	adminMade := db.User{
		Role: int32(model.UserRoleClinician), Subrole: int32(model.UserSubroleStaff),
		CreatedBy: sql.NullInt32{Int32: 1, Valid: true},
	}
	// cmd/seed leaves created_by NULL too; the superuser subrole is what tells
	// it apart from a self-registered account.
	seeded := db.User{Role: int32(model.UserRoleAdmin), Subrole: int32(model.UserSubroleSuperuser)}

	cases := []struct {
		name    string
		u       db.User
		regOpen bool
		want    bool
	}{
		{"self-registered, registration open", selfReg, true, false},
		{"self-registered, registration closed", selfReg, false, true},
		{"admin-created, registration open", adminMade, true, true},
		{"admin-created, registration closed", adminMade, false, true},
		{"seeded superuser, registration open", seeded, true, true},
	}
	for _, c := range cases {
		if got := canEditUser(&c.u, c.regOpen); got != c.want {
			t.Errorf("%s: canEditUser = %v, want %v", c.name, got, c.want)
		}
	}
}
