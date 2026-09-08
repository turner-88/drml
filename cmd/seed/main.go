// Command seed creates the initial administrator account.
//
// Usage:
//
//	go run ./cmd/seed -username admin -password 'secret'
//
// Refuses to run if any administrator already exists, so it cannot be used to
// silently mint a second superuser on a live system.
package main

import (
	"context"
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"golang.org/x/crypto/bcrypt"

	"github.com/remorac/drml/internal/database"
	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/shared/config"
	"github.com/remorac/drml/internal/shared/model"
)

func main() {
	username := flag.String("username", "admin", "administrator username")
	password := flag.String("password", "", "administrator password (required)")
	name := flag.String("name", "Administrator", "display name")
	email := flag.String("email", "", "email address")
	force := flag.Bool("force", false, "create even if an administrator already exists")
	flag.Parse()

	if *password == "" {
		fmt.Fprintln(os.Stderr, "error: -password is required")
		flag.Usage()
		os.Exit(2)
	}
	if len(*password) < 8 {
		fmt.Fprintln(os.Stderr, "error: password must be at least 8 characters")
		os.Exit(2)
	}

	cfg := config.Load()
	conn, err := database.Open(&cfg.DB)
	if err != nil {
		log.Fatalf("open database: %v", err)
	}
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if err := conn.PingContext(ctx); err != nil {
		log.Fatalf("connect to database: %v", err)
	}

	st := store.New(conn)

	admins, err := st.CountUsersByRole(ctx, int32(model.UserRoleAdmin))
	if err != nil {
		log.Fatalf("count administrators: %v", err)
	}
	if admins > 0 && !*force {
		log.Fatalf("refusing to seed: %d administrator(s) already exist (use -force to override)", admins)
	}

	if _, err := st.GetUserByUsername(ctx, *username); err == nil {
		log.Fatalf("user %q already exists", *username)
	} else if !errors.Is(err, sql.ErrNoRows) {
		log.Fatalf("check username: %v", err)
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(*password), bcrypt.DefaultCost)
	if err != nil {
		log.Fatalf("hash password: %v", err)
	}

	now := int32(time.Now().Unix())
	res, err := st.CreateUser(ctx, db.CreateUserParams{
		Name:         store.NullString(*name),
		Email:        store.NullString(*email),
		Username:     *username,
		PasswordHash: string(hash),
		Role:         int32(model.UserRoleAdmin),
		Subrole:      int32(model.UserSubroleSuperuser),
		// Seeded credentials are typically passed on a command line and end up
		// in shell history, so require a change at first login.
		MustChangePassword: store.NullInt32(1),
		CreatedAt:          store.NullInt32(now),
		UpdatedAt:          store.NullInt32(now),
	})
	if err != nil {
		log.Fatalf("create user: %v", err)
	}

	id, _ := res.LastInsertId()
	fmt.Printf("Created administrator %q (id %d).\n", *username, id)
	fmt.Println("Change this password after the first login.")
}
