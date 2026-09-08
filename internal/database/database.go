package database

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/go-sql-driver/mysql"

	"github.com/remorac/drml/internal/shared/config"
)

// Open connects to MySQL and configures a pool sized for a small VPS.
func Open(cfg *config.DBConfig) (*sql.DB, error) {
	db, err := sql.Open("mysql", cfg.DSN())
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	// The target box runs MySQL with max_connections=50 alongside the app and
	// an ONNX session; a large Go-side pool would just reserve memory that the
	// inference arena needs. Inference is serialized anyway, so request
	// concurrency here is low by construction.
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(30 * time.Minute)
	db.SetConnMaxIdleTime(5 * time.Minute)

	return db, nil
}
