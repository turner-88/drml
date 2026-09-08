// Package store wraps the generated sqlc queries with convenience helpers.
package store

import (
	"context"
	"database/sql"
	"fmt"

	db "github.com/remorac/drml/internal/database/sqlc"
)

// Store provides data access on top of the generated queries.
type Store struct {
	*db.Queries
	conn *sql.DB
}

func New(conn *sql.DB) *Store {
	return &Store{Queries: db.New(conn), conn: conn}
}

// Conn exposes the underlying pool for health checks.
func (s *Store) Conn() *sql.DB { return s.conn }

// CreateScan inserts a scan and returns the new row's ID.
func (s *Store) CreateScan(ctx context.Context, arg db.CreateScanParams) (int32, error) {
	res, err := s.Queries.CreateScan(ctx, arg)
	if err != nil {
		return 0, fmt.Errorf("insert scan: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("scan id: %w", err)
	}
	return int32(id), nil
}

// --- sql.Null* helpers -------------------------------------------------------

func NullInt32(v int32) sql.NullInt32 { return sql.NullInt32{Int32: v, Valid: true} }

func NullString(s string) sql.NullString {
	return sql.NullString{String: s, Valid: s != ""}
}

func NullInt16(v int16) sql.NullInt16 { return sql.NullInt16{Int16: v, Valid: true} }

func NullFloat64(v float64) sql.NullFloat64 {
	return sql.NullFloat64{Float64: v, Valid: true}
}
