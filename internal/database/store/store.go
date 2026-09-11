// Package store wraps the generated sqlc queries with convenience helpers.
package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strconv"
	"time"

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

// --- settings ----------------------------------------------------------------

// SettingGateEnabled is the runtime on/off switch for the non-fundus gate.
//
// The gate's thresholds stay in model/gate.json, where model/gate.py can
// measure them; only whether it runs at all is an operator decision, and it has
// to be changeable without a redeploy.
const SettingGateEnabled = "gate.enabled"

// SettingRegistrationEnabled opens /register to the public. Read with a false
// default: an open sign-up form on a clinical tool has to be a deliberate
// administrator decision, never a side effect of deploying this feature.
const SettingRegistrationEnabled = "registration.enabled"

// GetBoolSetting reads a stored flag, falling back to def when no row exists.
//
// An absent row means nobody has expressed a preference, which is not the same
// as a stored false: it lets the caller keep its own default (for the gate,
// gate.json's own `enabled`) until an administrator actually decides.
func (s *Store) GetBoolSetting(ctx context.Context, key string, def bool) (bool, error) {
	row, err := s.Queries.GetSetting(ctx, key)
	if errors.Is(err, sql.ErrNoRows) {
		return def, nil
	}
	if err != nil {
		return def, fmt.Errorf("read setting %q: %w", key, err)
	}
	v, err := strconv.ParseBool(row.SettingValue)
	if err != nil {
		// A corrupt value must not silently flip a safety control either way;
		// report it and let the caller keep its default.
		return def, fmt.Errorf("setting %q holds %q, not a boolean: %w", key, row.SettingValue, err)
	}
	return v, nil
}

// SetBoolSetting stores a flag, recording who changed it and when.
func (s *Store) SetBoolSetting(ctx context.Context, key string, v bool, actorID int32) error {
	if err := s.Queries.UpsertSetting(ctx, db.UpsertSettingParams{
		SettingKey:   key,
		SettingValue: strconv.FormatBool(v),
		UpdatedAt:    NullInt32(int32(time.Now().Unix())),
		UpdatedBy:    NullInt32(actorID),
	}); err != nil {
		return fmt.Errorf("write setting %q: %w", key, err)
	}
	return nil
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
