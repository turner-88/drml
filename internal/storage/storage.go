// Package storage abstracts blob storage for scan images and heatmaps.
//
// Scan rows persist an opaque storage *key*, never a URL. Keeping URL
// construction behind this interface is what makes moving from local disk to
// Cloudflare R2 a config change rather than a data migration.
package storage

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"path"
	"strings"
	"time"
)

// ErrNotFound is returned when a key has no object.
var ErrNotFound = errors.New("storage: object not found")

// Storage is a minimal blob store.
type Storage interface {
	// Put writes r under key, replacing any existing object.
	Put(ctx context.Context, key string, r io.Reader, contentType string) error
	// Get opens the object at key. The caller closes the reader.
	Get(ctx context.Context, key string) (io.ReadCloser, error)
	// Delete removes the object. Deleting a missing key is not an error.
	Delete(ctx context.Context, key string) error
	// URL returns a location the browser can load. For private backends this
	// is a time-limited presigned URL, so it must be generated per request
	// rather than cached in the database.
	URL(ctx context.Context, key string) (string, error)
}

// NewKey builds a collision-resistant, date-partitioned key.
//
// Date partitioning keeps any single directory (or R2 listing page) small as
// the archive grows, and the random component avoids leaking scan volume or
// letting one patient's key be guessed from another's.
func NewKey(prefix, ext string) (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("storage: generate key: %w", err)
	}
	now := time.Now().UTC()
	name := hex.EncodeToString(buf) + normalizeExt(ext)
	return path.Join(prefix, now.Format("2006"), now.Format("01"), name), nil
}

func normalizeExt(ext string) string {
	if ext == "" {
		return ""
	}
	if !strings.HasPrefix(ext, ".") {
		ext = "." + ext
	}
	return strings.ToLower(ext)
}

// validateKey rejects keys that could escape the storage root.
//
// Keys reaching Get/Delete can originate from database rows, so this guards
// against a corrupted or tampered row turning into an arbitrary file read.
func validateKey(key string) error {
	if key == "" {
		return errors.New("storage: empty key")
	}
	if path.IsAbs(key) || strings.HasPrefix(key, "/") {
		return fmt.Errorf("storage: key must be relative: %q", key)
	}
	if strings.Contains(key, "\x00") {
		return fmt.Errorf("storage: key contains NUL")
	}
	cleaned := path.Clean(key)
	if cleaned != key {
		return fmt.Errorf("storage: key is not in canonical form: %q", key)
	}
	for _, seg := range strings.Split(cleaned, "/") {
		if seg == ".." || seg == "." || seg == "" {
			return fmt.Errorf("storage: key has an invalid path segment: %q", key)
		}
	}
	return nil
}
