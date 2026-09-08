package storage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Local stores objects on the filesystem and serves them from a URL prefix.
type Local struct {
	root      string
	urlPrefix string
}

// NewLocal creates the storage root if needed.
func NewLocal(root, urlPrefix string) (*Local, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("storage: resolve root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, fmt.Errorf("storage: create root %q: %w", abs, err)
	}
	return &Local{root: abs, urlPrefix: strings.TrimRight(urlPrefix, "/")}, nil
}

// resolve maps a key to an absolute path, refusing anything outside the root.
func (l *Local) resolve(key string) (string, error) {
	if err := validateKey(key); err != nil {
		return "", err
	}
	full := filepath.Join(l.root, filepath.FromSlash(key))
	// Defence in depth: even with a validated key, confirm the result is inside
	// the root before touching the filesystem.
	if !strings.HasPrefix(full, l.root+string(os.PathSeparator)) {
		return "", fmt.Errorf("storage: key escapes root: %q", key)
	}
	return full, nil
}

func (l *Local) Put(ctx context.Context, key string, r io.Reader, contentType string) error {
	full, err := l.resolve(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return fmt.Errorf("storage: create dir: %w", err)
	}

	// Write to a temp file and rename, so a crash mid-write cannot leave a
	// truncated image that the UI would render as a corrupt scan.
	tmp, err := os.CreateTemp(filepath.Dir(full), ".tmp-*")
	if err != nil {
		return fmt.Errorf("storage: create temp: %w", err)
	}
	tmpName := tmp.Name()
	defer func() {
		tmp.Close()
		os.Remove(tmpName) // no-op once renamed
	}()

	if _, err := io.Copy(tmp, r); err != nil {
		return fmt.Errorf("storage: write: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("storage: sync: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("storage: close: %w", err)
	}
	if err := os.Chmod(tmpName, 0o644); err != nil {
		return fmt.Errorf("storage: chmod: %w", err)
	}
	if err := os.Rename(tmpName, full); err != nil {
		return fmt.Errorf("storage: rename: %w", err)
	}
	return nil
}

func (l *Local) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	full, err := l.resolve(key)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(full)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("storage: open: %w", err)
	}
	return f, nil
}

func (l *Local) Delete(ctx context.Context, key string) error {
	full, err := l.resolve(key)
	if err != nil {
		return err
	}
	if err := os.Remove(full); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("storage: delete: %w", err)
	}
	return nil
}

func (l *Local) URL(ctx context.Context, key string) (string, error) {
	if err := validateKey(key); err != nil {
		return "", err
	}
	return l.urlPrefix + "/" + key, nil
}

// Root exposes the storage root so the HTTP layer can mount a file server.
func (l *Local) Root() string { return l.root }
