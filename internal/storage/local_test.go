package storage

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func newTestLocal(t *testing.T) *Local {
	t.Helper()
	s, err := NewLocal(t.TempDir(), "/static/uploads")
	if err != nil {
		t.Fatalf("NewLocal: %v", err)
	}
	return s
}

func TestLocalRoundTrip(t *testing.T) {
	s := newTestLocal(t)
	ctx := context.Background()
	want := []byte("fundus-bytes")

	if err := s.Put(ctx, "scans/2026/09/abc.jpg", bytes.NewReader(want), "image/jpeg"); err != nil {
		t.Fatalf("Put: %v", err)
	}

	rc, err := s.Get(ctx, "scans/2026/09/abc.jpg")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer rc.Close()
	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}

	url, err := s.URL(ctx, "scans/2026/09/abc.jpg")
	if err != nil {
		t.Fatalf("URL: %v", err)
	}
	if url != "/static/uploads/scans/2026/09/abc.jpg" {
		t.Errorf("URL = %q", url)
	}

	if err := s.Delete(ctx, "scans/2026/09/abc.jpg"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := s.Get(ctx, "scans/2026/09/abc.jpg"); !errors.Is(err, ErrNotFound) {
		t.Errorf("after delete, Get returned %v, want ErrNotFound", err)
	}
}

// TestLocalRejectsTraversal matters because keys come from database rows: a
// corrupted or tampered row must not turn into an arbitrary filesystem read.
func TestLocalRejectsTraversal(t *testing.T) {
	s := newTestLocal(t)
	ctx := context.Background()

	// Plant a file outside the root to prove escape attempts cannot reach it.
	outside := filepath.Join(filepath.Dir(s.Root()), "secret.txt")
	if err := os.WriteFile(outside, []byte("classified"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Remove(outside) })

	bad := []string{
		"../secret.txt",
		"../../etc/passwd",
		"scans/../../secret.txt",
		"/etc/passwd",
		"./scans/x.jpg",
		"scans//x.jpg",
		"",
		"scans/./x.jpg",
	}

	for _, key := range bad {
		t.Run(key, func(t *testing.T) {
			if _, err := s.Get(ctx, key); err == nil {
				t.Errorf("Get(%q) succeeded; expected rejection", key)
			}
			if err := s.Put(ctx, key, strings.NewReader("x"), "text/plain"); err == nil {
				t.Errorf("Put(%q) succeeded; expected rejection", key)
			}
			if err := s.Delete(ctx, key); err == nil {
				t.Errorf("Delete(%q) succeeded; expected rejection", key)
			}
			if _, err := s.URL(ctx, key); err == nil {
				t.Errorf("URL(%q) succeeded; expected rejection", key)
			}
		})
	}

	// The planted file must still be intact.
	if b, err := os.ReadFile(outside); err != nil || string(b) != "classified" {
		t.Fatalf("file outside root was touched: %q %v", b, err)
	}
}

// TestLocalPutIsAtomic guards the temp-file-and-rename path: a reader must
// never observe a half-written scan image.
func TestLocalPutOverwriteLeavesNoTemp(t *testing.T) {
	s := newTestLocal(t)
	ctx := context.Background()
	key := "scans/2026/09/x.jpg"

	for i := 0; i < 3; i++ {
		if err := s.Put(ctx, key, strings.NewReader("v"), "image/jpeg"); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}

	dir := filepath.Join(s.Root(), "scans", "2026", "09")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".tmp-") {
			t.Errorf("leftover temp file: %s", e.Name())
		}
	}
	if len(entries) != 1 {
		t.Errorf("expected exactly 1 file, got %d", len(entries))
	}
}

func TestNewKeyIsPartitionedAndUnique(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 200; i++ {
		k, err := NewKey("scans", ".jpg")
		if err != nil {
			t.Fatal(err)
		}
		if seen[k] {
			t.Fatalf("duplicate key %q", k)
		}
		seen[k] = true

		if !strings.HasPrefix(k, "scans/") || !strings.HasSuffix(k, ".jpg") {
			t.Fatalf("unexpected key shape: %q", k)
		}
		// scans/YYYY/MM/<hex>.jpg
		if parts := strings.Split(k, "/"); len(parts) != 4 {
			t.Fatalf("expected date partitioning, got %q", k)
		}
		if err := validateKey(k); err != nil {
			t.Fatalf("generated key rejected by validateKey: %v", err)
		}
	}
}

func TestNewKeyNormalizesExtension(t *testing.T) {
	k, err := NewKey("heatmaps", "JPG")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(k, ".jpg") {
		t.Errorf("extension not normalized: %q", k)
	}
}
