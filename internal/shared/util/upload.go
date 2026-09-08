package util

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// ImageTypes is the set of allowed MIME types for photo/image uploads,
// mapped to their canonical file extension.
var ImageTypes = map[string]string{
	"image/jpeg": ".jpg",
	"image/png":  ".png",
	"image/webp": ".webp",
}

// DocumentTypes is the set of allowed MIME types for document uploads.
var DocumentTypes = map[string]string{
	"application/pdf": ".pdf",
}

// VideoTypes is the set of allowed MIME types for video uploads.
var VideoTypes = map[string]string{
	"video/mp4":  ".mp4",
	"video/webm": ".webm",
}

// SaveUploadedFile reads the named file field from r, validates its MIME type
// against allowedTypes (e.g. ImageTypes), enforces maxSize bytes, writes the
// file with a UUID-style hex filename into destDir, and returns the
// web-accessible path (e.g. "/static/uploads/residents/abc123.jpg").
//
// Returns ("", nil) when the field is absent or empty — not an error.
// Returns ("", err) on validation or I/O failure.
func SaveUploadedFile(r *http.Request, fieldName string, allowedTypes map[string]string, maxSize int64, destDir string) (string, error) {
	file, header, err := r.FormFile(fieldName)
	if err != nil {
		// Field missing or no file selected — treat as no upload.
		return "", nil
	}
	defer file.Close()

	if header.Size > maxSize {
		return "", fmt.Errorf("ukuran file terlalu besar (maks %d KB)", maxSize/1024)
	}

	// Detect MIME type from the first 512 bytes.
	buf := make([]byte, 512)
	n, err := file.Read(buf)
	if err != nil && err != io.EOF {
		return "", fmt.Errorf("gagal membaca file")
	}
	mimeType := strings.Split(http.DetectContentType(buf[:n]), ";")[0]

	ext, ok := allowedTypes[mimeType]
	if !ok {
		return "", fmt.Errorf("tipe file tidak didukung: %s", mimeType)
	}

	// Seek back to the start so we write the complete file.
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return "", fmt.Errorf("gagal memproses file")
	}

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", fmt.Errorf("gagal membuat direktori upload")
	}

	filename := randomHex(16) + ext
	dst := filepath.Join(destDir, filename)

	out, err := os.Create(dst)
	if err != nil {
		return "", fmt.Errorf("gagal menyimpan file")
	}
	defer out.Close()

	if _, err := io.Copy(out, file); err != nil {
		os.Remove(dst) //nolint:errcheck
		return "", fmt.Errorf("gagal menyimpan file")
	}

	// destDir is relative (e.g. "static/uploads/residents") → "/static/uploads/residents/file.jpg"
	return "/" + filepath.ToSlash(filepath.Join(destDir, filename)), nil
}

// AttachmentTypes combines ImageTypes and DocumentTypes for letter attachments
// (KTP/KK can be a photo or a scanned PDF).
var AttachmentTypes = func() map[string]string {
	m := make(map[string]string, len(ImageTypes)+len(DocumentTypes))
	for k, v := range ImageTypes {
		m[k] = v
	}
	for k, v := range DocumentTypes {
		m[k] = v
	}
	return m
}()

// SaveMultipleFiles handles <input type="file" multiple> uploads. It validates
// each file's MIME type and size, saves them with random hex filenames into
// destDir, and returns the web-accessible paths as a slice.
//
// Returns (nil, nil) when no files are provided — not an error.
// On any validation/IO failure, previously saved files in the batch are cleaned up.
func SaveMultipleFiles(r *http.Request, fieldName string, allowedTypes map[string]string, maxSizePerFile int64, maxFiles int, destDir string) ([]string, error) {
	if r.MultipartForm == nil {
		return nil, nil
	}
	files := r.MultipartForm.File[fieldName]
	if len(files) == 0 {
		return nil, nil
	}
	if len(files) > maxFiles {
		return nil, fmt.Errorf("maksimal %d file", maxFiles)
	}

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, fmt.Errorf("gagal membuat direktori upload")
	}

	var saved []string
	cleanup := func() {
		for _, p := range saved {
			os.Remove("." + p) //nolint:errcheck // best-effort cleanup
		}
	}

	for _, fh := range files {
		if fh.Size > maxSizePerFile {
			cleanup()
			return nil, fmt.Errorf("file %q terlalu besar (maks %d MB)", fh.Filename, maxSizePerFile/(1<<20))
		}

		file, err := fh.Open()
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("gagal membaca file %q", fh.Filename)
		}

		// Detect MIME type from the first 512 bytes.
		buf := make([]byte, 512)
		n, err := file.Read(buf)
		if err != nil && err != io.EOF {
			file.Close()
			cleanup()
			return nil, fmt.Errorf("gagal membaca file %q", fh.Filename)
		}
		mimeType := strings.Split(http.DetectContentType(buf[:n]), ";")[0]

		ext, ok := allowedTypes[mimeType]
		if !ok {
			file.Close()
			cleanup()
			return nil, fmt.Errorf("tipe file %q tidak didukung: %s", fh.Filename, mimeType)
		}

		if _, err := file.Seek(0, io.SeekStart); err != nil {
			file.Close()
			cleanup()
			return nil, fmt.Errorf("gagal memproses file %q", fh.Filename)
		}

		filename := randomHex(16) + ext
		dst := filepath.Join(destDir, filename)

		out, err := os.Create(dst)
		if err != nil {
			file.Close()
			cleanup()
			return nil, fmt.Errorf("gagal menyimpan file")
		}

		if _, err := io.Copy(out, file); err != nil {
			out.Close()
			file.Close()
			os.Remove(dst) //nolint:errcheck
			cleanup()
			return nil, fmt.Errorf("gagal menyimpan file")
		}
		out.Close()
		file.Close()

		webPath := "/" + filepath.ToSlash(filepath.Join(destDir, filename))
		saved = append(saved, webPath)
	}

	return saved, nil
}

// ParseAttachmentJSON unmarshals a JSON array string (from letter.attachment)
// into a Go string slice. Returns nil on empty/invalid input.
func ParseAttachmentJSON(raw string) []string {
	if raw == "" || raw == "[]" {
		return nil
	}
	var paths []string
	if err := json.Unmarshal([]byte(raw), &paths); err != nil {
		return nil
	}
	return paths
}

// IsImagePath returns true if the path ends with a known image extension.
func IsImagePath(path string) bool {
	lower := strings.ToLower(path)
	return strings.HasSuffix(lower, ".jpg") || strings.HasSuffix(lower, ".png") || strings.HasSuffix(lower, ".webp")
}

// DeleteUploadedFile removes an uploaded file from disk given its web-accessible
// URL path (e.g. "/static/uploads/posts/abc123.jpg"). Errors are returned but
// callers may choose to log and ignore them (best-effort cleanup).
func DeleteUploadedFile(urlPath string) error {
	if urlPath == "" || !strings.HasPrefix(urlPath, "/static/uploads/") {
		return nil
	}
	return os.Remove("." + urlPath)
}

// ExtractFileRefs parses a JSON string and recursively extracts all
// file paths matching /static/uploads/... or /static/qr/...
func ExtractFileRefs(jsonStr string) []string {
	var parsed interface{}
	if json.Unmarshal([]byte(jsonStr), &parsed) != nil {
		return nil
	}
	var refs []string
	extractFileRefs(parsed, &refs)
	return refs
}

func extractFileRefs(v interface{}, refs *[]string) {
	switch val := v.(type) {
	case string:
		if strings.HasPrefix(val, "/static/uploads/") || strings.HasPrefix(val, "/static/qr/") {
			*refs = append(*refs, val)
		}
	case []interface{}:
		for _, item := range val {
			extractFileRefs(item, refs)
		}
	case map[string]interface{}:
		for _, item := range val {
			extractFileRefs(item, refs)
		}
	}
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
