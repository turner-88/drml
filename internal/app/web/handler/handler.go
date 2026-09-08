// Package handler renders the DRML web UI.
package handler

import (
	"context"
	"database/sql"
	"fmt"
	"html/template"
	"io/fs"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/remorac/drml/internal/app/scan"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/infer"
	"github.com/remorac/drml/internal/shared/config"
	mw "github.com/remorac/drml/internal/shared/middleware"
	"github.com/remorac/drml/internal/shared/model"
	"github.com/remorac/drml/internal/storage"
)

// LowConfidenceThreshold is the probability below which a prediction is
// surfaced for human review. The served model is ~70% accurate, so flagging
// its own uncertainty is a core feature rather than a diagnostic.
const LowConfidenceThreshold = 0.5

// Handler holds shared dependencies.
type Handler struct {
	cfg     *config.Config
	store   *store.Store
	scans   *scan.Service
	engine  *infer.Engine
	storage storage.Storage

	tmplFS fs.FS
	pages  map[string]*template.Template
}

// New parses every page template up front so a syntax error fails at startup
// rather than on the first request that renders it.
func New(
	cfg *config.Config,
	st *store.Store,
	scans *scan.Service,
	engine *infer.Engine,
	blobs storage.Storage,
	tmplFS fs.FS,
) (*Handler, error) {
	h := &Handler{
		cfg: cfg, store: st, scans: scans, engine: engine,
		storage: blobs, tmplFS: tmplFS, pages: map[string]*template.Template{},
	}

	pages := []string{
		"login", "dashboard", "scan_new", "scan_detail", "scan_list",
		"analytics", "users", "user_form",
	}
	for _, name := range pages {
		t, err := template.New("base.html").Funcs(h.funcs()).ParseFS(
			tmplFS,
			"template/layouts/base.html",
			"template/partials/*.html",
			"template/pages/"+name+".html",
		)
		if err != nil {
			return nil, fmt.Errorf("parse template %q: %w", name, err)
		}
		h.pages[name] = t
	}
	return h, nil
}

func (h *Handler) funcs() template.FuncMap {
	return template.FuncMap{
		"gradeLabel":   gradeLabelID,
		"gradeBadge":   gradeBadgeClass,
		"roleLabel":    func(r int32) string { return model.RoleLabel(model.UserRole(r)) },
		"pct":          func(f float64) string { return fmt.Sprintf("%.1f%%", f*100) },
		"pct1":         func(f float64) string { return fmt.Sprintf("%.1f", f*100) },
		"unixDate":     func(ts int32) string { return time.Unix(int64(ts), 0).Format("02 Jan 2006 15:04") },
		"unixDateOnly": func(ts int32) string { return time.Unix(int64(ts), 0).Format("02 Jan 2006") },
		"add":          func(a, b int) int { return a + b },
		// Grades cross int/int16 boundaries between the DB, templates and the
		// label helpers; these keep the templates free of conversion noise.
		"int":         func(v int16) int { return int(v) },
		"int16":       func(v int) int16 { return int16(v) },
		"seq":         seq,
		"isReferable": func(g int16) bool { return g >= 2 },
	}
}

// render writes a page, reporting template errors loudly rather than serving a
// half-written response body.
func (h *Handler) render(w http.ResponseWriter, r *http.Request, page string, data map[string]any) {
	t, ok := h.pages[page]
	if !ok {
		log.Printf("render: unknown page %q", page)
		http.Error(w, "Halaman tidak ditemukan", http.StatusInternalServerError)
		return
	}

	h.withCommon(r, data)

	// Render into a buffer first so a mid-template failure does not emit a
	// partial page with a 200 status.
	var buf strings.Builder
	if err := t.ExecuteTemplate(&buf, "base.html", data); err != nil {
		log.Printf("render %q: %v", page, err)
		http.Error(w, "Terjadi kesalahan saat menampilkan halaman", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(buf.String()))
}

// withCommon injects values every page needs.
func (h *Handler) withCommon(r *http.Request, data map[string]any) {
	data["Theme"] = h.cfg.AppTheme
	data["Year"] = time.Now().Year()
	data["CSRFToken"] = mw.GetCSRFToken(r)
	data["Path"] = r.URL.Path

	if u := mw.GetUserFromContext(r.Context()); u != nil {
		data["AuthUser"] = u
		data["IsAdmin"] = u.IsAdmin()
		data["RoleLabel"] = model.RoleLabel(u.Role)
	}

	// Surfaced in the UI banner: if eval.py has not confirmed that output index
	// i means grade i, every displayed label is an unverified assumption.
	data["ModelID"] = h.engine.ModelID()
	data["OrderingVerified"] = h.engine.OrderingVerified()
}

// scope resolves the (allScans, createdBy) pair for the current user.
func scope(r *http.Request) (int64, int32) {
	u := mw.GetUserFromContext(r.Context())
	all, id := u.ScanScope()
	if all {
		return 1, id
	}
	return 0, id
}

func actor(r *http.Request) *model.User { return mw.GetUserFromContext(r.Context()) }

// --- small helpers -----------------------------------------------------------

func gradeLabelID(g int16) string {
	switch g {
	case 0:
		return "Tidak ada DR"
	case 1:
		return "Ringan"
	case 2:
		return "Sedang"
	case 3:
		return "Berat"
	case 4:
		return "Proliferatif"
	default:
		return "Tidak diketahui"
	}
}

// gradeBadgeClass maps severity to DaisyUI badge colours; grades 2+ are
// referable and use warning/error tones.
func gradeBadgeClass(g int16) string {
	switch g {
	case 0:
		return "badge-success"
	case 1:
		return "badge-info"
	case 2:
		return "badge-warning"
	case 3, 4:
		return "badge-error"
	default:
		return "badge-ghost"
	}
}

func atoiDefault(s string, def int) int {
	if s == "" {
		return def
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return v
}

// parseDateParam accepts YYYY-MM-DD and returns a unix timestamp.
func parseDateParam(s string, endOfDay bool) (int32, bool) {
	if s == "" {
		return 0, false
	}
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		return 0, false
	}
	if endOfDay {
		t = t.Add(24*time.Hour - time.Second)
	}
	return int32(t.Unix()), true
}

// nullTS builds a nullable timestamp for the optional date filters.
func nullTS(v int32, ok bool) sql.NullInt32 {
	if !ok {
		return sql.NullInt32{}
	}
	return sql.NullInt32{Int32: v, Valid: true}
}

// blobURL resolves a storage key for templates, tolerating absent keys.
func (h *Handler) blobURL(ctx context.Context, key string) string {
	if key == "" {
		return ""
	}
	u, err := h.storage.URL(ctx, key)
	if err != nil {
		log.Printf("handler: resolve blob url %q: %v", key, err)
		return ""
	}
	return u
}

// seq returns 0..n-1 for templates that need to iterate a fixed range.
func seq(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}
