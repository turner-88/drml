package handler

import (
	"database/sql"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"

	scansvc "github.com/remorac/drml/internal/app/scan"
	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/infer"
	"github.com/remorac/drml/internal/shared/pagination"
)

// allowedUploadExt maps sniffed MIME types to canonical extensions.
var allowedUploadExt = map[string]string{
	"image/jpeg": ".jpg",
	"image/png":  ".png",
	"image/webp": ".webp",
}

// NewScanPage renders the upload form.
func (h *Handler) NewScanPage(w http.ResponseWriter, r *http.Request) {
	h.render(w, r, "scan_new", map[string]any{"Title": "Pemeriksaan Baru"})
}

// CreateScan handles the upload and runs inference synchronously.
func (h *Handler) CreateScan(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(scansvc.MaxImageBytes); err != nil {
		h.renderScanError(w, r, "Gagal membaca formulir unggahan.")
		return
	}

	file, header, err := r.FormFile("image")
	if err != nil {
		h.renderScanError(w, r, "Pilih berkas gambar fundus terlebih dahulu.")
		return
	}
	defer file.Close()

	// Sniff the content type rather than trusting the filename extension.
	buf := make([]byte, 512)
	n, _ := file.Read(buf)
	mime := strings.Split(http.DetectContentType(buf[:n]), ";")[0]
	ext, ok := allowedUploadExt[mime]
	if !ok {
		h.renderScanError(w, r, "Tipe berkas tidak didukung: "+mime+". Gunakan JPG, PNG, atau WebP.")
		return
	}
	if _, err := file.Seek(0, 0); err != nil {
		h.renderScanError(w, r, "Gagal memproses berkas.")
		return
	}
	_ = header

	u := actor(r)
	id, err := h.scans.Create(r.Context(), file, scansvc.Input{
		PatientRef: strings.TrimSpace(r.FormValue("patient_ref")),
		Notes:      strings.TrimSpace(r.FormValue("notes")),
		Ext:        ext,
		ActorID:    u.ID,
	})
	if err != nil {
		switch {
		case errors.Is(err, infer.ErrBusy):
			// Shed load rather than queueing: on 2 vCPU a backlog becomes swap.
			w.WriteHeader(http.StatusServiceUnavailable)
			h.renderScanError(w, r, "Server sedang sibuk memproses pemeriksaan lain. Coba lagi sebentar.")
		case errors.Is(err, scansvc.ErrUnsupportedImage):
			h.renderScanError(w, r, "Berkas tidak dapat dibaca sebagai gambar.")
		case errors.Is(err, infer.ErrNotFundus):
			// The failing feature stays in the log: it would mean nothing to a
			// clinician, and spelling it out would explain how to get past it.
			log.Printf("scan create: gate rejected upload: %v", err)
			h.renderScanError(w, r, "Gambar tidak dikenali sebagai foto fundus retina. "+
				"Pastikan yang diunggah adalah hasil kamera fundus, bukan foto biasa "+
				"atau tangkapan layar.")
		default:
			log.Printf("scan create: %v", err)
			h.renderScanError(w, r, "Gagal memproses pemeriksaan: "+err.Error())
		}
		return
	}

	http.Redirect(w, r, "/scans/"+strconv.Itoa(int(id)), http.StatusFound)
}

func (h *Handler) renderScanError(w http.ResponseWriter, r *http.Request, msg string) {
	h.render(w, r, "scan_new", map[string]any{
		"Title":      "Pemeriksaan Baru",
		"Error":      msg,
		"PatientRef": r.FormValue("patient_ref"),
		"Notes":      r.FormValue("notes"),
	})
}

// ScanDetail shows one prediction with its heatmap and probability breakdown.
func (h *Handler) ScanDetail(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(chi.URLParam(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}

	all, uid := scope(r)
	row, err := h.store.GetScanForUser(r.Context(), db.GetScanForUserParams{
		ID:        int32(id),
		AllScans:  all,
		CreatedBy: store.NullInt32(uid),
	})
	if err != nil {
		// A clinician requesting someone else's scan gets 404, not 403: a 403
		// would confirm the row exists.
		if !errors.Is(err, sql.ErrNoRows) {
			log.Printf("scan detail %d: %v", id, err)
		}
		http.NotFound(w, r)
		return
	}

	var probs []float64
	if len(row.Probabilities) > 0 {
		if err := json.Unmarshal(row.Probabilities, &probs); err != nil {
			log.Printf("scan detail %d: decode probabilities: %v", id, err)
		}
	}

	data := map[string]any{
		"Title":      "Hasil Pemeriksaan",
		"Scan":       row,
		"Probs":      probs,
		"ImageURL":   h.blobURL(r.Context(), row.ImageKey),
		"HeatmapURL": h.blobURL(r.Context(), row.HeatmapKey.String),
		"LowConfidence": row.Confidence.Valid &&
			row.Confidence.Float64 < LowConfidenceThreshold,
	}
	if row.CreatedBy.Valid {
		if creator, err := h.store.GetUserByID(r.Context(), row.CreatedBy.Int32); err == nil {
			data["CreatedByName"] = creator.Username
		}
	}
	h.render(w, r, "scan_detail", data)
}

// DeleteScan removes a scan the caller is allowed to touch.
func (h *Handler) DeleteScan(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(chi.URLParam(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	u := actor(r)
	all, _ := u.ScanScope()

	if err := h.scans.Delete(r.Context(), int32(id), all, u.ID); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			http.NotFound(w, r)
			return
		}
		log.Printf("scan delete %d: %v", id, err)
		http.Error(w, "Gagal menghapus pemeriksaan", http.StatusInternalServerError)
		return
	}

	http.Redirect(w, r, "/scans", http.StatusSeeOther)
}

// ScanList renders the paginated, filterable history.
func (h *Handler) ScanList(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	page := atoiDefault(q.Get("page"), 1)
	if page < 1 {
		page = 1
	}
	size := h.cfg.PageSize

	all, uid := scope(r)

	var grade sql.NullInt16
	if g := q.Get("grade"); g != "" {
		if v, err := strconv.Atoi(g); err == nil && v >= 0 && v < infer.NumGrades {
			grade = sql.NullInt16{Int16: int16(v), Valid: true}
		}
	}
	patientRef := store.NullString(strings.TrimSpace(q.Get("patient_ref")))
	fromTS, hasFrom := parseDateParam(q.Get("from"), false)
	toTS, hasTo := parseDateParam(q.Get("to"), true)

	total, err := h.store.CountScans(r.Context(), db.CountScansParams{
		AllScans:   all,
		CreatedBy:  store.NullInt32(uid),
		Grade:      grade,
		PatientRef: patientRef,
		FromTs:     nullTS(fromTS, hasFrom),
		ToTs:       nullTS(toTS, hasTo),
	})
	if err != nil {
		log.Printf("scan list count: %v", err)
		http.Error(w, "Gagal memuat riwayat", http.StatusInternalServerError)
		return
	}

	rows, err := h.store.ListScans(r.Context(), db.ListScansParams{
		AllScans:   all,
		CreatedBy:  store.NullInt32(uid),
		Grade:      grade,
		PatientRef: patientRef,
		FromTs:     nullTS(fromTS, hasFrom),
		ToTs:       nullTS(toTS, hasTo),
		Limit:      int32(size),
		Offset:     int32((page - 1) * size),
	})
	if err != nil {
		log.Printf("scan list: %v", err)
		http.Error(w, "Gagal memuat riwayat", http.StatusInternalServerError)
		return
	}

	// Preserve active filters across pagination links.
	base := "/scans?"
	for _, k := range []string{"grade", "patient_ref", "from", "to"} {
		if v := q.Get(k); v != "" {
			base += k + "=" + url.QueryEscape(v) + "&"
		}
	}

	h.render(w, r, "scan_list", map[string]any{
		"Title":      "Riwayat Pemeriksaan",
		"Scans":      rows,
		"Pagination": pagination.New(page, total, size, base, true),
		"Filter": map[string]string{
			"grade":       q.Get("grade"),
			"patient_ref": q.Get("patient_ref"),
			"from":        q.Get("from"),
			"to":          q.Get("to"),
		},
	})
}
