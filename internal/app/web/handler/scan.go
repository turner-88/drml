package handler

import (
	"database/sql"
	"encoding/json"
	"errors"
	"io"
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
	"github.com/remorac/drml/internal/pdfdoc"
	"github.com/remorac/drml/internal/shared/pagination"
)

// allowedUploadExt maps sniffed MIME types to canonical extensions.
var allowedUploadExt = map[string]string{
	"image/jpeg": ".jpg",
	"image/png":  ".png",
	"image/webp": ".webp",
}

// modePDF is the value the upload form posts when the PDF tab is selected.
const modePDF = "pdf"

// pdfMIME is what http.DetectContentType reports for a PDF.
const pdfMIME = "application/pdf"

// NewScanPage renders the upload form.
func (h *Handler) NewScanPage(w http.ResponseWriter, r *http.Request) {
	h.render(w, r, "scan_new", map[string]any{"Title": "Pemeriksaan Baru"})
}

// CreateScan handles the upload and runs inference synchronously.
func (h *Handler) CreateScan(w http.ResponseWriter, r *http.Request) {
	// The overall body size is already bounded by mw.MaxBodyBytes and, in
	// production, by nginx; the per-format cap is applied further in, once the
	// type has actually been sniffed, so a PDF never fails with a misleading
	// "too large" error just because the mode field was stale.
	//
	// The small in-memory budget here is deliberate: it pushes the upload into
	// a temp file, which is what lets a PDF be parsed through an io.ReaderAt
	// instead of being held on the heap.
	if err := r.ParseMultipartForm(1 << 20); err != nil {
		h.renderScanError(w, r, "Gagal memproses data unggahan. Pastikan ukuran berkas tidak melebihi "+
			strconv.Itoa(scansvc.MaxPDFBytes>>20)+" MB.")
		return
	}

	wantPDF := r.FormValue("mode") == modePDF

	file, header, err := r.FormFile("image")
	if err != nil {
		h.renderScanError(w, r, "Pilih file foto fundus terlebih dahulu.")
		return
	}
	defer file.Close()

	// Sniff the content type rather than trusting the filename extension.
	buf := make([]byte, 512)
	n, _ := file.Read(buf)
	mime := strings.Split(http.DetectContentType(buf[:n]), ";")[0]
	if _, err := file.Seek(0, 0); err != nil {
		h.renderScanError(w, r, "Gagal memproses file gambar.")
		return
	}

	// The sniffed type decides, not the posted mode: a mislabelled tab must not
	// be able to route a PDF into the image path or the other way round, and a
	// form posted from a browser where upload.js never ran must not be rejected
	// over a mode field it had no chance to correct. wantPDF only sharpens the
	// error message when the two disagree.
	if mime == pdfMIME {
		h.createScanFromPDF(w, r, file, header.Size)
		return
	}

	ext, ok := allowedUploadExt[mime]
	if !ok {
		if wantPDF {
			h.renderScanError(w, r, "Mode PDF dipilih, tetapi file yang diunggah bukan PDF ("+mime+").")
			return
		}
		h.renderScanError(w, r, "Format file tidak didukung: "+mime+". Gunakan JPG, PNG, atau WebP.")
		return
	}

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
			h.renderScanError(w, r, "Server sedang sibuk memproses antrean pemeriksaan lain. Silakan coba beberapa saat lagi.")
		case errors.Is(err, scansvc.ErrUnsupportedImage):
			h.renderScanError(w, r, "File tidak valid atau rusak sehingga tidak dapat dibaca sebagai gambar.")
		case errors.Is(err, infer.ErrNotFundus):
			// The failing feature stays in the log: it would mean nothing to a
			// clinician, and spelling it out would explain how to get past it.
			log.Printf("scan create: gate rejected upload: %v", err)
			h.renderScanError(w, r, "Citra tidak dikenali sebagai foto fundus retina. "+
				"Pastikan file yang diunggah adalah hasil foto kamera fundus asli, bukan foto biasa "+
				"atau tangkapan layar.")
		default:
			log.Printf("scan create: %v", err)
			h.renderScanError(w, r, "Gagal memproses pemeriksaan: "+err.Error())
		}
		return
	}

	http.Redirect(w, r, "/scans/"+strconv.Itoa(int(id)), http.StatusFound)
}

// createScanFromPDF screens every eye in an IMAGEnet report.
//
// file is the multipart part, which is an io.ReaderAt, so the report is parsed
// off disk rather than buffered.
func (h *Handler) createScanFromPDF(w http.ResponseWriter, r *http.Request, file io.ReaderAt, size int64) {
	u := actor(r)
	results, err := h.scans.CreateFromPDF(r.Context(), file, size, scansvc.Input{
		PatientRef: strings.TrimSpace(r.FormValue("patient_ref")),
		Notes:      strings.TrimSpace(r.FormValue("notes")),
		ActorID:    u.ID,
	})

	// Some eyes may have been graded even when err is set. Recording those and
	// telling the clinician which failed beats discarding good screenings.
	if len(results) == 0 {
		switch {
		case errors.Is(err, infer.ErrBusy):
			w.WriteHeader(http.StatusServiceUnavailable)
			h.renderScanError(w, r, "Server sedang sibuk memproses antrean pemeriksaan lain. Silakan coba beberapa saat lagi.")
		case errors.Is(err, pdfdoc.ErrNotSupportedPDF):
			h.renderScanError(w, r, "File PDF ini tidak dapat dibaca. Gunakan hasil ekspor PDF dari IMAGEnet, "+
				"atau unggah foto fundusnya langsung sebagai gambar.")
		case errors.Is(err, pdfdoc.ErrNoFundus):
			h.renderScanError(w, r, "Tidak ditemukan foto fundus di dalam PDF ini.")
		case errors.Is(err, pdfdoc.ErrAmbiguousEye):
			// Refusing beats guessing: a scan filed against the wrong eye is
			// worse than one the clinician has to upload manually.
			log.Printf("scan create pdf: %v", err)
			h.renderScanError(w, r, "Label mata (OD/OS) pada PDF tidak dapat dicocokkan dengan fotonya, "+
				"sehingga sisi mata tidak dapat dipastikan. Unggah foto fundusnya langsung sebagai gambar.")
		case errors.Is(err, scansvc.ErrPDFTooLarge):
			h.renderScanError(w, r, "Ukuran PDF melebihi "+
				strconv.Itoa(scansvc.MaxPDFBytes>>20)+" MB.")
		case errors.Is(err, infer.ErrNotFundus):
			log.Printf("scan create pdf: gate rejected extracted image: %v", err)
			h.renderScanError(w, r, "Citra yang diambil dari PDF tidak dikenali sebagai foto fundus retina.")
		default:
			log.Printf("scan create pdf: %v", err)
			h.renderScanError(w, r, "Gagal memproses PDF: "+err.Error())
		}
		return
	}

	if err != nil {
		log.Printf("scan create pdf: partial success (%d recorded): %v", len(results), err)
	}
	http.Redirect(w, r, "/scans/"+strconv.Itoa(int(results[0].ID)), http.StatusFound)
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

	// A scan taken from a report links to the other eye, which is what the
	// shared source_ref buys instead of an exam table. The two conditions are
	// separate: the grouping always exists, while the source document is only
	// there when the deployment chose to keep it.
	if row.SourceRef.Valid && row.SourceRef.String != "" {
		siblings, err := h.store.ListScansBySourceRef(r.Context(), db.ListScansBySourceRefParams{
			SourceRef: row.SourceRef,
			AllScans:  all,
			CreatedBy: store.NullInt32(uid),
		})
		if err != nil {
			log.Printf("scan detail %d: siblings: %v", id, err)
		}
		for _, sib := range siblings {
			if sib.ID != row.ID {
				data["OtherEye"] = sib
				break
			}
		}
	}
	if row.SourceKey.Valid && row.SourceKey.String != "" {
		data["SourceURL"] = h.blobURL(r.Context(), row.SourceKey.String)
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
		http.Error(w, "Gagal menghapus data pemeriksaan", http.StatusInternalServerError)
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
		http.Error(w, "Gagal memuat riwayat pemeriksaan", http.StatusInternalServerError)
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
		http.Error(w, "Gagal memuat riwayat pemeriksaan", http.StatusInternalServerError)
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
