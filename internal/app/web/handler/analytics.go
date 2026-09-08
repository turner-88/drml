package handler

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"time"

	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/infer"
)

// dateRange resolves the from/to filters shared by the dashboard and charts.
// Defaults to the last 30 days so a first visit shows something meaningful.
func dateRange(r *http.Request) (from, to sql.NullInt32, fromStr, toStr string) {
	q := r.URL.Query()
	fromStr, toStr = q.Get("from"), q.Get("to")

	if ts, ok := parseDateParam(fromStr, false); ok {
		from = sql.NullInt32{Int32: ts, Valid: true}
	} else {
		from = sql.NullInt32{Int32: int32(time.Now().AddDate(0, 0, -30).Unix()), Valid: true}
		fromStr = time.Now().AddDate(0, 0, -30).Format("2006-01-02")
	}
	if ts, ok := parseDateParam(toStr, true); ok {
		to = sql.NullInt32{Int32: ts, Valid: true}
	}
	return
}

// Dashboard is the landing page: headline numbers plus the most recent scans.
func (h *Handler) Dashboard(w http.ResponseWriter, r *http.Request) {
	all, uid := scope(r)
	from, to, fromStr, toStr := dateRange(r)

	summary, err := h.store.SummaryStats(r.Context(), db.SummaryStatsParams{
		LowConfidence: store.NullFloat64(LowConfidenceThreshold),
		AllScans:      all,
		CreatedBy:     store.NullInt32(uid),
		FromTs:        from,
		ToTs:          to,
	})
	if err != nil {
		log.Printf("dashboard summary: %v", err)
		http.Error(w, "Gagal memuat ringkasan", http.StatusInternalServerError)
		return
	}

	recent, err := h.store.RecentScans(r.Context(), db.RecentScansParams{
		AllScans:  all,
		CreatedBy: store.NullInt32(uid),
		Limit:     8,
	})
	if err != nil {
		log.Printf("dashboard recent: %v", err)
		http.Error(w, "Gagal memuat pemeriksaan terbaru", http.StatusInternalServerError)
		return
	}

	// MySQL returns SUM()/AVG() as DECIMAL, which the driver hands back as
	// []byte; toFloat normalizes every shape these can take.
	referableCount := toFloat(summary.ReferableCount)
	var referableRate float64
	if summary.TotalScans > 0 {
		referableRate = referableCount / float64(summary.TotalScans)
	}

	h.render(w, r, "dashboard", map[string]any{
		"Title":              "Dasbor",
		"Summary":            summary,
		"ReferableCount":     int64(referableCount),
		"MeanConfidence":     toFloat(summary.MeanConfidence),
		"LowConfidenceCount": int64(toFloat(summary.LowConfidenceCount)),
		"MeanInferenceMs":    int64(toFloat(summary.MeanInferenceMs)),
		"ReferableRate":      referableRate,
		"Recent":             recent,
		"From":               fromStr,
		"To":                 toStr,
	})
}

// AnalyticsPage renders the charts shell; the data arrives via the JSON API so
// the page itself stays cheap to render.
func (h *Handler) AnalyticsPage(w http.ResponseWriter, r *http.Request) {
	all, uid := scope(r)
	from, to, fromStr, toStr := dateRange(r)

	lowConf, err := h.store.LowConfidenceScans(r.Context(), db.LowConfidenceScansParams{
		LowConfidence: store.NullFloat64(LowConfidenceThreshold),
		AllScans:      all,
		CreatedBy:     store.NullInt32(uid),
		FromTs:        from,
		ToTs:          to,
		Limit:         15,
	})
	if err != nil {
		log.Printf("analytics low-confidence: %v", err)
		http.Error(w, "Gagal memuat data analitik", http.StatusInternalServerError)
		return
	}

	h.render(w, r, "analytics", map[string]any{
		"Title":         "Analitik",
		"LowConfidence": lowConf,
		"Threshold":     LowConfidenceThreshold,
		"From":          fromStr,
		"To":            toStr,
	})
}

// analyticsPayload is the shape Chart.js consumes.
type analyticsPayload struct {
	GradeLabels   []string  `json:"grade_labels"`
	GradeCounts   []int64   `json:"grade_counts"`
	Days          []string  `json:"days"`
	DailyTotal    []int64   `json:"daily_total"`
	DailyReferral []int64   `json:"daily_referable"`
	MeanConf      []float64 `json:"mean_confidence"`
	MinConf       []float64 `json:"min_confidence"`
}

// AnalyticsData serves the chart series as JSON.
func (h *Handler) AnalyticsData(w http.ResponseWriter, r *http.Request) {
	all, uid := scope(r)
	from, to, _, _ := dateRange(r)

	out := analyticsPayload{
		GradeLabels: make([]string, infer.NumGrades),
		GradeCounts: make([]int64, infer.NumGrades),
		MeanConf:    make([]float64, infer.NumGrades),
		MinConf:     make([]float64, infer.NumGrades),
	}
	for i := 0; i < infer.NumGrades; i++ {
		out.GradeLabels[i] = gradeLabelID(int16(i))
	}

	dist, err := h.store.GradeDistribution(r.Context(), db.GradeDistributionParams{
		AllScans: all, CreatedBy: store.NullInt32(uid), FromTs: from, ToTs: to,
	})
	if err != nil {
		log.Printf("analytics distribution: %v", err)
		http.Error(w, `{"error":"gagal memuat distribusi"}`, http.StatusInternalServerError)
		return
	}
	// Index by grade so absent grades render as 0 rather than shifting the
	// chart's categories.
	for _, row := range dist {
		if row.PredictedGrade.Valid {
			if g := int(row.PredictedGrade.Int16); g >= 0 && g < infer.NumGrades {
				out.GradeCounts[g] = row.Total
			}
		}
	}

	perDay, err := h.store.ScansPerDay(r.Context(), db.ScansPerDayParams{
		AllScans: all, CreatedBy: store.NullInt32(uid), FromTs: from, ToTs: to,
	})
	if err != nil {
		log.Printf("analytics per-day: %v", err)
		http.Error(w, `{"error":"gagal memuat tren harian"}`, http.StatusInternalServerError)
		return
	}
	for _, row := range perDay {
		out.Days = append(out.Days, row.Day)
		out.DailyTotal = append(out.DailyTotal, row.Total)
		out.DailyReferral = append(out.DailyReferral, int64(toFloat(row.Referable)))
	}

	conf, err := h.store.ConfidenceByGrade(r.Context(), db.ConfidenceByGradeParams{
		AllScans: all, CreatedBy: store.NullInt32(uid), FromTs: from, ToTs: to,
	})
	if err != nil {
		log.Printf("analytics confidence: %v", err)
		http.Error(w, `{"error":"gagal memuat konfidens"}`, http.StatusInternalServerError)
		return
	}
	for _, row := range conf {
		if !row.PredictedGrade.Valid {
			continue
		}
		g := int(row.PredictedGrade.Int16)
		if g < 0 || g >= infer.NumGrades {
			continue
		}
		out.MeanConf[g] = toFloat(row.MeanConfidence)
		out.MinConf[g] = toFloat(row.MinConfidence)
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if err := json.NewEncoder(w).Encode(out); err != nil {
		log.Printf("analytics encode: %v", err)
	}
}

// toFloat normalizes the assorted numeric types MySQL aggregates return
// (SUM yields DECIMAL as []byte, AVG yields NullFloat64).
func toFloat(v any) float64 {
	switch t := v.(type) {
	case nil:
		return 0
	case float64:
		return t
	case float32:
		return float64(t)
	case int64:
		return float64(t)
	case int32:
		return float64(t)
	case sql.NullFloat64:
		if t.Valid {
			return t.Float64
		}
		return 0
	case []byte:
		// DECIMAL arrives as its textual representation, e.g. "0.7123".
		f, err := strconv.ParseFloat(string(t), 64)
		if err != nil {
			return 0
		}
		return f
	case string:
		f, err := strconv.ParseFloat(t, 64)
		if err != nil {
			return 0
		}
		return f
	default:
		return 0
	}
}
