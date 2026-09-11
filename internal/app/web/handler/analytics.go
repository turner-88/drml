package handler

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"

	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/infer"
)

// defaultRangeDays is what a first visit sees: today plus the previous 29
// days, i.e. 30 columns on the trend chart.
const defaultRangeDays = 30

// scan.created_at is an int(11) of unix seconds, so a date outside the 32-bit
// epoch cannot match a row and would overflow the int32 filter on the way down.
var (
	rangeFloor = time.Date(2000, 1, 1, 0, 0, 0, 0, time.Local)
	rangeCeil  = time.Date(2038, 1, 1, 0, 0, 0, 0, time.Local)
)

// dateWindow is a fully resolved [from,to] filter; both ends are always set.
//
// An open-ended "to" was not a neutral default. It left one date input blank
// while its partner was filled, printed a dangling "sampai" in the subtitle,
// let future-dated rows into every total, and gave the trend chart no way to
// know which days in the window simply had no scans.
type dateWindow struct {
	FromDay time.Time     // local midnight of the first day in the window
	ToDay   time.Time     // local midnight of the last day in the window
	From    sql.NullInt32 // FromDay as unix seconds
	To      sql.NullInt32 // last second of ToDay, as unix seconds
	FromStr string        // YYYY-MM-DD, for <input type=date> and the subtitle
	ToStr   string
}

// Days is the inclusive number of calendar days in the window. The half-day
// nudge absorbs the 23- and 25-hour days a DST transition produces.
func (w dateWindow) Days() int {
	return int((w.ToDay.Sub(w.FromDay)+12*time.Hour)/(24*time.Hour)) + 1
}

// dateRange resolves the from/to filters shared by the analytics headline
// numbers and charts, always onto local-time day boundaries.
func dateRange(r *http.Request) dateWindow {
	q := r.URL.Query()
	fromDay, hasFrom := parseLocalDay(q.Get("from"))
	toDay, hasTo := parseLocalDay(q.Get("to"))

	// Resolve whichever end was given, then derive the other *from it*.
	// Defaulting a missing "from" to today instead meant that submitting only
	// "to" silently asked for the last 30 days ending now - a range that
	// usually starts after the requested "to", so every chart on the page went
	// empty at once and the filter looked broken.
	switch {
	case !hasFrom && !hasTo:
		toDay = todayLocal()
		fromDay = toDay.AddDate(0, 0, -(defaultRangeDays - 1))
	case !hasFrom:
		fromDay = toDay.AddDate(0, 0, -(defaultRangeDays - 1))
	case !hasTo:
		toDay = todayLocal()
		// A "from" in the future would otherwise invert the window.
		if toDay.Before(fromDay) {
			toDay = fromDay
		}
	}
	// A reversed range is a typo, not a request for nothing.
	if toDay.Before(fromDay) {
		fromDay, toDay = toDay, fromDay
	}
	fromDay, toDay = clampDay(fromDay), clampDay(toDay)

	return dateWindow{
		FromDay: fromDay,
		ToDay:   toDay,
		From:    store.NullInt32(int32(fromDay.Unix())),
		To:      store.NullInt32(int32(endOfLocalDay(toDay).Unix())),
		FromStr: fromDay.Format("2006-01-02"),
		ToStr:   toDay.Format("2006-01-02"),
	}
}

// todayLocal is local midnight of the current day. Anchoring the default range
// to midnight rather than to time.Now() keeps the oldest and newest columns
// whole days, so their bars are comparable to the ones between them and the
// window matches the dates the picker shows.
func todayLocal() time.Time {
	n := time.Now()
	return time.Date(n.Year(), n.Month(), n.Day(), 0, 0, 0, 0, n.Location())
}

func clampDay(t time.Time) time.Time {
	switch {
	case t.Before(rangeFloor):
		return rangeFloor
	case t.After(rangeCeil):
		return rangeCeil
	}
	return t
}

// Dashboard is the landing page: a welcome hero, shortcuts into the main
// areas, and the most recent scans. The headline numbers live on Analitik,
// next to the date filter that scopes them.
func (h *Handler) Dashboard(w http.ResponseWriter, r *http.Request) {
	all, uid := scope(r)

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

	h.render(w, r, "dashboard", map[string]any{
		"Title":  "Dashboard",
		"Hero":   true,
		"Recent": recent,
	})
}

// AnalyticsPage renders the headline numbers and the charts. Every series is
// computed here and shaped into presentation-ready percentages, so the
// templates draw plain CSS bars with no arithmetic and the page needs no
// charting library.
func (h *Handler) AnalyticsPage(w http.ResponseWriter, r *http.Request) {
	all, uid := scope(r)
	win := dateRange(r)

	summary, err := h.store.SummaryStats(r.Context(), db.SummaryStatsParams{
		LowConfidence: store.NullFloat64(LowConfidenceThreshold),
		AllScans:      all,
		CreatedBy:     store.NullInt32(uid),
		FromTs:        win.From,
		ToTs:          win.To,
	})
	if err != nil {
		log.Printf("analytics summary: %v", err)
		http.Error(w, "Gagal memuat ringkasan", http.StatusInternalServerError)
		return
	}

	// MySQL returns SUM()/AVG() as DECIMAL, which the driver hands back as
	// []byte; toFloat normalizes every shape these can take.
	referableCount := toFloat(summary.ReferableCount)
	var referableRate float64
	if summary.TotalScans > 0 {
		referableRate = referableCount / float64(summary.TotalScans)
	}

	series, err := h.analyticsSeries(r.Context(), all, uid, win)
	if err != nil {
		log.Printf("analytics series: %v", err)
		http.Error(w, "Gagal memuat data analitik", http.StatusInternalServerError)
		return
	}

	lowConf, err := h.store.LowConfidenceScans(r.Context(), db.LowConfidenceScansParams{
		LowConfidence: store.NullFloat64(LowConfidenceThreshold),
		AllScans:      all,
		CreatedBy:     store.NullInt32(uid),
		FromTs:        win.From,
		ToTs:          win.To,
		Limit:         15,
	})
	if err != nil {
		log.Printf("analytics low-confidence: %v", err)
		http.Error(w, "Gagal memuat data analitik", http.StatusInternalServerError)
		return
	}

	h.render(w, r, "analytics", map[string]any{
		"Title":         "Analitik",
		"Subtitle":           "Ringkasan hasil screening " + win.FromStr + " sampai " + win.ToStr,
		"Summary":            summary,
		"ReferableCount":     int64(referableCount),
		"ReferableRate":      referableRate,
		"MeanConfidence":     toFloat(summary.MeanConfidence),
		"LowConfidenceCount": int64(toFloat(summary.LowConfidenceCount)),
		"MeanInferenceMs":    int64(toFloat(summary.MeanInferenceMs)),
		"Grades":             series.Grades,
		"TotalGraded":        series.TotalGraded,
		"Days":               series.Days,
		"PeakDaily":          series.PeakDaily,
		"AxisMax":            series.AxisMax,
		"AxisMid":            series.AxisMid,
		"BinDays":            series.BinDays,
		"LowConfidence":      lowConf,
		"Threshold":          LowConfidenceThreshold,
		"From":               win.FromStr,
		"To":                 win.ToStr,
	})
}

// gradeSeries is one row of the distribution and confidence charts.
type gradeSeries struct {
	Grade       int16
	Label       string
	Count       int64
	SharePct    float64 // of all graded scans, 0-100
	MeanConfPct float64 // 0-100
	MinConfPct  float64 // 0-100
}

// daySeries is one column of the trend chart. Over a long window a column
// covers several days (see maxTrendCols), which is why it carries both ends.
type daySeries struct {
	Start, End   time.Time
	Day          string // ISO date of the first day in the bin
	Label        string // "11 Aug", or "11 Aug-17 Aug" once a bin is wider
	Total        int64
	Referable    int64
	RestPct      float64 // height of the grade 0-1 segment, 0-100
	ReferablePct float64 // height of the referable segment, 0-100
	Tick         bool    // this column carries an axis label
	AllReferable bool    // nothing but referable scans, so no segment above
}

// analyticsSeries is everything the charts draw.
type analyticsSeries struct {
	Grades      []gradeSeries
	TotalGraded int64
	Days        []daySeries
	PeakDaily   int64
	AxisMax     int64 // y-axis ceiling the bars are scaled against
	AxisMid     int64 // midpoint label, 0 when it would not be a whole number
	BinDays     int   // calendar days per column
}

// maxTrendCols caps the number of columns. Past it the bars would be thinner
// than the gaps between them, so a longer window folds days into wider bins
// rather than scrolling sideways - which would hide half the chart, break the
// print stylesheet, and be awkward on the tablets this runs on.
const maxTrendCols = 62

// analyticsSeries runs the three aggregate queries behind the charts and
// normalises their results for direct rendering.
func (h *Handler) analyticsSeries(
	ctx context.Context, all int64, uid int32, win dateWindow,
) (analyticsSeries, error) {
	var out analyticsSeries

	out.Grades = make([]gradeSeries, infer.NumGrades)
	for i := range out.Grades {
		out.Grades[i].Grade = int16(i)
		out.Grades[i].Label = gradeLabelID(int16(i))
	}

	dist, err := h.store.GradeDistribution(ctx, db.GradeDistributionParams{
		AllScans: all, CreatedBy: store.NullInt32(uid), FromTs: win.From, ToTs: win.To,
	})
	if err != nil {
		return out, fmt.Errorf("grade distribution: %w", err)
	}
	// Index by grade so an absent grade stays a zero row rather than shifting
	// the categories under its neighbours.
	for _, row := range dist {
		if !row.PredictedGrade.Valid {
			continue
		}
		if g := int(row.PredictedGrade.Int16); g >= 0 && g < infer.NumGrades {
			out.Grades[g].Count = row.Total
			out.TotalGraded += row.Total
		}
	}
	if out.TotalGraded > 0 {
		for i := range out.Grades {
			out.Grades[i].SharePct = float64(out.Grades[i].Count) / float64(out.TotalGraded) * 100
		}
	}

	conf, err := h.store.ConfidenceByGrade(ctx, db.ConfidenceByGradeParams{
		AllScans: all, CreatedBy: store.NullInt32(uid), FromTs: win.From, ToTs: win.To,
	})
	if err != nil {
		return out, fmt.Errorf("confidence by grade: %w", err)
	}
	for _, row := range conf {
		if !row.PredictedGrade.Valid {
			continue
		}
		g := int(row.PredictedGrade.Int16)
		if g < 0 || g >= infer.NumGrades {
			continue
		}
		out.Grades[g].MeanConfPct = toFloat(row.MeanConfidence) * 100
		out.Grades[g].MinConfPct = toFloat(row.MinConfidence) * 100
	}

	// The chart buckets by this offset in SQL and zero-fills by it here; both
	// have to read it off the same instant or the join on day_index drops
	// rows. One fixed offset per window means a DST transition inside the
	// window would shift the buckets on one side of it - theoretical in
	// Indonesia, but the reason this comes from a real instant.
	tzOffset := tzOffsetAt(win.FromDay)

	perDay, err := h.store.ScansPerDay(ctx, db.ScansPerDayParams{
		TzOffset:  store.NullInt32(int32(tzOffset)),
		AllScans:  all,
		CreatedBy: store.NullInt32(uid),
		FromTs:    win.From,
		ToTs:      win.To,
	})
	if err != nil {
		return out, fmt.Errorf("scans per day: %w", err)
	}

	// GROUP BY only emits days that had scans, so index them and walk the
	// whole window below instead. Rendering one column per returned row drew a
	// 30-day window with three active days as three full-width bars: it read
	// as a three-day history rather than as a mostly idle month.
	type totals struct{ total, referable int64 }
	byIndex := make(map[int64]totals, len(perDay))
	for _, row := range perDay {
		byIndex[int64(row.DayIndex)] = totals{
			total:     row.Total,
			referable: int64(toFloat(row.Referable)),
		}
	}

	nDays := win.Days()
	out.BinDays = (nDays + maxTrendCols - 1) / maxTrendCols
	if out.BinDays < 1 {
		out.BinDays = 1
	}

	for day, i := win.FromDay, 0; !day.After(win.ToDay); day, i = day.AddDate(0, 0, 1), i+1 {
		if i%out.BinDays == 0 {
			out.Days = append(out.Days, daySeries{Start: day, Day: day.Format("2006-01-02")})
		}
		col := &out.Days[len(out.Days)-1]
		t := byIndex[dayIndex(day, tzOffset)]
		col.Total += t.total
		col.Referable += t.referable
		col.End = day
	}

	for i := range out.Days {
		col := &out.Days[i]
		col.Label = col.Start.Format("02 Jan")
		if col.End.After(col.Start) {
			col.Label += "-" + col.End.Format("02 Jan")
		}
		if col.Referable > col.Total { // SUM() cannot exceed COUNT(*); belt and braces.
			col.Referable = col.Total
		}
		col.AllReferable = col.Total > 0 && col.Referable == col.Total
		if col.Total > out.PeakDaily {
			out.PeakDaily = col.Total
		}
	}

	// Scale against a rounded ceiling rather than the peak itself. Dividing by
	// the peak made the busiest column exactly full height in every window, so
	// a quiet week was indistinguishable from a busy one and there was no
	// scale to read a value off.
	out.AxisMax = niceMax(out.PeakDaily)
	if out.AxisMax%2 == 0 {
		out.AxisMid = out.AxisMax / 2
	}
	if out.AxisMax > 0 {
		for i := range out.Days {
			d := &out.Days[i]
			d.ReferablePct = float64(d.Referable) / float64(out.AxisMax) * 100
			// The two segments stack, so the lower one's height comes out of
			// the upper one instead of being painted over it.
			d.RestPct = float64(d.Total-d.Referable) / float64(out.AxisMax) * 100
		}
	}
	markAxisTicks(out.Days)

	return out, nil
}

const daySeconds = 86400

// tzOffsetAt is the zone offset in seconds east of UTC at t.
func tzOffsetAt(t time.Time) int64 {
	_, off := t.Zone()
	return int64(off)
}

// dayIndex is the bucket key for a local midnight, matching ScansPerDay's
// (created_at + tz_offset) DIV 86400.
func dayIndex(day time.Time, tzOffset int64) int64 {
	return (day.Unix() + tzOffset) / daySeconds
}

// niceMax rounds a peak up to the next 1/2/5 x 10^n, so the y axis carries
// round numbers a reader can interpolate against.
func niceMax(v int64) int64 {
	if v <= 0 {
		return 0
	}
	for step := int64(1); ; step *= 10 {
		for _, m := range []int64{1, 2, 5} {
			if n := m * step; n >= v {
				return n
			}
		}
	}
}

// markAxisTicks labels roughly five evenly spaced columns, always including
// the last: the end of the window is the date the reader came to check.
// Labelling only the first and last, as this chart used to, said nothing about
// where any column in between sat in time.
func markAxisTicks(days []daySeries) {
	n := len(days)
	if n == 0 {
		return
	}
	const want = 5
	step := (n + want - 1) / want
	if step < 1 {
		step = 1
	}
	last := 0
	for i := 0; i < n; i += step {
		days[i].Tick = true
		last = i
	}
	// Drop a penultimate tick that would collide with the end label.
	if last != n-1 && n-1-last < step/2 {
		days[last].Tick = false
	}
	days[n-1].Tick = true
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
