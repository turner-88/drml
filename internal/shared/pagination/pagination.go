package pagination

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// Pagination holds all data needed to render a pagination component.
type Pagination struct {
	Page       int
	TotalPages int
	TotalCount int64
	HasPrev    bool
	HasNext    bool
	PrevPage   int
	NextPage   int
	SeqStart   int
	SeqEnd     int
	PageBase   string // URL prefix ending with "&" or "?" for appending "page=N"
	Pages      []int  // page numbers with 0 = ellipsis sentinel
	InfoText   string // e.g. "Menampilkan 1–10 dari 50 data"; empty when showInfo is false
}

// New creates a Pagination from the essential inputs.
// When showInfo is true, InfoText is populated with "Menampilkan X–Y dari Z data".
func New(page int, total int64, pageSize int, baseURL string, showInfo bool) Pagination {
	tp := calcTotalPages(total, pageSize)
	if page < 1 {
		page = 1
	}
	if page > tp {
		page = tp
	}
	offset := (page - 1) * pageSize
	seqEnd := offset + pageSize
	if int64(seqEnd) > total {
		seqEnd = int(total)
	}

	p := Pagination{
		Page:       page,
		TotalPages: tp,
		TotalCount: total,
		HasPrev:    page > 1,
		HasNext:    page < tp,
		PrevPage:   page - 1,
		NextPage:   page + 1,
		SeqStart:   offset + 1,
		SeqEnd:     seqEnd,
		PageBase:   baseURL,
		Pages:      paginationPages(page, tp),
	}
	if showInfo && total > 0 {
		p.InfoText = fmt.Sprintf("Menampilkan %d\u2013%d dari %d data", p.SeqStart, p.SeqEnd, total)
	}
	return p
}

// ParsePageQ extracts the search query and page number from the request URL.
// Returns the raw query string, the 1-based page number, and the DB offset.
func ParsePageQ(r *http.Request, pageSize int) (q string, page int, offset int32) {
	q = strings.TrimSpace(r.URL.Query().Get("q"))
	p, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if p < 1 {
		p = 1
	}
	return q, p, int32((p - 1) * pageSize)
}

// ParseSortOrder extracts and validates sort column and direction from the request.
// allowed maps URL param name → DB column name. defaultCol is the DB column used
// when no valid sort param is present (param will be returned as "").
func ParseSortOrder(r *http.Request, allowed map[string]string, defaultCol string) (param, dbCol, dir string) {
	param = strings.TrimSpace(r.URL.Query().Get("sort"))
	dir = strings.TrimSpace(r.URL.Query().Get("order"))
	var ok bool
	dbCol, ok = allowed[param]
	if !ok {
		param = ""
		dbCol = defaultCol
	}
	if dir != "asc" && dir != "desc" {
		dir = "asc"
	}
	return
}

// calcTotalPages divides total rows by pageSize, rounding up.
func calcTotalPages(total int64, pageSize int) int {
	if total <= 0 {
		return 1
	}
	tp := int(total) / pageSize
	if int(total)%pageSize != 0 {
		tp++
	}
	return tp
}

// paginationPages returns the page numbers to render, with 0 as ellipsis sentinel.
// Wing = 2 pages on each side of current. Pages 1 and totalPages are always included.
func paginationPages(page, totalPages int) []int {
	if totalPages <= 1 {
		return nil
	}
	const wing = 2
	if totalPages <= wing*2+3 { // ≤7 pages: show all
		pages := make([]int, totalPages)
		for i := range pages {
			pages[i] = i + 1
		}
		return pages
	}
	var result []int
	prev := 0
	for p := 1; p <= totalPages; p++ {
		if p == 1 || p == totalPages || (p >= page-wing && p <= page+wing) {
			if prev != 0 && p > prev+1 {
				result = append(result, 0) // gap → ellipsis
			}
			result = append(result, p)
			prev = p
		}
	}
	return result
}
