package middleware

import (
	"net/http"
	"strconv"
)

// bodyOverhead is the slack allowed above the caller's limit for multipart
// boundaries and the other form fields, so a file exactly at the limit is not
// rejected for the few hundred bytes wrapped around it.
const bodyOverhead = 1 << 20

// MaxBodyBytes caps every request body.
//
// This has to run before the CSRF middleware, not inside a handler: CSRF reads
// the submitted token with FormValue, which parses the multipart body, so by
// the time a handler is reached the whole upload has already been streamed to a
// temp file. Wrapping r.Body there would be dead code.
//
// Declared lengths are refused before the body is read at all. That is not only
// cheaper: truncating the body instead would leave the CSRF middleware unable
// to find its token, and the clinician would be told to reload the page rather
// than that the file is too big.
//
// nginx's client_max_body_size is the outer guard in production; this keeps the
// same bound when the app is reached directly.
func MaxBodyBytes(limit int64) func(http.Handler) http.Handler {
	max := limit + bodyOverhead
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.ContentLength > max {
				http.Error(w,
					"Ukuran berkas melebihi batas "+strconv.FormatInt(limit>>20, 10)+" MB.",
					http.StatusRequestEntityTooLarge)
				return
			}
			// A chunked upload declares no length, so it still needs a hard
			// ceiling as it is read.
			if r.Body != nil {
				r.Body = http.MaxBytesReader(w, r.Body, max)
			}
			next.ServeHTTP(w, r)
		})
	}
}
