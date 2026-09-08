package middleware

import (
	"fmt"
	"net/http"
)

// CacheControl returns a middleware that sets Cache-Control headers.
func CacheControl(maxAgeSeconds int) func(http.Handler) http.Handler {
	val := fmt.Sprintf("public, max-age=%d", maxAgeSeconds)
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Cache-Control", val)
			next.ServeHTTP(w, r)
		})
	}
}
