package middleware

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"net/http"
	"strings"
)

type csrfCtxKey struct{}

// CSRFConfig configures the CSRF middleware.
type CSRFConfig struct {
	CookieName string // Name of the CSRF cookie (e.g. "portal_csrf").
	FieldName  string // HTML hidden form field name (e.g. "_csrf").
	CookiePath string // Cookie path scope.
}

// CSRFMiddleware implements the double-submit cookie CSRF protection pattern.
//
// On every request it reads the existing CSRF cookie (or generates a fresh
// 16-byte random token), stores it as a cookie, and injects the value into
// the request context so handlers can embed it in rendered forms.
//
// On POST requests it validates that the submitted form field (FieldName)
// matches the cookie value, responding with 403 Forbidden on mismatch.
// r.FormValue is used so the check works for both application/x-www-form-urlencoded
// and multipart/form-data bodies.
func CSRFMiddleware(cfg CSRFConfig) func(http.Handler) http.Handler {
	if cfg.CookieName == "" {
		cfg.CookieName = "csrf_token"
	}
	if cfg.FieldName == "" {
		cfg.FieldName = "_csrf"
	}
	if cfg.CookiePath == "" {
		cfg.CookiePath = "/"
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Read existing token or generate a new one.
			token := ""
			if cookie, err := r.Cookie(cfg.CookieName); err == nil && cookie.Value != "" {
				token = cookie.Value
			}
			if token == "" {
				b := make([]byte, 16)
				_, _ = rand.Read(b)
				token = hex.EncodeToString(b)
				http.SetCookie(w, &http.Cookie{
					Name:     cfg.CookieName,
					Value:    token,
					Path:     cfg.CookiePath,
					HttpOnly: true,
					SameSite: http.SameSiteLaxMode,
				})
			}

			// Inject token into context so handlers can pass it to templates.
			ctx := context.WithValue(r.Context(), csrfCtxKey{}, token)

			// Validate every state-changing method, not just POST. HTMX issues
			// DELETE/PUT/PATCH directly (hx-delete on the scan detail page), and
			// checking POST alone would leave those routes unprotected.
			if isStateChanging(r.Method) {
				// FormValue handles url-encoded and multipart bodies; the header
				// is what HTMX sends for non-POST verbs, which carry no form body.
				submitted := r.Header.Get("X-CSRF-Token")
				if submitted == "" {
					// FormValue would parse a multipart body with Go's 32 MiB
					// in-memory default, which would hold a whole PDF report on
					// the heap before the handler ever sees it. Parsing first
					// with a small budget spills the large parts to a temp file
					// instead; a non-multipart body errors here and FormValue
					// then handles it as before.
					if isMultipart(r) {
						_ = r.ParseMultipartForm(multipartMemory)
					}
					submitted = r.FormValue(cfg.FieldName)
				}
				if submitted == "" || subtle.ConstantTimeCompare([]byte(submitted), []byte(token)) != 1 {
					http.Error(w, "Permintaan tidak valid. Silakan muat ulang halaman dan coba lagi.", http.StatusForbidden)
					return
				}
			}

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// GetCSRFToken retrieves the CSRF token from the request context.
// Returns an empty string if the middleware was not applied.
func GetCSRFToken(r *http.Request) string {
	if token, ok := r.Context().Value(csrfCtxKey{}).(string); ok {
		return token
	}
	return ""
}

// isStateChanging reports whether a method may mutate server state and so
// requires a CSRF token. GET/HEAD/OPTIONS/TRACE are safe by definition.
func isStateChanging(method string) bool {
	switch method {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		return true
	default:
		return false
	}
}

// multipartMemory is how much of a multipart upload is kept in memory before
// the rest spills to a temp file.
const multipartMemory = 1 << 20

func isMultipart(r *http.Request) bool {
	ct := r.Header.Get("Content-Type")
	return strings.HasPrefix(ct, "multipart/form-data")
}
