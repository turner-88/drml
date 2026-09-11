// Package web wires the HTTP routes.
package web

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"

	scansvc "github.com/remorac/drml/internal/app/scan"
	"github.com/remorac/drml/internal/app/web/handler"
	"github.com/remorac/drml/internal/shared/config"
	mw "github.com/remorac/drml/internal/shared/middleware"
	"github.com/remorac/drml/internal/shared/model"
)

// Routes builds the application router.
func Routes(cfg *config.Config, h *handler.Handler) chi.Router {
	r := chi.NewRouter()

	// Bound login attempts per IP so a stolen username list cannot be
	// brute-forced against bcrypt at leisure.
	loginRL := mw.NewLoginRateLimiter(mw.RateLimiterConfig{
		MaxAttempts: 5,
		Window:      15 * time.Minute,
		RedirectURL: "/login?blocked=1",
	})

	// Separate budget for sign-ups: it bounds bulk account creation and
	// username probing from one address without eating into login attempts.
	registerRL := mw.NewLoginRateLimiter(mw.RateLimiterConfig{
		MaxAttempts: 5,
		Window:      time.Hour,
		RedirectURL: "/register?blocked=1",
	})

	// Each forgot-password request can send an email, so this budget also
	// bounds how much mail one address can aim at someone else's inbox.
	forgotRL := mw.NewLoginRateLimiter(mw.RateLimiterConfig{
		MaxAttempts: 5,
		Window:      time.Hour,
		RedirectURL: "/forgot-password?blocked=1",
	})
	// Tokens are unguessable; this only bounds bcrypt work per address.
	resetRL := mw.NewLoginRateLimiter(mw.RateLimiterConfig{
		MaxAttempts: 10,
		Window:      time.Hour,
		RedirectURL: "/forgot-password?blocked=1",
	})

	csrf := mw.CSRFMiddleware(mw.CSRFConfig{
		CookieName: "drml_csrf",
		FieldName:  "_csrf",
		CookiePath: "/",
	})

	auth := mw.RequireAuth(&mw.AuthConfig{
		JWTSecret:   cfg.JWT.SecretKey,
		RedirectURL: "/login",
		CookieName:  "drml_token",
	})

	// Ahead of CSRF on purpose: that middleware parses the multipart body to
	// read the token, so this is the last point at which an oversized upload
	// can be cut off before it is written to a temp file.
	r.Use(mw.MaxBodyBytes(scansvc.MaxPDFBytes))

	r.Group(func(r chi.Router) {
		r.Use(csrf)

		// Public
		r.Get("/login", h.LoginPage)
		r.With(loginRL).Post("/login", h.Login)
		r.Get("/logout", h.Logout)
		// 404 unless an administrator has enabled public registration.
		r.Get("/register", h.RegisterPage)
		r.With(registerRL).Post("/register", h.Register)
		// 404 unless outgoing mail is configured.
		r.Get("/forgot-password", h.ForgotPasswordPage)
		r.With(forgotRL).Post("/forgot-password", h.ForgotPassword)
		r.Get("/reset-password", h.ResetPasswordPage)
		r.With(resetRL).Post("/reset-password", h.ResetPassword)

		// Authenticated — clinician and admin
		r.Group(func(r chi.Router) {
			r.Use(auth)

			r.Get("/", h.Dashboard)
			r.Get("/analytics", h.AnalyticsPage)

			r.Get("/scans", h.ScanList)
			r.Get("/scans/new", h.NewScanPage)
			r.Post("/scans", h.CreateScan)
			r.Get("/scans/{id}", h.ScanDetail)
			// A plain form POST rather than DELETE: the UI ships no JavaScript
			// framework, and forms cannot issue any verb but GET or POST.
			r.Post("/scans/{id}/delete", h.DeleteScan)

			// Admin only
			r.Group(func(r chi.Router) {
				r.Use(mw.RequireRole(model.UserRoleAdmin))

				r.Get("/admin/users", h.UserList)
				r.Get("/admin/users/new", h.NewUserPage)
				r.Post("/admin/users", h.CreateUser)
				r.Post("/admin/users/{id}/suspend", h.SuspendUser)

				r.Get("/admin/settings", h.SettingsPage)
				r.Post("/admin/settings", h.UpdateSettings)
			})
		})
	})

	// Liveness probe, outside auth and CSRF.
	r.Get("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	return r
}
