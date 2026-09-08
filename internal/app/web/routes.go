// Package web wires the HTTP routes.
package web

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"

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

	r.Group(func(r chi.Router) {
		r.Use(csrf)

		// Public
		r.Get("/login", h.LoginPage)
		r.With(loginRL).Post("/login", h.Login)
		r.Get("/logout", h.Logout)

		// Authenticated — clinician and admin
		r.Group(func(r chi.Router) {
			r.Use(auth)

			r.Get("/", h.Dashboard)
			r.Get("/analytics", h.AnalyticsPage)
			r.Get("/api/analytics", h.AnalyticsData)

			r.Get("/scans", h.ScanList)
			r.Get("/scans/new", h.NewScanPage)
			r.Post("/scans", h.CreateScan)
			r.Get("/scans/{id}", h.ScanDetail)
			r.Delete("/scans/{id}", h.DeleteScan)

			// Admin only
			r.Group(func(r chi.Router) {
				r.Use(mw.RequireRole(model.UserRoleAdmin))

				r.Get("/admin/users", h.UserList)
				r.Get("/admin/users/new", h.NewUserPage)
				r.Post("/admin/users", h.CreateUser)
				r.Post("/admin/users/{id}/suspend", h.SuspendUser)
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
