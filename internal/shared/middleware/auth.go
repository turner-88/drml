package middleware

import (
	"context"
	"net/http"
	"strings"

	"github.com/remorac/drml/internal/shared/model"
	"github.com/remorac/drml/internal/shared/util"
)

type contextKey string

const UserContextKey contextKey = "user"

// AuthConfig holds authentication middleware configuration
type AuthConfig struct {
	JWTSecret   string
	RedirectURL string // If set, redirect here on auth failure instead of returning 401
	CookieName  string // Cookie name to read token from; defaults to "auth_token"
}

// RequireAuth middleware validates JWT token and adds user to context.
// It checks the Authorization header first, then falls back to the configured cookie.
func RequireAuth(config *AuthConfig) func(http.Handler) http.Handler {
	cookieName := config.CookieName
	if cookieName == "" {
		cookieName = "auth_token"
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tokenString := ""

			// 1. Try Authorization: Bearer <token> header
			if authHeader := r.Header.Get("Authorization"); authHeader != "" {
				parts := strings.SplitN(authHeader, " ", 2)
				if len(parts) == 2 && parts[0] == "Bearer" {
					tokenString = parts[1]
				}
			}

			// 2. Fall back to the subsystem-specific cookie
			if tokenString == "" {
				if cookie, err := r.Cookie(cookieName); err == nil {
					tokenString = cookie.Value
				}
			}

			if tokenString == "" {
				authFail(w, r, config.RedirectURL, "Missing authorization")
				return
			}

			claims, err := util.ValidateToken(tokenString, config.JWTSecret)
			if err != nil {
				// Clear invalid cookie if present
				http.SetCookie(w, &http.Cookie{Name: cookieName, Value: "", Path: "/", MaxAge: -1})
				authFail(w, r, config.RedirectURL, "Invalid or expired token")
				return
			}

			user := &model.User{
				ID:       int32(claims.UserID),
				Username: claims.Username,
				Role:     claims.Role,
				Subrole:  claims.Subrole,
			}

			ctx := context.WithValue(r.Context(), UserContextKey, user)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// authFail redirects to redirectURL if set, otherwise returns a 401 JSON response.
func authFail(w http.ResponseWriter, r *http.Request, redirectURL, msg string) {
	if redirectURL != "" {
		http.Redirect(w, r, redirectURL, http.StatusFound)
		return
	}
	util.WriteUnauthorized(w, msg)
}

// GetUserFromContext retrieves the user from the request context
func GetUserFromContext(ctx context.Context) *model.User {
	user, ok := ctx.Value(UserContextKey).(*model.User)
	if !ok {
		return nil
	}
	return user
}

// RequireRole middleware checks if user has a specific role
func RequireRole(role model.UserRole) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			user := GetUserFromContext(r.Context())
			if user == nil {
				util.WriteUnauthorized(w, "User not authenticated")
				return
			}

			if user.Role != role {
				util.WriteForbidden(w, "Insufficient permissions")
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}
