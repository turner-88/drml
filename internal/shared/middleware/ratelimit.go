package middleware

import (
	"net/http"
	"strings"
	"sync"
	"time"
)

// RateLimiterConfig configures the rate limiting middleware.
type RateLimiterConfig struct {
	MaxAttempts int           // Maximum POST requests allowed per IP within Window.
	Window      time.Duration // Time window for the rate limit counter.
	RedirectURL string        // If set, redirect here on limit exceeded; otherwise return 429.
	Message     string        // Plain-text body returned with 429 (ignored when RedirectURL is set).
}

type rateLimitEntry struct {
	count   int
	resetAt time.Time
}

type rateLimiter struct {
	mu      sync.Mutex
	entries map[string]*rateLimitEntry
	cfg     RateLimiterConfig
}

// NewLoginRateLimiter returns a chi-compatible middleware that restricts POST
// requests from a single IP to cfg.MaxAttempts within cfg.Window.
// Defaults: 5 attempts per 15 minutes.
func NewLoginRateLimiter(cfg RateLimiterConfig) func(http.Handler) http.Handler {
	if cfg.MaxAttempts == 0 {
		cfg.MaxAttempts = 5
	}
	if cfg.Window == 0 {
		cfg.Window = 15 * time.Minute
	}
	if cfg.Message == "" {
		cfg.Message = "Terlalu banyak percobaan. Silakan coba lagi dalam beberapa menit."
	}
	rl := &rateLimiter{
		entries: make(map[string]*rateLimitEntry),
		cfg:     cfg,
	}
	go rl.cleanup()
	return rl.handler
}

func (rl *rateLimiter) clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		return strings.TrimSpace(strings.SplitN(xff, ",", 2)[0])
	}
	addr := r.RemoteAddr
	if i := strings.LastIndexByte(addr, ':'); i >= 0 {
		addr = addr[:i]
	}
	return addr
}

func (rl *rateLimiter) handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			next.ServeHTTP(w, r)
			return
		}

		ip := rl.clientIP(r)
		now := time.Now()

		rl.mu.Lock()
		entry, ok := rl.entries[ip]
		if !ok || now.After(entry.resetAt) {
			entry = &rateLimitEntry{count: 0, resetAt: now.Add(rl.cfg.Window)}
			rl.entries[ip] = entry
		}
		entry.count++
		exceeded := entry.count > rl.cfg.MaxAttempts
		rl.mu.Unlock()

		if exceeded {
			if rl.cfg.RedirectURL != "" {
				http.Redirect(w, r, rl.cfg.RedirectURL, http.StatusSeeOther)
				return
			}
			http.Error(w, rl.cfg.Message, http.StatusTooManyRequests)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// cleanup periodically removes expired entries to avoid unbounded memory growth.
func (rl *rateLimiter) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		now := time.Now()
		rl.mu.Lock()
		for ip, entry := range rl.entries {
			if now.After(entry.resetAt) {
				delete(rl.entries, ip)
			}
		}
		rl.mu.Unlock()
	}
}
