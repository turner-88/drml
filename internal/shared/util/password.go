package util

import (
	"bufio"
	"crypto/sha1"
	"fmt"
	"log"
	"net/http"
	"strings"
	"unicode"

	"github.com/remorac/drml/internal/shared/config"
)

// ValidatePassword checks the given password against the configured policy.
// Rules are applied in order: length → complexity → blacklist → breach check.
// Returns a user-facing error message (in Indonesian) on the first violation,
// or nil when the password passes all enabled checks.
func ValidatePassword(password string, cfg *config.PasswordPolicyConfig) error {
	// 1. Length
	if len(password) < 8 {
		return fmt.Errorf("Password minimal 8 karakter.")
	}
	if len(password) > 64 {
		return fmt.Errorf("Password maksimal 64 karakter.")
	}

	// 2. Complexity
	switch cfg.ComplexityLevel {
	case "medium":
		if !containsDigit(password) {
			return fmt.Errorf("Password harus mengandung angka.")
		}
	case "hard":
		if !containsDigit(password) {
			return fmt.Errorf("Password harus mengandung angka.")
		}
		if !containsUpper(password) {
			return fmt.Errorf("Password harus mengandung huruf kapital.")
		}
	case "extreme":
		if !containsDigit(password) {
			return fmt.Errorf("Password harus mengandung angka.")
		}
		if !containsUpper(password) {
			return fmt.Errorf("Password harus mengandung huruf kapital.")
		}
		if !containsSpecial(password) {
			return fmt.Errorf("Password harus mengandung karakter khusus.")
		}
	}

	// 3. Blacklist
	if cfg.BlacklistEnabled {
		lower := strings.ToLower(password)
		pwLen := len(lower)
		for _, blocked := range BlacklistedPasswords() {
			if strings.Contains(lower, blocked) && len(blocked)*2 >= pwLen {
				return fmt.Errorf("Password terlalu umum. Gunakan password yang lebih unik.")
			}
		}
	}

	// 4. HIBP breach check (k-anonymity — password never leaves the server)
	if cfg.BreachCheckEnabled {
		if breached, err := isBreachedPassword(password); err != nil {
			log.Printf("ValidatePassword: HIBP check failed: %v", err)
			// fail-open: don't block user on network errors
		} else if breached {
			return fmt.Errorf("Password ini pernah bocor dalam data breach. Gunakan password lain.")
		}
	}

	return nil
}

// containsDigit returns true if s contains at least one decimal digit.
func containsDigit(s string) bool {
	for _, r := range s {
		if unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

// containsUpper returns true if s contains at least one uppercase letter.
func containsUpper(s string) bool {
	for _, r := range s {
		if unicode.IsUpper(r) {
			return true
		}
	}
	return false
}

// containsSpecial returns true if s contains at least one special character.
func containsSpecial(s string) bool {
	const specials = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
	for _, r := range s {
		if strings.ContainsRune(specials, r) {
			return true
		}
	}
	return false
}

// isBreachedPassword uses the HIBP Pwned Passwords k-anonymity API to check
// whether the password has appeared in a known data breach.
// Only the first 5 characters of the SHA-1 hash are sent; the full hash never
// leaves this process.
func isBreachedPassword(password string) (bool, error) {
	sum := sha1.Sum([]byte(password))
	hex := fmt.Sprintf("%X", sum)
	prefix, suffix := hex[:5], hex[5:]

	resp, err := http.Get("https://api.pwnedpasswords.com/range/" + prefix)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		// Each line is "SUFFIX:count"
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 && strings.EqualFold(parts[0], suffix) {
			return true, nil
		}
	}
	return false, scanner.Err()
}
