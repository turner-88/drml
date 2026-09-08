package util

// BlacklistedPasswords returns a list of commonly used weak passwords.
// ValidatePassword checks against this list (case-insensitive) when
// PasswordPolicyConfig.BlacklistEnabled is true.
func BlacklistedPasswords() []string {
	return []string{
		"password",
		"12345678",
		"bismillah",
		"katasandi",
		"semangat",
		"admin", "admin123",
		"sandi", "sandi123",
		"sukses", "sukses12",
		"rahasia", "rahasia1",
		"abc12345", "abcd1234", "abcde123",
		"qwerty", "qwerty123", "qwertyuiop",
	}
}
