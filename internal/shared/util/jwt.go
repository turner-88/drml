package util

import (
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/remorac/drml/internal/shared/model"
)

var (
	ErrInvalidToken = errors.New("invalid token")
	ErrExpiredToken = errors.New("token has expired")
)

// Claims represents JWT claims
type Claims struct {
	UserID   int64          `json:"user_id"`
	Username string         `json:"username,omitempty"`
	Role     model.UserRole `json:"role"`
	Subrole  *int32         `json:"subrole,omitempty"`
	jwt.RegisteredClaims
}

// JWTConfig holds JWT configuration
type JWTConfig struct {
	SecretKey       string
	ExpirationHours int
}

// GenerateToken generates a new JWT token for a user
func GenerateToken(user *model.User, config *JWTConfig) (string, error) {
	claims := Claims{
		UserID:   int64(user.ID),
		Username: user.Username,
		Role:     user.Role,
		Subrole:  user.Subrole,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour * time.Duration(config.ExpirationHours))),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(config.SecretKey))
}

// ValidateToken validates a JWT token and returns the claims
func ValidateToken(tokenString string, secretKey string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		return []byte(secretKey), nil
	})

	if err != nil {
		return nil, err
	}

	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return nil, ErrInvalidToken
	}

	// Check expiration
	if claims.ExpiresAt != nil && claims.ExpiresAt.Before(time.Now()) {
		return nil, ErrExpiredToken
	}

	return claims, nil
}
