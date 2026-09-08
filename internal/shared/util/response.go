package util

import (
	"encoding/json"
	"net/http"
)

// JSONResponse represents a standard JSON response
type JSONResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message,omitempty"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// WriteJSON writes a JSON response
func WriteJSON(w http.ResponseWriter, status int, data interface{}) error {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	return json.NewEncoder(w).Encode(data)
}

// WriteSuccess writes a successful JSON response
func WriteSuccess(w http.ResponseWriter, message string, data interface{}) error {
	return WriteJSON(w, http.StatusOK, JSONResponse{
		Success: true,
		Message: message,
		Data:    data,
	})
}

// WriteError writes an error JSON response
func WriteError(w http.ResponseWriter, status int, message string) error {
	return WriteJSON(w, status, JSONResponse{
		Success: false,
		Error:   message,
	})
}

// WriteBadRequest writes a 400 Bad Request response
func WriteBadRequest(w http.ResponseWriter, message string) error {
	return WriteError(w, http.StatusBadRequest, message)
}

// WriteUnauthorized writes a 401 Unauthorized response
func WriteUnauthorized(w http.ResponseWriter, message string) error {
	return WriteError(w, http.StatusUnauthorized, message)
}

// WriteForbidden writes a 403 Forbidden response
func WriteForbidden(w http.ResponseWriter, message string) error {
	return WriteError(w, http.StatusForbidden, message)
}

// WriteNotFound writes a 404 Not Found response
func WriteNotFound(w http.ResponseWriter, message string) error {
	return WriteError(w, http.StatusNotFound, message)
}

// WriteInternalError writes a 500 Internal Server Error response
func WriteInternalError(w http.ResponseWriter, message string) error {
	return WriteError(w, http.StatusInternalServerError, message)
}
