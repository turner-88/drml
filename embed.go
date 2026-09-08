// Package drml exposes the embedded static assets and templates so the binary
// is self-contained apart from the ONNX model and uploads.
package drml

import "embed"

// StaticFS holds CSS/JS/vendored libraries.
//
//go:embed static/css static/js static/cdn
var StaticFS embed.FS

// TemplateFS holds the HTML templates.
//
//go:embed template
var TemplateFS embed.FS
