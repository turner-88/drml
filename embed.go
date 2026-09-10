// Package drml exposes the embedded static assets and templates so the binary
// is self-contained apart from the ONNX model and uploads.
package drml

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"io/fs"
)

// StaticFS holds CSS/JS/vendored libraries.
//
//go:embed static/css static/js static/cdn
var StaticFS embed.FS

// TemplateFS holds the HTML templates.
//
//go:embed template
var TemplateFS embed.FS

// StaticVersion fingerprints the embedded CSS and JS so the templates can hang
// a ?v= on their URLs.
//
// Those two are served under stable names with a one-day cache, so without it a
// browser keeps the previous upload.js for up to a day after a deploy: the
// freshly rendered template and the script that drives it disagree, and the
// script wins. static/cdn is left out because it carries its version in the
// filename already, which is what makes its longer cache safe.
var StaticVersion = staticVersion()

func staticVersion() string {
	sum := sha256.New()
	for _, dir := range []string{"static/css", "static/js"} {
		// Errors are swallowed on purpose: a fingerprint is a cache hint, and
		// failing to build one must not stop the server from booting. The worst
		// case is the stale-cache behaviour that existed before it.
		_ = fs.WalkDir(StaticFS, dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return err
			}
			b, err := StaticFS.ReadFile(path)
			if err != nil {
				return err
			}
			sum.Write([]byte(path))
			sum.Write(b)
			return nil
		})
	}
	return hex.EncodeToString(sum.Sum(nil))[:12]
}
