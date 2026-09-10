// Package scan implements the screening workflow: accept a fundus image, run
// inference, persist the result and its heatmap.
package scan

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"time"

	"golang.org/x/image/webp"

	db "github.com/remorac/drml/internal/database/sqlc"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/infer"
	"github.com/remorac/drml/internal/storage"
)

func init() {
	// Register WebP for image.Decode; phones commonly produce it and the upload
	// validator already accepts image/webp.
	image.RegisterFormat("webp", "RIFF????WEBPVP8", webp.Decode, webp.DecodeConfig)
}

// MaxImageBytes bounds an upload. Fundus images are a few hundred KB once the
// browser has downscaled them; this is a backstop against memory exhaustion on
// a 2 GB box, not a quality limit.
const MaxImageBytes = 12 << 20 // 12 MiB

// ErrUnsupportedImage is returned when the upload is not a decodable image.
var ErrUnsupportedImage = errors.New("format gambar tidak didukung")

// Service runs the screening workflow.
type Service struct {
	engine  *infer.Engine
	store   *store.Store
	storage storage.Storage
	// heatmaps gates the saliency overlay; see config.ModelConfig.
	heatmaps bool
}

func NewService(engine *infer.Engine, st *store.Store, blobs storage.Storage, heatmaps bool) *Service {
	return &Service{engine: engine, store: st, storage: blobs, heatmaps: heatmaps}
}

// Input describes one screening request.
type Input struct {
	PatientRef string
	Notes      string
	// Ext is the source file extension (".jpg"), used for the storage key.
	Ext string
	// ActorID is the user recording the scan.
	ActorID int32
}

// Create runs inference and persists the result.
//
// The original image is stored regardless of whether inference succeeds: a
// failed prediction still leaves a reviewable record rather than discarding
// clinical data.
func (s *Service) Create(ctx context.Context, r io.Reader, in Input) (int32, error) {
	raw, err := io.ReadAll(io.LimitReader(r, MaxImageBytes+1))
	if err != nil {
		return 0, fmt.Errorf("baca berkas: %w", err)
	}
	if len(raw) > MaxImageBytes {
		return 0, fmt.Errorf("ukuran gambar melebihi %d MB", MaxImageBytes>>20)
	}

	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return 0, ErrUnsupportedImage
	}
	// image.Decode drops EXIF orientation but the browser applies it, so a
	// rotated upload would be graded sideways and its heatmap drawn against a
	// differently rotated base image. Correcting it here, before anything else
	// touches the pixels, keeps the gate, the model, the overlay and the
	// clinician's view in agreement. The stored bytes and their digest are the
	// untouched original either way.
	img = infer.Upright(img, raw)

	// First-tier gate, before anything is stored and before Predict competes
	// for the single inference slot. A non-fundus upload leaves no blob and no
	// row: it is not a failed screening, it is not a screening at all.
	if rep := s.engine.CheckImage(img); !rep.Passed {
		return 0, fmt.Errorf("%w (%s score=%.3f red=%.2f contrast=%.2f)",
			infer.ErrNotFundus, rep.Reason,
			rep.Metrics["score"], rep.Metrics["red_ratio"],
			rep.Metrics["centre_contrast"])
	}

	sum := sha256.Sum256(raw)
	digest := hex.EncodeToString(sum[:])

	imageKey, err := storage.NewKey("scans", in.Ext)
	if err != nil {
		return 0, err
	}
	if err := s.storage.Put(ctx, imageKey, bytes.NewReader(raw), contentTypeFor(in.Ext)); err != nil {
		return 0, fmt.Errorf("simpan gambar: %w", err)
	}

	params := db.CreateScanParams{
		PatientRef:  store.NullString(in.PatientRef),
		Notes:       store.NullString(in.Notes),
		ImageKey:    imageKey,
		ImageSha256: digest,
		ModelID:     s.engine.ModelID(),
		ModelSha256: s.engine.ModelSHA256(),
		Status:      "done",
		CreatedAt:   store.NullInt32(int32(time.Now().Unix())),
		CreatedBy:   store.NullInt32(in.ActorID),
	}

	res, err := s.engine.Predict(ctx, img)
	if err != nil {
		// Transient failures, and gate rejections, must not leave a misleading
		// 'failed' row behind. ErrNotFundus is here for a different reason from
		// the rest: it is a verdict about the upload rather than a failure to
		// grade it, so there is no screening to record. The image is already
		// stored by this point, so drop it.
		if errors.Is(err, infer.ErrBusy) || errors.Is(err, infer.ErrNotFundus) ||
			errors.Is(err, context.Canceled) ||
			errors.Is(err, context.DeadlineExceeded) {
			if delErr := s.storage.Delete(ctx, imageKey); delErr != nil {
				log.Printf("scan: orphaned image %s: %v", imageKey, delErr)
			}
			return 0, err
		}
		// Everything else: keep the image and record the failure, so the
		// clinician sees an explicit error row instead of the upload silently
		// vanishing.
		params.Status = "failed"
		params.ErrorMessage = store.NullString(truncate(err.Error(), 255))
		id, dbErr := s.store.CreateScan(ctx, params)
		if dbErr != nil {
			return 0, fmt.Errorf("%w (and could not record failure: %v)", err, dbErr)
		}
		return id, nil
	}

	probs, err := json.Marshal(res.Probs)
	if err != nil {
		return 0, fmt.Errorf("encode probabilitas: %w", err)
	}

	params.PredictedGrade = store.NullInt16(int16(res.Grade))
	params.Confidence = store.NullFloat64(float64(res.Confidence))
	params.Probabilities = probs
	params.InferenceMs = store.NullInt32(int32(res.InferenceMS))

	// The heatmap is an aid, not the result: if rendering or storing it fails,
	// log it and keep the prediction rather than failing the whole scan.
	if s.heatmaps && res.Saliency != nil {
		key, err := s.storeHeatmap(ctx, img, res)
		switch {
		case err != nil:
			log.Printf("scan: heatmap render failed: %v", err)
		case key != "":
			params.HeatmapKey = store.NullString(key)
		}
	}

	return s.store.CreateScan(ctx, params)
}

// storeHeatmap renders and uploads the attention overlay.
//
// An empty key with a nil error means the saliency was too diffuse to draw
// honestly: the scan then has no heatmap at all, which the detail page already
// handles, rather than an overlay that would show the clinician a confident
// peak the model never had.
func (s *Service) storeHeatmap(ctx context.Context, src image.Image, res *infer.Result) (string, error) {
	opts := infer.DefaultHeatmapOptions()
	if opts.EffectiveAlpha(res.Saliency.Concentration) <= 0 {
		return "", nil
	}
	overlay := infer.RenderHeatmap(src, res.Saliency, opts)

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, overlay, &jpeg.Options{Quality: 85}); err != nil {
		return "", fmt.Errorf("encode heatmap: %w", err)
	}

	key, err := storage.NewKey("heatmaps", ".jpg")
	if err != nil {
		return "", err
	}
	if err := s.storage.Put(ctx, key, bytes.NewReader(buf.Bytes()), "image/jpeg"); err != nil {
		return "", fmt.Errorf("store heatmap: %w", err)
	}
	return key, nil
}

// Delete removes a scan and its stored objects, enforcing role scoping.
func (s *Service) Delete(ctx context.Context, id int32, allScans bool, actorID int32) error {
	row, err := s.store.GetScanForUser(ctx, db.GetScanForUserParams{
		ID:        id,
		AllScans:  boolToInt64(allScans),
		CreatedBy: store.NullInt32(actorID),
	})
	if err != nil {
		return err
	}

	if err := s.store.DeleteScan(ctx, db.DeleteScanParams{
		ID:        id,
		AllScans:  boolToInt64(allScans),
		CreatedBy: store.NullInt32(actorID),
	}); err != nil {
		return err
	}

	// Blobs are removed after the row, so a failure here leaves an orphaned
	// file rather than a row pointing at a missing image.
	for _, key := range []string{row.ImageKey, row.HeatmapKey.String} {
		if key == "" {
			continue
		}
		if err := s.storage.Delete(ctx, key); err != nil {
			log.Printf("scan: delete blob %s: %v", key, err)
		}
	}

	return nil
}

func boolToInt64(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

func contentTypeFor(ext string) string {
	switch ext {
	case ".png":
		return "image/png"
	case ".webp":
		return "image/webp"
	default:
		return "image/jpeg"
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
