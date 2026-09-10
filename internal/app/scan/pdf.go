package scan

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"image/jpeg"
	"io"
	"log"

	"github.com/remorac/drml/internal/pdfdoc"
	"github.com/remorac/drml/internal/storage"
)

// ErrPDFTooLarge is returned when a report exceeds MaxPDFBytes.
var ErrPDFTooLarge = fmt.Errorf("ukuran pdf melebihi %d mb", MaxPDFBytes>>20)

// PDFResult is the scan recorded for one eye of a report.
type PDFResult struct {
	ID  int32
	Eye string
}

// fundusQuality is the JPEG quality for an extracted eye.
//
// Higher than the heatmap's 85: this image is the screening record itself, and
// it is the only copy a clinician looks at day to day.
const fundusQuality = 90

// newSourceRef returns an opaque token grouping the eyes of one report.
func newSourceRef() (string, error) {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "", fmt.Errorf("generate source ref: %w", err)
	}
	return hex.EncodeToString(buf[:]), nil
}

// CreateFromPDF screens every eye in an IMAGEnet report.
//
// src is read lazily, so passing the multipart file keeps the report off the
// heap. The eyes are screened one at a time: inference is serialised to a
// single slot anyway, and doing them concurrently would only stack their
// working images on top of each other.
//
// A non-nil error alongside a non-empty result means some eyes were recorded
// and others were not, which the caller reports rather than discarding the
// work that succeeded.
func (s *Service) CreateFromPDF(ctx context.Context, src io.ReaderAt, size int64, in Input) ([]PDFResult, error) {
	if size > MaxPDFBytes {
		return nil, ErrPDFTooLarge
	}

	eyes, err := pdfdoc.Extract(src, size)
	if err != nil {
		return nil, err
	}

	// Both eyes carry the same token, which is what links them on the detail
	// page. It is generated whether or not the report is kept, so the link does
	// not depend on a storage decision.
	sourceRef, err := newSourceRef()
	if err != nil {
		return nil, err
	}

	// The report itself is kept only if the deployment asked for it: it is
	// ~15 MB against ~1 MB for the images extracted from it. Stored before the
	// scans so every row can reference it.
	var sourceKey string
	if s.opts.KeepSourcePDF {
		sourceKey, err = storage.NewKey("sources", ".pdf")
		if err != nil {
			return nil, err
		}
		if err := s.storage.Put(ctx, sourceKey, io.NewSectionReader(src, 0, size), "application/pdf"); err != nil {
			return nil, fmt.Errorf("simpan dokumen PDF: %w", err)
		}
	}

	var (
		out      []PDFResult
		firstErr error
	)
	for _, f := range eyes {
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, f.Img, &jpeg.Options{Quality: fundusQuality}); err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("mata %s: encode gambar: %w", f.Eye, err)
			}
			continue
		}

		eyeIn := in
		eyeIn.Ext = ".jpg"
		eyeIn.Eye = string(f.Eye)
		eyeIn.SourceKind = SourcePDF
		eyeIn.SourceRef = sourceRef
		eyeIn.SourceKey = sourceKey

		id, err := s.createFromImage(ctx, f.Img, buf.Bytes(), eyeIn)
		if err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("mata %s: %w", f.Eye, err)
			}
			continue
		}
		out = append(out, PDFResult{ID: id, Eye: string(f.Eye)})
	}

	if len(out) == 0 {
		// Nothing was recorded, so the report has nothing pointing at it.
		if sourceKey != "" {
			if delErr := s.storage.Delete(ctx, sourceKey); delErr != nil {
				log.Printf("scan: orphaned source pdf %s: %v", sourceKey, delErr)
			}
		}
		if firstErr != nil {
			return nil, firstErr
		}
		return nil, pdfdoc.ErrNoFundus
	}
	return out, firstErr
}
