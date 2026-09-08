package storage

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

// R2Config configures the Cloudflare R2 backend.
type R2Config struct {
	AccountID       string
	AccessKeyID     string
	SecretAccessKey string
	Bucket          string
	// PresignExpiry bounds how long a generated URL stays valid.
	PresignExpiry time.Duration
}

// R2 stores objects in Cloudflare R2 over its S3-compatible API.
//
// Objects stay private: scan images are patient data, so they are never
// public-read. URL() issues a short-lived presigned GET instead.
type R2 struct {
	client  *s3.Client
	presign *s3.PresignClient
	bucket  string
	expiry  time.Duration
}

// NewR2 builds an R2-backed store.
func NewR2(cfg R2Config) (*R2, error) {
	if cfg.AccountID == "" || cfg.Bucket == "" {
		return nil, errors.New("storage: R2 account ID and bucket are required")
	}
	expiry := cfg.PresignExpiry
	if expiry <= 0 {
		expiry = 15 * time.Minute
	}

	endpoint := fmt.Sprintf("https://%s.r2.cloudflarestorage.com", cfg.AccountID)
	client := s3.New(s3.Options{
		// R2 ignores the region but the SDK requires one; "auto" is Cloudflare's
		// documented placeholder.
		Region:       "auto",
		BaseEndpoint: aws.String(endpoint),
		Credentials: credentials.NewStaticCredentialsProvider(
			cfg.AccessKeyID, cfg.SecretAccessKey, "",
		),
	})

	return &R2{
		client:  client,
		presign: s3.NewPresignClient(client),
		bucket:  cfg.Bucket,
		expiry:  expiry,
	}, nil
}

func (r *R2) Put(ctx context.Context, key string, body io.Reader, contentType string) error {
	if err := validateKey(key); err != nil {
		return err
	}
	// The S3 API needs a seekable body or a known length to sign the request;
	// buffering here keeps the interface's io.Reader contract intact. Scan
	// images are a few hundred KB, so this is cheap.
	buf, err := io.ReadAll(body)
	if err != nil {
		return fmt.Errorf("storage: read body: %w", err)
	}
	_, err = r.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(r.bucket),
		Key:         aws.String(key),
		Body:        newSeekableReader(buf),
		ContentType: aws.String(contentType),
	})
	if err != nil {
		return fmt.Errorf("storage: put %q: %w", key, err)
	}
	return nil
}

func (r *R2) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	if err := validateKey(key); err != nil {
		return nil, err
	}
	out, err := r.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(r.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		var nsk *types.NoSuchKey
		if errors.As(err, &nsk) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("storage: get %q: %w", key, err)
	}
	return out.Body, nil
}

func (r *R2) Delete(ctx context.Context, key string) error {
	if err := validateKey(key); err != nil {
		return err
	}
	_, err := r.client.DeleteObject(ctx, &s3.DeleteObjectInput{
		Bucket: aws.String(r.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return fmt.Errorf("storage: delete %q: %w", key, err)
	}
	return nil
}

func (r *R2) URL(ctx context.Context, key string) (string, error) {
	if err := validateKey(key); err != nil {
		return "", err
	}
	req, err := r.presign.PresignGetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(r.bucket),
		Key:    aws.String(key),
	}, s3.WithPresignExpires(r.expiry))
	if err != nil {
		return "", fmt.Errorf("storage: presign %q: %w", key, err)
	}
	return req.URL, nil
}

// newSeekableReader wraps a byte slice so the S3 SDK can rewind it when it
// signs and, if needed, retries the request.
func newSeekableReader(b []byte) io.ReadSeeker { return bytes.NewReader(b) }
