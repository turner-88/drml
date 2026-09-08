package main

import (
	"context"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	chimw "github.com/go-chi/chi/v5/middleware"

	assets "github.com/remorac/drml"
	"github.com/remorac/drml/internal/app/scan"
	"github.com/remorac/drml/internal/app/web"
	"github.com/remorac/drml/internal/app/web/handler"
	"github.com/remorac/drml/internal/database"
	"github.com/remorac/drml/internal/database/store"
	"github.com/remorac/drml/internal/infer"
	"github.com/remorac/drml/internal/shared/config"
	sharedmw "github.com/remorac/drml/internal/shared/middleware"
	"github.com/remorac/drml/internal/storage"
)

func main() {
	cfg := config.Load()

	db, err := database.Open(&cfg.DB)
	if err != nil {
		log.Fatalf("FATAL: open database: %v", err)
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	if err := db.PingContext(ctx); err != nil {
		cancel()
		log.Fatalf("FATAL: connect to database: %v", err)
	}
	cancel()
	log.Println("Database connected")

	// Loading the ONNX session is the slowest and most memory-hungry step;
	// doing it before the listener opens means a bad model fails the process
	// rather than every request.
	engine, err := infer.New(infer.Config{
		ModelPath:      cfg.Model.Path,
		LabelsPath:     cfg.Model.LabelsPath,
		PreprocessPath: cfg.Model.PreprocessPath,
		ORTLibPath:     cfg.ORT.LibPath,
		IntraOpThreads: cfg.ORT.IntraOpNumThread,
		InterOpThreads: cfg.ORT.InterOpNumThread,
		MaxQueueDepth:  cfg.ORT.MaxQueueDepth,
	})
	if err != nil {
		log.Fatalf("FATAL: load model: %v", err)
	}
	defer engine.Close()
	log.Printf("Model loaded: %s (sha256 %s…)", engine.ModelID(), engine.ModelSHA256()[:12])
	if !engine.OrderingVerified() {
		log.Println("WARNING: model class ordering is UNVERIFIED — run model/eval.py " +
			"before relying on displayed severity labels")
	}

	blobs, localRoot, err := newStorage(cfg)
	if err != nil {
		log.Fatalf("FATAL: init storage: %v", err)
	}

	st := store.New(db)
	scans := scan.NewService(engine, st, blobs, cfg.Model.HeatmapEnabled)

	h, err := handler.New(cfg, st, scans, engine, blobs, assets.TemplateFS)
	if err != nil {
		log.Fatalf("FATAL: parse templates: %v", err)
	}

	r := chi.NewRouter()
	r.Use(sharedmw.Logger)
	r.Use(chimw.Recoverer)
	r.Use(chimw.Compress(5))
	r.Use(chimw.RedirectSlashes)

	// Embedded assets: long cache, they are versioned by filename.
	staticSub, err := fs.Sub(assets.StaticFS, "static")
	if err != nil {
		log.Fatalf("FATAL: static assets: %v", err)
	}
	r.With(sharedmw.CacheControl(604800)).Handle("/static/cdn/*",
		http.StripPrefix("/static/", http.FileServer(http.FS(staticSub))))
	r.With(sharedmw.CacheControl(86400)).Handle("/static/css/*",
		http.StripPrefix("/static/", http.FileServer(http.FS(staticSub))))
	r.With(sharedmw.CacheControl(86400)).Handle("/static/js/*",
		http.StripPrefix("/static/", http.FileServer(http.FS(staticSub))))

	// Uploaded scans, only when using the local driver. These are patient
	// images, so they are served through the auth-protected router rather than
	// from a public bucket.
	if localRoot != "" {
		r.Group(func(r chi.Router) {
			r.Use(sharedmw.RequireAuth(&sharedmw.AuthConfig{
				JWTSecret:   cfg.JWT.SecretKey,
				RedirectURL: "/login",
				CookieName:  "drml_token",
			}))
			r.Handle(cfg.Storage.LocalURLPrefix+"/*", http.StripPrefix(
				cfg.Storage.LocalURLPrefix+"/", http.FileServer(http.Dir(localRoot))))
		})
	}

	r.Mount("/", web.Routes(cfg, h))

	srv := &http.Server{
		Addr:    cfg.ServerAddr(),
		Handler: r,
		// Inference is synchronous and takes 1-2s on the target hardware; the
		// write timeout must leave room for it plus the queue wait.
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      120 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		log.Printf("Server starting on http://%s", cfg.ServerAddr())
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("FATAL: server: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down...")

	// Long enough for an in-flight inference to finish rather than returning a
	// 502 to a clinician mid-scan.
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("FATAL: forced shutdown: %v", err)
	}
	log.Println("Server exited gracefully")
}

// newStorage builds the configured blob backend. The second return value is
// the local filesystem root, empty for remote drivers.
func newStorage(cfg *config.Config) (storage.Storage, string, error) {
	switch cfg.Storage.Driver {
	case "r2":
		s, err := storage.NewR2(storage.R2Config{
			AccountID:       cfg.Storage.R2AccountID,
			AccessKeyID:     cfg.Storage.R2AccessKeyID,
			SecretAccessKey: cfg.Storage.R2SecretAccessKey,
			Bucket:          cfg.Storage.R2Bucket,
			PresignExpiry:   time.Duration(cfg.Storage.R2PresignExpirySeconds) * time.Second,
		})
		if err != nil {
			return nil, "", err
		}
		log.Printf("Storage: Cloudflare R2 bucket %q", cfg.Storage.R2Bucket)
		return s, "", nil
	default:
		s, err := storage.NewLocal(cfg.Storage.LocalDir, cfg.Storage.LocalURLPrefix)
		if err != nil {
			return nil, "", err
		}
		log.Printf("Storage: local directory %q", s.Root())
		return s, s.Root(), nil
	}
}
