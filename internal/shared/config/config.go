package config

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

// Config holds all application configuration.
type Config struct {
	Server         ServerConfig
	JWT            JWTConfig
	DB             DBConfig
	Model          ModelConfig
	ORT            ORTConfig
	Storage        StorageConfig
	PasswordPolicy PasswordPolicyConfig
	Env            string
	AppURL         string
	PageSize       int
	AppTheme       string
}

// ServerConfig holds server-specific configuration.
type ServerConfig struct {
	Host string
	Port string
}

// JWTConfig holds JWT-specific configuration.
type JWTConfig struct {
	SecretKey       string
	ExpirationHours int
}

// DBConfig holds database connection configuration.
type DBConfig struct {
	Host     string
	Port     string
	Name     string
	User     string
	Password string
}

// DSN returns the MySQL Data Source Name string.
func (c *DBConfig) DSN() string {
	return fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?parseTime=true&charset=utf8mb4&collation=utf8mb4_unicode_ci",
		c.User, c.Password, c.Host, c.Port, c.Name)
}

// ModelConfig points at the exported ONNX artifacts produced by model/export.py.
type ModelConfig struct {
	// Path is the .onnx file loaded at startup.
	Path string
	// ID records which HuggingFace model produced it, stored on every scan row
	// so results stay interpretable after a model swap.
	ID string
	// LabelsPath and PreprocessPath are emitted alongside the model so Go never
	// hardcodes class names or normalization constants.
	LabelsPath     string
	PreprocessPath string
	// GatePath holds the calibrated thresholds for the non-fundus gate, written
	// by model/gate.py. Absent, the gate is disabled and every decodable image
	// is graded — see MODEL_GATE_ENABLED.
	GatePath string
	// HeatmapEnabled controls the attention-rollout overlay.
	//
	// Rollout is forward-only and correct as implemented, but on the default
	// checkpoint it is only weakly content-driven: two different fundus images
	// can yield maps correlating at ~0.82, and diseased images show a strong
	// top-edge bias rather than highlighting lesions. It is kept as an
	// exploratory aid with an explicit caveat in the UI; set
	// MODEL_HEATMAP_ENABLED=false to switch it off entirely.
	HeatmapEnabled bool
}

// ORTConfig configures the ONNX Runtime session.
//
// Thread counts default to the 2-vCPU target: one inference at a time, using
// both cores for the single operation rather than running two inferences that
// contend for the same cores.
type ORTConfig struct {
	// LibPath is the onnxruntime shared library. onnxruntime_go loads this
	// dynamically, so it must exist on the host (or be baked into the image).
	LibPath          string
	IntraOpNumThread int
	InterOpNumThread int
	// MaxQueueDepth bounds how many requests may wait for the inference
	// semaphore before the handler sheds load with 503.
	MaxQueueDepth int
}

// StorageConfig selects and configures the blob backend.
type StorageConfig struct {
	// Driver is "local" or "r2".
	Driver string
	// LocalDir is the on-disk root for the local driver.
	LocalDir string
	// LocalURLPrefix is the public path the local driver's files are served at.
	LocalURLPrefix string

	// R2 settings, used only when Driver == "r2".
	R2AccountID       string
	R2AccessKeyID     string
	R2SecretAccessKey string
	R2Bucket          string
	// R2PresignExpirySeconds bounds the lifetime of generated GET URLs. Scan
	// images are patient data, so objects stay private and are served presigned.
	R2PresignExpirySeconds int
}

// PasswordPolicyConfig controls password validation rules.
type PasswordPolicyConfig struct {
	// ComplexityLevel sets character-class requirements: easy | medium | hard | extreme
	//   easy    — no extra rules beyond length
	//   medium  — must contain a digit
	//   hard    — must contain a digit and an uppercase letter
	//   extreme — must contain a digit, uppercase letter, and special character
	ComplexityLevel  string
	BlacklistEnabled bool
	// BreachCheckEnabled queries HaveIBeenPwned via k-anonymity when a password
	// is set. Off by default: it puts an external network call in the critical
	// path of a clinical tool that may run on a restricted network.
	BreachCheckEnabled bool
}

// Load reads configuration from environment variables.
func Load() *Config {
	_ = godotenv.Load()

	cfg := &Config{
		Server: ServerConfig{
			Host: getEnv("SERVER_HOST", "localhost"),
			Port: getEnv("SERVER_PORT", "8080"),
		},
		JWT: JWTConfig{
			SecretKey:       getEnv("JWT_SECRET", "your-secret-key-change-in-production"),
			ExpirationHours: getEnvInt("JWT_EXPIRATION_HOURS", 24),
		},
		DB: DBConfig{
			Host:     getEnv("DB_HOST", "localhost"),
			Port:     getEnv("DB_PORT", "3306"),
			Name:     getEnv("DB_NAME", "drml"),
			User:     getEnv("DB_USER", "root"),
			Password: getEnv("DB_PASSWORD", ""),
		},
		Model: ModelConfig{
			Path:           getEnv("MODEL_PATH", "model/vit_dr_int8.onnx"),
			ID:             getEnv("MODEL_ID", "Kontawat/vit-diabetic-retinopathy-classification"),
			LabelsPath:     getEnv("MODEL_LABELS_PATH", "model/labels.json"),
			PreprocessPath: getEnv("MODEL_PREPROCESS_PATH", "model/preprocess.json"),
			GatePath:       getEnv("MODEL_GATE_PATH", "model/gate.json"),
			HeatmapEnabled: getEnvBool("MODEL_HEATMAP_ENABLED", true),
		},
		ORT: ORTConfig{
			LibPath:          getEnv("ORT_LIB_PATH", defaultORTLibPath()),
			IntraOpNumThread: getEnvInt("ORT_INTRA_OP_THREADS", 2),
			InterOpNumThread: getEnvInt("ORT_INTER_OP_THREADS", 1),
			MaxQueueDepth:    getEnvInt("ORT_MAX_QUEUE_DEPTH", 4),
		},
		Storage: StorageConfig{
			Driver:                 getEnv("STORAGE_DRIVER", "local"),
			LocalDir:               getEnv("STORAGE_LOCAL_DIR", "static/uploads"),
			LocalURLPrefix:         getEnv("STORAGE_LOCAL_URL_PREFIX", "/static/uploads"),
			R2AccountID:            getEnv("R2_ACCOUNT_ID", ""),
			R2AccessKeyID:          getEnv("R2_ACCESS_KEY_ID", ""),
			R2SecretAccessKey:      getEnv("R2_SECRET_ACCESS_KEY", ""),
			R2Bucket:               getEnv("R2_BUCKET", ""),
			R2PresignExpirySeconds: getEnvInt("R2_PRESIGN_EXPIRY_SECONDS", 900),
		},
		PasswordPolicy: PasswordPolicyConfig{
			ComplexityLevel:    getEnv("PASSWORD_COMPLEXITY_LEVEL", "medium"),
			BlacklistEnabled:   getEnvBool("PASSWORD_BLACKLIST_ENABLED", true),
			BreachCheckEnabled: getEnvBool("PASSWORD_BREACH_CHECK", false),
		},
		Env:      getEnv("ENV", "development"),
		AppURL:   getEnv("APP_URL", "http://localhost:"+getEnv("SERVER_PORT", "8080")),
		PageSize: getEnvInt("APP_PAGESIZE", 20),
		AppTheme: getEnv("APP_THEME", "lofi"),
	}

	// In production, reject insecure defaults and missing critical values.
	if cfg.IsProduction() {
		if cfg.JWT.SecretKey == "your-secret-key-change-in-production" {
			log.Fatal("FATAL: JWT_SECRET must be changed from default in production")
		}
		if cfg.DB.Password == "" {
			log.Fatal("FATAL: DB_PASSWORD must be set in production")
		}
	}
	if cfg.Storage.Driver == "r2" {
		for k, v := range map[string]string{
			"R2_ACCOUNT_ID":        cfg.Storage.R2AccountID,
			"R2_ACCESS_KEY_ID":     cfg.Storage.R2AccessKeyID,
			"R2_SECRET_ACCESS_KEY": cfg.Storage.R2SecretAccessKey,
			"R2_BUCKET":            cfg.Storage.R2Bucket,
		} {
			if v == "" {
				log.Fatalf("FATAL: %s must be set when STORAGE_DRIVER=r2", k)
			}
		}
	}

	log.Printf("Configuration loaded: %s:%s (env: %s, storage: %s)",
		cfg.Server.Host, cfg.Server.Port, cfg.Env, cfg.Storage.Driver)
	return cfg
}

// ServerAddr returns the full server address.
func (c *Config) ServerAddr() string {
	return c.Server.Host + ":" + c.Server.Port
}

// IsProduction returns true when ENV is set to "production".
func (c *Config) IsProduction() bool {
	return c.Env == "production"
}

// defaultORTLibPath guesses the onnxruntime shared library location so local
// development works without an explicit ORT_LIB_PATH. Production sets it.
func defaultORTLibPath() string {
	candidates := []string{
		"/opt/homebrew/lib/libonnxruntime.dylib", // macOS arm64 (Homebrew)
		"/usr/local/lib/libonnxruntime.dylib",    // macOS amd64 (Homebrew)
		"/usr/local/lib/libonnxruntime.so",       // Linux / container
		"/usr/lib/libonnxruntime.so",
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return candidates[0]
}

// getEnv retrieves an environment variable or returns a default value.
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvBool retrieves a boolean environment variable or returns a default value.
func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return strings.ToLower(value) == "true"
	}
	return defaultValue
}

// getEnvInt retrieves an integer environment variable or returns a default value.
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		var intValue int
		if _, err := fmt.Sscanf(value, "%d", &intValue); err == nil {
			return intValue
		}
	}
	return defaultValue
}
