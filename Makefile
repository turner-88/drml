.PHONY: help build run test css css-watch sqlc export eval seed schema db-shell docker clean

# Database targets go through scripts/with-env.sh so they read the same .env the
# server does — make does not load it on its own. scripts/db.sh then connects
# over TCP rather than the unix socket, because MariaDB authenticates local
# socket connections for root via unix_socket, ignoring DB_PASSWORD.
ENVRUN = scripts/with-env.sh

# Tailwind is compiled ahead of time by the standalone CLI - no Node, no npm.
# The binary is downloaded on demand into bin/, which .gitignore already covers.
# static/css/app.css is committed so `go build` alone still produces a working
# binary on a machine that has never run this target.
TAILWIND_VERSION ?= v4.3.3
TAILWIND_BIN     := bin/tailwindcss

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  TW_OS := macos
else
  TW_OS := linux
endif
ifeq ($(filter arm64 aarch64,$(UNAME_M)),)
  TW_ARCH := x64
else
  TW_ARCH := arm64
endif
TW_ASSET := tailwindcss-$(TW_OS)-$(TW_ARCH)

$(TAILWIND_BIN):
	@mkdir -p bin
	curl -fsSL "https://github.com/tailwindlabs/tailwindcss/releases/download/$(TAILWIND_VERSION)/$(TW_ASSET)" -o $(TAILWIND_BIN)
	chmod +x $(TAILWIND_BIN)

css: $(TAILWIND_BIN) ## Compile the stylesheet
	$(TAILWIND_BIN) -i static/css/app.src.css -o static/css/app.css --minify

css-watch: $(TAILWIND_BIN) ## Recompile the stylesheet on change
	$(TAILWIND_BIN) -i static/css/app.src.css -o static/css/app.css --watch

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: css ## Build the server and seed binaries
	go build -trimpath -o bin/drml ./cmd/server
	go build -trimpath -o bin/seed ./cmd/seed

run: css ## Run the server
	go run ./cmd/server

test: ## Run all Go tests
	go test ./...

sqlc: ## Regenerate database code from SQL
	sqlc generate

schema: ## Create/reset the database schema (DESTRUCTIVE - drops all tables)
	@$(ENVRUN) scripts/db.sh schema


db-shell: ## Open a MySQL shell on the app database
	@$(ENVRUN) scripts/db.sh shell


seed: ## Create the first administrator (SEED_PASSWORD and SEED_EMAIL required)
	@if [ -z "$$SEED_PASSWORD" ] || [ -z "$$SEED_EMAIL" ]; then \
		echo "error: SEED_PASSWORD and SEED_EMAIL are required, e.g."; \
		echo "       SEED_PASSWORD='ChangeMe123' SEED_EMAIL=admin@example.com make seed"; \
		exit 1; \
	fi
	go run ./cmd/seed -username $${SEED_USER:-admin} -password "$$SEED_PASSWORD" -email "$$SEED_EMAIL"

export: ## Re-export the ONNX model from HuggingFace
	cd model && ./.venv/bin/python export.py

eval: ## Verify class ordering and int8 vs fp32
	cd model && ./.venv/bin/python eval.py --per-class 30

docker: ## Build and start the full stack
	cd deploy && docker compose up -d --build

clean:
	rm -rf bin/
