.PHONY: help build run test sqlc export eval seed schema db-shell docker clean

# Database targets go through scripts/with-env.sh so they read the same .env the
# server does — make does not load it on its own. scripts/db.sh then connects
# over TCP rather than the unix socket, because MariaDB authenticates local
# socket connections for root via unix_socket, ignoring DB_PASSWORD.
ENVRUN = scripts/with-env.sh

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build the server and seed binaries
	go build -trimpath -o bin/drml ./cmd/server
	go build -trimpath -o bin/seed ./cmd/seed

run: ## Run the server
	go run ./cmd/server

test: ## Run all Go tests
	go test ./...

sqlc: ## Regenerate database code from SQL
	sqlc generate

schema: ## Create/reset the database schema (DESTRUCTIVE - drops all tables)
	@$(ENVRUN) scripts/db.sh schema


db-shell: ## Open a MySQL shell on the app database
	@$(ENVRUN) scripts/db.sh shell


seed: ## Create the first administrator (SEED_PASSWORD required)
	@if [ -z "$$SEED_PASSWORD" ]; then \
		echo "error: SEED_PASSWORD is required, e.g."; \
		echo "       SEED_PASSWORD='ChangeMe123' make seed"; \
		exit 1; \
	fi
	go run ./cmd/seed -username $${SEED_USER:-admin} -password "$$SEED_PASSWORD"

export: ## Re-export the ONNX model from HuggingFace
	cd model && ./.venv/bin/python export.py

eval: ## Verify class ordering and int8 vs fp32
	cd model && ./.venv/bin/python eval.py --per-class 30

docker: ## Build and start the full stack
	cd deploy && docker compose up -d --build

clean:
	rm -rf bin/
