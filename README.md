# DRML — Diabetic Retinopathy Screening

A Go monolith that grades retina fundus images for diabetic retinopathy using a
**pretrained** model (no training), stores every prediction, and charts the
history.

> **Screening aid, not a diagnostic device.** The served model is roughly 70%
> accurate on 5-class grading. Every result must be confirmed by an
> ophthalmologist. The UI carries this disclaimer on every page.

---

## How it works

The model is exported to ONNX **once on a dev machine**; the server loads it
through the ONNX Runtime C API via cgo. There is no Python at runtime.

```
fundus image ──┐
               │
IMAGEnet PDF ──┤ (extract eyes: OD/OS from the report label)
               │
               └► Go: decode → non-fundus gate ──► rejected, nothing stored
                   │                    (52 µs, before the inference slot)
                   ├─► resize 224² → normalize
                   │
                   ├─► ONNX Runtime (ViT-B/16, int8, forward + backward baked in)
                   │      ├─► logits ──► softmax ──► grade 0–4
                   │      └─► attentions + attn_grads [12,1,12,197,197]
                   │             ──► gradient-weighted relevance ──► 14×14 heatmap
                   │
                   └─► MySQL row + image/heatmap blobs ──► history ──► analytics
```

**Why not Python?** The target is a 2 vCPU / 2 GB VPS. `import torch` alone is
~1 GB RSS before the model loads, which does not fit beside MySQL. ONNX Runtime
with an int8-quantized ViT measures **~470 MB** under sustained load, backward
pass included.

**How does the heatmap get gradients without PyTorch?** ONNX Runtime cannot
differentiate, so `export.py` differentiates the model in PyTorch
(`torch.func.grad`) and exports the resulting forward+backward computation as
one graph. ORT then produces d(predicted logit)/d(attention) in the same `Run`
as the logits, and Go turns it into the class-specific relevance map of
Chefer et al. (ICCV 2021). Plain attention rollout, which needs no gradients,
remains as the fallback for a graph exported with `--forward-only`; it is a
much weaker localizer — see [Known limitations](#known-limitations).

---

## Measured behaviour

Verified on this machine (Apple Silicon; a 2 vCPU x86 VPS will be several times
slower per inference, still within budget):

| Metric | Forward-only graph | Forward + backward graph (default) |
|---|---|---|
| int8 artifact on disk | 87 MB | 171 MB (backward weights are stored pre-transposed so they quantize) |
| RSS, model loaded | +152 MB | +271 MB |
| RSS, after 40 sequential predictions | +278 MB (plateaus, no creep) | +458 MB (plateaus, no creep) |
| Per prediction, incl. preprocessing, saliency and heatmap render | 103 ms | 189 ms |
| Concurrency | serialized to 1; excess sheds with `503` | same |

RSS deltas are over an idle Go process, measured in-process with
`DRML_MEASURE_RSS=1 go test ./internal/infer/ -run TestMemoryFootprint -v`.
ORT's memory-pattern planner is disabled (`SetMemPattern(false)`): on the
backward graph it pre-reserved ~75 MB for no latency gain.

Projected VPS budget: app ~510 MB + MySQL (tuned) ~450 MB + OS ~250 MB
≈ **1.2 GB of 2 GB**. If that is too tight, `export.py --forward-only` gives
the left column back at the cost of the class-specific heatmap.

---

## Model

Default: [`Kontawat/vit-diabetic-retinopathy-classification`](https://huggingface.co/Kontawat/vit-diabetic-retinopathy-classification)
— ViT-B/16, 224px, 5 classes, Apache-2.0.

The checkpoint ships a placeholder `id2label` of `{"0":"0",…,"4":"4"}`, so the
severity ordering was **not** documented. `model/eval.py` establishes it
empirically by brute-forcing all 120 index→grade permutations against two
independent labelled datasets:

| Dataset | Coverage | Result |
|---|---|---|
| `sngsfydy/aptos_test` | all 5 grades, numeric labels | identity best; acc 0.707, **QWK 0.854** |
| `EslamHasan/APTOS2019…` | grades 0–2, **named** labels | identity best; acc 0.833 |

The named-label dataset pins the semantics ("No DR", "Mild", "Moderate"); the
numeric one extends the check to grades 3–4. Both agree that output index *i*
means grade *i*, so `labels.json` records `ordering_verified: true`. If that
flag is false, the app shows a warning banner on every page.

Quantization is safe: int8 costs 0.7 points vs fp32 (0.700 vs 0.707).

> Accuracy figures are **optimistic** — APTOS's public labels are its train
> split, which this checkpoint was almost certainly fine-tuned on. What the run
> establishes is the class ordering, not held-out skill.

### Swapping the model

```bash
cd model
./.venv/bin/python export.py --model-id <hf-id>
./.venv/bin/python eval.py --per-class 30      # re-verify ordering
```

Go reads `labels.json` and `preprocess.json` at startup, so a model with
different normalization or input size needs no code change.

---

## Rejecting non-fundus uploads

**No DR model can do this, and swapping the model will not help.** Every
diabetic-retinopathy classifier — this one included — is a closed-set 5-way
softmax trained only on fundus images. It has no "not a retina" class and its
probabilities always sum to 1, so it grades a selfie as confidently as a real
scan. The rejection has to happen outside the model.

`internal/infer/gate.go` does it in two tiers, calibrated by `model/gate.py`
against 220 real fundus images and 240 non-fundus ones (food, everyday photos,
animals, aerial imagery). Thresholds live in `model/gate.json` and are read at
startup, like `preprocess.json`.

**Tier 1 — image heuristics.** Two absolute prefilters (shorter side ≥ 224 px,
aspect ≤ 2:1), then a weighted vote over three features measured inside the
illuminated disc:

| Feature | AUROC | Fundus mean | Non-fundus mean |
|---|---|---|---|
| Red dominance, `meanR/meanG` | 0.979 | 1.86 | 1.05 |
| Centre-vs-corner contrast | 0.981 | 0.71 | 0.01 |
| ROI texture, σ of luma | 0.223 | 0.067 | 0.160 |
| ↳ scored as a *band* | 0.911 | 0.999 | 0.437 |
| **Combined score** | **0.998** | **0.95** | **0.17** |

It is a weighted vote rather than an AND because each feature alone has real
fundus images that fail it — tightly cropped scans have no dark border to find.

The texture row is why the features were measured rather than assumed. Texture
began as a floor, on the reasoning that a solid fill has none: that scored
**AUROC 0.46 — it was voting for the wrong class.** Retinas are smooth and
photographs are busy, so "more texture" is evidence *against* a fundus image,
and σ alone is anti-correlated at 0.223. Scoring it as a band (reject too flat
*and* too busy) instead of a floor took it to 0.911 and cut non-fundus
admissions from 10.4% to 1.7% at an unchanged 97.7% fundus admission.

**Measured on 460 held-out images: 97.7% of fundus images admitted, 1.7% of
non-fundus images admitted.** The threshold is chosen at a target *true
positive* rate, not at best accuracy: rejecting a real scan leaves the clinician
unable to use the tool at all, whereas an admitted non-fundus image still gets
the low-confidence warning.

**Tier 2 — energy score, currently disabled.** `-logsumexp(logits)` (Liu et al.,
NeurIPS 2020) rides free on the forward pass. It separates the two classes well
in isolation (**AUROC 0.930**, better than max-logit at 0.918 or entropy at
0.911), but it is shipped **off**, because tier 1 turned out strong enough to
spend the entire 2% false-rejection budget: at a threshold that rejects no
additional fundus image, it catches none of the four non-fundus images tier 1
admits. `gate.py` enforces that empirically — a tier that catches nothing is
disabled rather than shipped on the strength of its AUROC. The code path is
live and re-enables itself if a future model changes the picture.

Rejections are hard: no image is stored and no scan row is written, so nothing
reaches history or analytics. The user sees an Indonesian message; the failing
feature stays in the server log, because it means nothing to a clinician and
publishing it would explain how to get past the gate.

```bash
cd model
./.venv/bin/python gate.py                 # recalibrate, writes gate.json
./.venv/bin/python gate.py --no-write      # report only
```

### Turning it off

An administrator can switch the gate off at **`/admin/settings`**, with no
restart and no redeploy. That exists because the 2.3% false-rejection rate stops
being a statistic the moment a clinic's own camera trips it: the people affected
need a way out before anyone can recalibrate.

The preference is stored in the `app_setting` table and reapplied at startup;
`gate.json`'s own `enabled` field is only the default for a database where no
administrator has ever expressed a preference. Thresholds are deliberately
**not** editable from the UI — they are measured evidence from `gate.py`, and
there is no way to eyeball a better number for them.

While the gate is off, every page carries a warning banner, because a clinician
reading a result needs to know that any photo could have produced it — not just
the administrator who turned it off. The gate cannot be enabled at all when
`gate.json` is missing: there would be no thresholds to apply, so the toggle
renders disabled rather than pretending to work.

Images are cached under `model/.cache/`, so a re-run after a threshold change
costs seconds. Delete `model/gate.json` to disable the gate entirely; the server
logs a warning and grades every decodable image, which is also what an older
deployment does after a binary upgrade.

---

## Setup

### 1. Prerequisites

```bash
brew install onnxruntime sqlc mysql   # or mariadb
```

### 2. Export the model (once)

```bash
cd model
python3 -m venv .venv
./.venv/bin/pip install torch torchvision transformers onnx onnxscript onnxruntime pillow numpy certifi
./.venv/bin/python export.py     # writes vit_dr_int8.onnx, labels.json, preprocess.json
./.venv/bin/python eval.py       # verifies ordering + int8 vs fp32
./.venv/bin/python explain.py --out /tmp/overlays   # optional: eyeball rollout vs gradient maps
```

### 3. Database

```bash
mysql -u root -e "CREATE DATABASE drml CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;"
mysql -u root drml < internal/database/migration/schema.sql
```

### 4. Configure and run

```bash
cp .env.example .env    # then edit DB_* and JWT_SECRET
go run ./cmd/seed -username admin -password 'ChangeMe123' -email admin@example.com
go run ./cmd/server
```

Open http://localhost:8080.

---

## Frontend

Server-rendered `html/template` with no JavaScript framework: no HTMX, no
Alpine, no charting library. The drawer, the heatmap overlay toggle and the
analytics charts are CSS; the only page that ships script is the upload
drop-zone (`static/js/upload.js`, ~70 lines of vanilla JS). Lucide is the sole
runtime dependency, vendored under `static/cdn/`.

Styling is Tailwind v4 compiled **ahead of time** by the standalone CLI — no
Node, no npm. The component vocabulary (`btn`, `card`, `pill`, `table`,
`field`…) is defined once in `@layer components`:

```bash
make css          # static/css/app.src.css -> static/css/app.css
make css-watch    # recompile while editing templates
```

The CLI is downloaded on demand into `bin/` on first use. **`static/css/app.css`
is committed**, so a fresh clone builds and runs without it — but any change to
a template's classes needs a recompile before those classes exist. `make build`
and `make run` do this for you; a bare `go run ./cmd/server` does not.

Charts are computed in Go (`analyticsSeries` in
`internal/app/web/handler/analytics.go`) and rendered as sized `<span>`s, so
there is no client-side data fetch and no chart JSON endpoint.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | localhost:3306, `drml` | required in production |
| `JWT_SECRET` | dev placeholder | **must** be changed in production |
| `MODEL_PATH` | `model/vit_dr_int8.onnx` | set to `model/vit_dr.onnx` for fp32 |
| `MODEL_HEATMAP_ENABLED` | `true` | saliency overlay on the scan page |
| `MODEL_EXPLAIN_METHOD` | `auto` | `auto` picks gradient relevance when the graph has `attn_grads`, else rollout; `rollout` forces the fallback; `grad-relevance` refuses to start without gradients. Forcing rollout does not skip the backward pass — ORT runs the whole graph regardless of which outputs are fetched |
| `MODEL_GATE_PATH` | `model/gate.json` | non-fundus gate thresholds; absent ⇒ gate off. On/off is overridden at runtime by `/admin/settings` |
| `ORT_LIB_PATH` | auto-detected | onnxruntime shared library |
| `ORT_INTRA_OP_THREADS` | `2` | matches 2 vCPU |
| `ORT_MAX_QUEUE_DEPTH` | `4` | requests beyond this get `503` |
| `STORAGE_DRIVER` | `local` | or `r2` |
| `STORAGE_KEEP_SOURCE_PDF` | `false` | keep the uploaded report too; ~15 MB per screening against ~1 MB for the imagery |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET` | — | required when driver is `r2` |
| `SMTP_HOST` / `SMTP_PORT` | — / `587` | outgoing mail for password-reset links. Unset ⇒ forgot-password is off in production and messages are logged in development. `465` is implicit TLS, any other port STARTTLS |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | — | only ever sent over TLS; a server without STARTTLS is refused |
| `SMTP_FROM` / `SMTP_FROM_NAME` | — / `DRML` | sender; `SMTP_FROM` is required when `SMTP_HOST` is set. Links are built from `APP_URL` |

Scan rows store an opaque storage **key**, never a URL, so moving to R2 is a
config change rather than a data migration. R2 objects stay private and are
served through short-lived presigned URLs.

---

## Roles

| Role | Sees | Can |
|---|---|---|
| User (`role=1`) | only their own scans | upload, view, delete own |
| Administrator (`role=2`) | all scans | + manage users |

Scoping is enforced **in SQL** — every scan query takes `(all_scans, created_by)`
— so a clinician cannot receive another clinician's rows even if a handler is
wrong. Requesting another user's scan returns `404`, not `403`, so the response
does not confirm the row exists.

Anyone can also create their own account at **`/register`**, once an
administrator opens registration at **`/admin/settings`**. It is closed by
default, and `/register` returns `404` while it is. A self-registered account is
always a User, is active immediately (the registrant is signed straight
in), and has a NULL `created_by`. Sign-ups are rate-limited to 5 per hour per IP.

Every account has an email address, unique case-insensitively, required at
`/register`, in the administrator's user form and by `seed -email`. It is where
**`/forgot-password`** sends a reset link, valid for 60 minutes, when the address
belongs to an unsuspended account. The reply is the same either way, and the
lookup and send run after the response is written, so neither the page nor its
timing reveals which addresses are registered. The link is a stateless HMAC
token bound to the account's current password hash, so it works once: setting a
new password by any route invalidates every outstanding link. Requests are
limited to 5 per hour per IP. Without `SMTP_HOST` the feature is off in
production and its pages return `404`.

---

## Deployment

### Docker (quickest)

```bash
cd deploy
DB_PASSWORD=… DB_ROOT_PASSWORD=… JWT_SECRET=… docker compose up -d
```

The image bakes in both the ONNX model and `libonnxruntime.so`, so the VPS
never downloads either at boot. `deploy/my.cnf` holds MySQL to ~450 MB
(`performance_schema=OFF` is the single biggest win).

Add 2 GB of swap as a safety net for session init.

---

### VPS: systemd + nginx

The non-Docker path. Written for Debian 12 / Ubuntu 24.04 on the 2 vCPU / 2 GB
target, with the binary compiled on the box — cgo links against the machine's
own `libonnxruntime`, so there is nothing to cross-compile.

Assumed already in place: sudo, a DNS `A` record pointing at the VPS (certbot
needs it to issue), and a reachable MySQL or MariaDB server.

#### 1. Swap

Two things need it: ONNX session init spikes, and the Go compiler, which wants
~1 GB and will otherwise be OOM-killed mid-build.

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 2. System packages

```bash
sudo apt update
sudo apt install -y build-essential curl git ca-certificates \
                    nginx default-mysql-client certbot python3-certbot-nginx
```

`build-essential` is not optional — cgo needs a C compiler to link the ONNX
Runtime C API. certbot is the only Python that ends up on the server;
`model/*.py` is calibration tooling that runs on a workstation and never ships.

#### 3. Go toolchain

Distro packages lag `go 1.25.1` (see `go.mod`), so take the tarball:

```bash
GO_VERSION=1.25.1
curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tgz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf /tmp/go.tgz
echo 'export PATH=$PATH:/usr/local/go/bin' | sudo tee /etc/profile.d/go.sh
source /etc/profile.d/go.sh && go version
```

Use `linux-arm64` instead on an ARM VPS.

#### 4. onnxruntime shared library

Same version and layout as the Dockerfile's `ort` stage:

```bash
ORT_VERSION=1.20.1
ORT_ARCH=x64        # aarch64 on ARM
curl -fsSL -o /tmp/ort.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz"
sudo mkdir -p /opt/ort
sudo tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1
sudo cp /opt/ort/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
```

Leave `ORT_LIB_PATH` unset afterwards — `defaultORTLibPath` probes
`/usr/local/lib/libonnxruntime.so` and will find it.

#### 5. Service user and directories

Matches `WorkingDirectory` and `ReadWritePaths` in `deploy/drml.service`:

```bash
sudo useradd --system --home-dir /opt/drml --shell /usr/sbin/nologin drml
sudo mkdir -p /opt/drml/{bin,model,static/uploads}
sudo chown -R drml:drml /opt/drml
```

#### 6. Source and model artifacts

```bash
sudo git clone <your-repo-url> /opt/drml/src
```

> **Two files are not in the clone.** `model/vit_dr_int8.onnx` is gitignored
> and `model/gate.json` is untracked, so copy both from the machine that ran
> `export.py` / `gate.py`:
>
> ```bash
> scp model/vit_dr_int8.onnx model/gate.json user@vps:/tmp/
> sudo mv /tmp/vit_dr_int8.onnx /tmp/gate.json /opt/drml/model/
> sudo cp /opt/drml/src/model/{labels.json,preprocess.json} /opt/drml/model/
> sudo chown drml:drml /opt/drml/model/*
> ```
>
> `labels.json` and `preprocess.json` *are* in git. The two failure modes differ:
> without the `.onnx` the server exits at startup; without `gate.json` it boots
> fine with the non-fundus gate **off**, grading every decodable image.

#### 7. Build

```bash
cd /opt/drml/src
sudo -u drml env PATH=$PATH CGO_ENABLED=1 \
  go build -trimpath -ldflags="-s -w" -o /opt/drml/bin/drml ./cmd/server
sudo -u drml env PATH=$PATH CGO_ENABLED=1 \
  go build -trimpath -ldflags="-s -w" -o /opt/drml/bin/seed  ./cmd/seed
```

`go build`, not `make build` — the `build` target depends on `css`, which
downloads the Tailwind CLI. `static/css/app.css` is committed, and the compiled
stylesheet, the JS and every template are baked into the binary by `go:embed`.

#### 8. Environment file

```bash
sudo cp /opt/drml/src/.env.example /opt/drml/.env
sudo chown drml:drml /opt/drml/.env && sudo chmod 600 /opt/drml/.env
openssl rand -hex 32        # -> JWT_SECRET
```

Four settings matter more than the rest:

| Setting | Value | Why |
|---|---|---|
| `ENV` | `production` | Enables the fatal checks on `JWT_SECRET`/`DB_PASSWORD`, and is what sets `Secure` on the auth cookie |
| `SERVER_HOST` | `127.0.0.1` | nginx is the only ingress; never bind `0.0.0.0` |
| `APP_URL` | `https://<domain>` | |
| `JWT_SECRET` | 32 random bytes | |

`MODEL_PATH` and friends stay at their defaults: they are relative to
`WorkingDirectory=/opt/drml`, which is where step 6 put them.

**systemd's `EnvironmentFile` is not a shell.** No `export`, no `$VAR`
expansion, and an unquoted `#` mid-value starts a comment — quote any password
containing one (`DB_PASSWORD='p#ss'`).

#### 9. Database

```bash
mysql -h <db-host> -u root -p <<'SQL'
CREATE DATABASE drml CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
CREATE USER 'drml'@'%' IDENTIFIED BY '<password>';
GRANT ALL PRIVILEGES ON drml.* TO 'drml'@'%';
FLUSH PRIVILEGES;
SQL

mysql -h <db-host> -u drml -p drml < /opt/drml/src/internal/database/migration/schema.sql
cd /opt/drml && sudo -u drml ./bin/seed -username admin -password 'ChangeMe123' -email admin@example.com
```

The `cd` is load-bearing: `config.Load()` reads `.env` relative to the working
directory. `seed` refuses to run if an administrator already exists.

`schema.sql` drops every table, so it is for a **new** database only. An
existing deployment upgrades in place by applying the migration files beside it
instead — they are additive and safe to run against live data:

```bash
mysql -h <db-host> -u drml -p drml \
  < /opt/drml/src/internal/database/migration/2026-09-10-scan-pdf-source.sql
```

That one adds PDF report input (`eye`, `source_kind`, `source_ref`,
`source_key`). Existing
rows become `source_kind = 'image'` with a NULL eye, which is exactly what they
are. See [step 15](#15-updating) for where a migration belongs in a release.

`2026-09-11-user-email-required.sql` makes `email` mandatory and unique. Order
matters here: **backfill, migrate, then deploy**. First give every account
without an email, or sharing one, an address of its own (the file's header has
the queries that find them); then apply the file; then deploy the new binary.
The new binary must not run before the backfill — it reads `email` as a
non-nullable string, so one NULL breaks that user's login and the admin user
list.

There is no migration runner and no schema-version table: re-applying a
migration fails with `Duplicate column name` rather than passing silently, so a
double-apply is loud instead of ambiguous.

If this box also runs the database, copy `deploy/my.cnf` into
`/etc/mysql/conf.d/` first — it holds MySQL to ~450 MB, which is what makes the
2 GB budget work.

#### 10. systemd unit

```bash
sudo cp /opt/drml/src/deploy/drml.service /etc/systemd/system/
# Database is managed elsewhere, so drop mysql.service from the ordering:
sudo sed -i 's/^After=.*/After=network-online.target/' /etc/systemd/system/drml.service

sudo systemctl daemon-reload
sudo systemctl enable --now drml
journalctl -u drml -n 40 --no-pager
```

A good boot logs `Model loaded: … (sha256 …)`, `Non-fundus gate: ENABLED`, and
`Server starting on http://127.0.0.1:8080`. A `class ordering is UNVERIFIED`
warning means `model/eval.py` has not been run for this checkpoint — the
severity labels are not trustworthy until it has.

The unit caps the process at `MemoryMax=1200M` and gives in-flight inference
45 s to finish before a stop is escalated.

#### 11. nginx

`/etc/nginx/sites-available/drml`:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name drml.example.com;

    # The app caps uploads at 12 MiB for an image (MaxImageBytes) and 32 MiB
    # for an IMAGEnet PDF report (MaxPDFBytes) — the reports are large because
    # the fundus rasters inside them are stored uncompressed. Sitting above the
    # larger cap lets the app return its own error page instead of a bare
    # nginx 413.
    client_max_body_size 40m;

    # There is deliberately no `location /static` block:
    #   - CSS/JS/templates are embedded in the binary, so there is no directory
    #     to point at;
    #   - /static/uploads is patient imagery served through RequireAuth. An
    #     alias block there would publish it to anyone with the URL.
    # Proxy everything and let the app decide.
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Overwrite, NOT $proxy_add_x_forwarded_for. The login rate limiter
        # trusts the leftmost X-Forwarded-For value; appending would let a
        # client supply its own header and get a fresh bucket per attempt.
        proxy_set_header X-Forwarded-For   $remote_addr;

        # Matches the server's 120 s WriteTimeout. Inference is 1-2 s, but up
        # to 4 requests may be queued ahead of one on 2 vCPU.
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # No gzip here — the app already runs chi's Compress(5).
}
```

```bash
sudo ln -s /etc/nginx/sites-available/drml /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

#### 12. TLS

```bash
sudo certbot --nginx -d drml.example.com
sudo certbot renew --dry-run
systemctl status certbot.timer
```

certbot rewrites the server block above for `listen 443 ssl` and adds the
HTTP→HTTPS redirect. TLS is not optional here: `ENV=production` marks the auth
cookie `Secure`, so login silently fails over plain HTTP.

#### 13. Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

Port 8080 needs no rule — `SERVER_HOST=127.0.0.1` keeps it off the public
interface entirely.

#### 14. Verify

```bash
curl -fsS https://drml.example.com/healthz     # -> ok
```

Then in a browser: log in as the seeded admin, upload a real fundus image and
confirm a grade, upload a non-fundus image and confirm it is rejected, toggle
the gate at `/admin/settings`, then `sudo systemctl restart drml` and confirm
the toggle survived — that setting lives in the database, not in `gate.json`.

#### 15. Updating

A release with no schema or nginx change is three commands:

```bash
cd /opt/drml/src && sudo -u drml git pull
sudo -u drml env PATH=$PATH CGO_ENABLED=1 \
  go build -trimpath -ldflags="-s -w" -o /opt/drml/bin/drml ./cmd/server
sudo systemctl restart drml
```

**Check `internal/database/migration/` for a migration first.** Skipping one
does not degrade gracefully: sqlc expands `SELECT *` into explicit column
lists, so a column the binary expects and the table lacks fails *every* scan
query — history, detail, dashboard and analytics — with `Unknown column`, not
just the new feature.

Migrations here are written to be additive, and the order follows from that:

```bash
# 1. schema first, while the OLD binary is still serving
sudo -u drml mysql -h <db-host> -u drml -p drml \
  < /opt/drml/src/internal/database/migration/<migration>.sql
```

Running it *before* the restart is deliberate and needs no downtime window. The
running release's generated SQL names its own columns, so added ones are
invisible to it, and added columns are nullable or carry a default, so its
inserts still satisfy every constraint. Running it *after* the restart would
instead leave the new binary querying columns that do not exist yet.

```bash
# 2. nginx, only when a release changes upload limits
sudo nginx -t && sudo systemctl reload nginx
```

PDF input was such a release: reports run 13.4–16.9 MB, so the previous
`client_max_body_size 16m` rejected the larger ones with a bare nginx 413
before the app was ever reached. See [step 11](#11-nginx) for the current value.

Then run the three commands above.

A change to a template's Tailwind classes also needs `make css` run and
`static/css/app.css` committed from a dev machine — the VPS has no Tailwind CLI,
and the binary embeds whatever the stylesheet was at build time. The Docker
build guards only that the file is non-empty, not that it is current, so a
stale commit ships silently and shows up as unstyled markup.

---

## Known limitations

**The heatmap is gradient-weighted attention relevance, not a validated
lesion detector.** It answers "which patches pushed the predicted grade's
logit up?", which is the right question, but nothing here has been scored
against lesion annotations — the repository holds no such ground truth. What
*was* measured on the six `model/testdata/fundus` samples:

- **It is class-specific and image-specific.** Maps from different images
  correlate at 0.08 on average (max 0.64). The previous method, attention
  rollout, correlated at 0.82 between two different patients and peaked in the
  top row of the image on five of the six samples; the gradient map's peak
  moves with the image, and on the grade-2 sample with central exudates it
  lands on discrete spots inside the retina rather than on the frame corners.
- **Rollout is still there as the fallback**, for a graph exported with
  `--forward-only`. It is class-agnostic, and on this checkpoint only weakly
  tied to the image content; the rendering mitigations below exist because of
  it and still apply to both methods.
- **int8 quantization moves the map a little.** Relevance from the served int8
  graph agrees with the fp32 graph at cosine ≥ 0.96 over the samples, with the
  same peak cell on five of six. The Go implementation itself matches the
  Python reference to 1e-5 on the fp32 graph (`TestRelevanceParity`).
- **Resolution is 14×14 patches.** A 16-pixel-square cell at 224 px cannot
  outline a microaneurysm; it can say which region carried the evidence.

Rendering is built not to overstate whichever map it gets:

- **The overlay is confined to the illuminated disc.** `infer.FundusMask` drops
  grid cells on the black surround before normalization, and `RenderHeatmap`
  additionally refuses to paint any pixel below `ROILumaFloor` — both are
  needed, because upsampling a 14x14 grid interpolates a lit boundary cell out
  past the disc edge and the ramp crosses the paint threshold in the black.
  Saliency there is not weak evidence about the retina, it is none.
- **Normalization clips to p2/p98 of the ROI** rather than its min and max, so
  one runaway patch no longer compresses the rest of the map into the bottom of
  the colour ramp.
- **`Saliency.Concentration`** (p98/mean of the raw map over the ROI; uniform
  scores 1.0) measures how much structure a map actually carries, and
  `HeatmapOptions.MinConcentration`/`FullConcentration` ramp the overlay's alpha
  by it, down to rendering no heatmap at all. A map with no positive gradient
  anywhere has concentration 0 and is not drawn.

  **The ramp ships disabled (both 0).** Over the six samples the gradient
  method's concentration runs p0 2.57, p50 5.16, p100 9.90 — far more peaked
  than rollout's 1.50–2.09, so any thresholds measured under rollout are void.
  Re-measure on a real corpus and set them from the printed distribution:

  ```bash
  DRML_SWEEP_DIR=/abs/path/to/scans DRML_SWEEP_OUT=/tmp/overlays \
    go test ./internal/infer/ -run TestSaliencyConcentrationSweep -v
  ```

  The sweep also prints the rank correlation between the rollout and gradient
  maps per image and writes both overlays, which is the quickest way to see
  what the change buys on your own data.

The UI says the map is a coarse aid and not a validated lesion marker. Set
`MODEL_HEATMAP_ENABLED=false` to remove it entirely. Scans graded before the
switch keep their stored rollout overlays: only the rendered JPEG is persisted,
never the grid.

**EXIF orientation is applied at decode time.** `image.Decode` ignores the tag
while browsers honour it, which used to grade a phone-shot fundus sideways and
then draw its heatmap against a base image the browser had rotated —
off by a rotation, and by an aspect ratio. `infer.Upright` now corrects the
decoded pixels before the gate, the model and the overlay see them; the stored
bytes and their SHA-256 remain the untouched original. Scans graded before this
change keep their old heatmap: the 14x14 grid is never persisted, only the
baked JPEG, so re-rendering one would mean re-running inference.

**Grade 3 (Severe) is under-detected** — in the confusion matrix, 15 of 30
severe cases were called Moderate. Both are ≥ 2, so the referable/not-referable
decision is largely unaffected, but the severity number should not be relied on
in isolation.

**PDF reports are read without a rasterizer.** IMAGEnet exports its screening
report through Chrome's print-to-PDF, which embeds the fundus photographs as
`FlateDecode` RGB — plain zlib-compressed pixels, already black outside the
disc. `internal/pdfdoc` inflates them directly, so there is no MuPDF, no
`pdftoppm`, no cgo beyond ONNX Runtime and nothing new in the runtime image,
and the photographs come out at their original framing instead of re-rendered
at print DPI.

The rasters are 40–50 MB each once inflated, which does not fit the memory
budget, so they are never materialised. Extraction takes **two streaming
passes** over the same stream: the first inflates it row by row and box-averages
it into a coarse 1024 px raster, which is enough to locate the discs; the second
inflates it again and copies just those windows out at native resolution. Both
eyes of a shared raster are filled during that one second pass, so a two-eye
report costs two inflates rather than three.

The crop keeps full resolution **because the report is not kept** — see below.
That costs memory: a native 2368×2360 crop is ~22 MB, both eyes are live at
once, and `RenderHeatmap` allocates a full-size RGBA *and* a full-size Gray
beside the image it is grading. The arithmetic suggests ~72 MB; **measured, a
two-eye report adds 38 MB** over the warm 517 MB steady state, peaking at
556 MB, because the first eye's crop is collected before the second is graded.
Comfortable under `MemoryMax=1200M`, and a deliberate trade rather than an
oversight. `TestExtractStreamsLargeRasters` fails if anyone replaces the
streaming decode with a whole-image inflate, or quietly reverts the crop to a
decimated one.

**Laterality comes from the report text, never from position.** A report
carrying a single eye centres it on the page whether it is the left or the
right one — two such reports are byte-identical in layout and differ only in
their `OD(R)`/`OS(L)` label — so the label is parsed through the font's
ToUnicode CMap, with the graphics state tracked so the labels resolve to
absolute page positions. When labels and images cannot be matched one-for-one
the upload is refused: a scan filed against the wrong eye is worse than one the
clinician has to upload by hand.

A two-eye report produces **two scan rows**, one per eye, linked by a shared
`source_ref` rather than an exam table. That is what lets the detail page offer
the other eye without a new join. `source_ref` is an opaque token, not a
storage key, so the link holds whether or not the report itself was kept.

**The source report is discarded by default.** Keeping it costs roughly 15x the
storage: a screening is ~1 MB of imagery, and the report adds ~15 MB on top, so
100 reports would be ~1.6 GB rather than ~100 MB on a VPS disk that also carries
MySQL and a 171 MB model. Set `STORAGE_KEEP_SOURCE_PDF=true` for deployments
that want the original document — the letterhead, the capture dates — and it is
then stored once, referenced by both eyes through `source_key`, and deleted with
the last scan referencing it.

These two decisions are linked: because the report is normally gone, the
extracted crop is the only record of the image, which is why it is kept at the
resolution the report stored it in rather than being decimated. Discarding the
report saves ~15 MB per screening; the full-resolution crop spends ~0.8 MB of
it back.

**No batch upload** — dropped by request; uploads are synchronous, one at a
time. A PDF is the one exception, and only because it is one document for one
patient: its eyes are still screened sequentially through the single inference
slot, so a two-eye report takes roughly twice as long as an image — measured at
1.4–1.7 s end to end, including the two streaming passes over the document.

**A password reset does not end existing sessions.** Sessions are stateless
JWTs checked only against their signature, so one issued before a reset stays
valid until it expires (`JWT_EXPIRATION_HOURS`, 24 h by default); suspension
has the same gap. Closing it needs a per-request lookup of a
`password_changed_at` or session version, which is a separate change.

---

## Testing

```bash
go test ./...              # includes Go↔Python preprocessing parity
cd model && ./.venv/bin/python eval.py
cd model && ./.venv/bin/python gate.py --no-write   # re-measure the gate
```

Notable tests:

- `TestPreprocessParity` — Go resize/normalize matches PIL to within **1.00 LSB**
  (the 8-bit rounding floor); mean drift 0.24 LSB.
- `TestLogitsParity` — full Go path vs PyTorch probabilities, max diff 0.0074,
  same argmax.
- `TestPredictSerializes` — proves the capacity-1 semaphore admits one forward
  pass at a time.
- `TestRelevanceParity` — Go's `GradRelevance` on the graph's own tensors vs
  the Python reference in `golden.json`, on a noise fixture and a real fundus:
  1e-5 on the fp32 graph (`DRML_TEST_MODEL=vit_dr.onnx`), correlation ≥ 0.96
  on the served int8 graph.
- `TestGradRelevanceIsClassSpecificNotAttentionDriven` — a heavily attended
  patch with zero gradient scores nothing while a lightly attended one with a
  positive gradient carries the map; rollout does the opposite on the same
  tensors. This is the property the method was adopted for.
- `TestExplainMethodSelection` — the operator override to rollout works on a
  gradient graph, and `auto` picks gradients when they are there.
- `TestAttentionRolloutUniformIsFlat` — a flat attention map must render as *no
  signal*; without a relative epsilon, min-max normalization amplifies float
  noise into a vivid fictitious heatmap.
- `TestLocalRejectsTraversal` — storage keys come from database rows, so escape
  attempts must not reach the filesystem.
- `TestGateOnRealImages` — runs the Go gate over real fundus and non-fundus
  images cached by `gate.py`. Synthetic shapes prove the arithmetic; only real
  retinas prove the thresholds, and the calibration runs in Python while the
  decision runs in Go.
- `TestCheckImageRejectsGoldenFixture` — `model/testdata/fixture.png` is seeded
  random noise from `export.py`, not a retina. The test exists because it is an
  easy fixture to mistake for one.
- `TestEngineToleratesMissingGate` — a binary deployed before its calibration
  run must serve with the gate off, not refuse to start.
- `TestSetGateEnabled` / `TestGateToggleIsRaceFree` — the admin toggle must
  reach the request path with no restart, and the flag is read by every upload
  while an administrator may flip it from another goroutine, so it is an
  `atomic.Bool` and the test runs under `-race`.
- `TestResetTokenDiesWithPasswordChange` — a reset link is single-use without a
  table because its MAC covers the password hash; the test pins that down, and
  its neighbours cover expiry, an email change and every tampered field.
- `TestBuildMessageRejectsHeaderInjection` — a CR/LF in the recipient or subject
  must not become an extra header (`Bcc:`) in outgoing mail.

---

## Layout

```
cmd/{server,seed}          entrypoints
internal/
  app/scan                 screening workflow (inference → storage → DB)
  app/web                  routes + handlers + templates
  infer                    ONNX session, preprocessing, relevance + rollout, overlay
  storage                  blob interface: local | Cloudflare R2
  database                 schema, sqlc queries, store
  shared                   config, middleware, auth, pagination
template/                  layouts (base, guest) + pages + partials
static/css/app.src.css     design tokens + components (Tailwind source)
static/css/app.css         compiled output, committed
model/                     export.py, eval.py, explain.py, ONNX artifacts
deploy/                    Dockerfile, compose, my.cnf, systemd unit
```
