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
fundus image ──► Go: decode → non-fundus gate ──► rejected, nothing stored
                   │                    (52 µs, before the inference slot)
                   ├─► resize 224² → normalize
                   │
                   ├─► ONNX Runtime (ViT-B/16, int8)  ──► logits ──► softmax ──► grade 0–4
                   └─► attentions [12,1,12,197,197]   ──► attention rollout ──► 14×14 heatmap
                   │
                   └─► MySQL row + image/heatmap blobs ──► history ──► analytics
```

**Why not Python?** The target is a 2 vCPU / 2 GB VPS. `import torch` alone is
~1 GB RSS before the model loads, which does not fit beside MySQL. ONNX Runtime
with an int8-quantized ViT measures **328 MB** under sustained load.

**Why attention rollout and not Grad-CAM?** Grad-CAM needs a backward pass,
which ONNX Runtime cannot do. Rollout is forward-only. See
[Known limitations](#known-limitations) — it is a weak localizer on this
checkpoint.

---

## Measured behaviour

Verified on this machine (Apple Silicon; a 2 vCPU x86 VPS will be several times
slower per inference, still within budget):

| Metric | Measured |
|---|---|
| RSS, idle with model loaded | 163 MB |
| RSS, after 40 sequential inferences | 328 MB (plateaus, no creep) |
| Inference time | 131 ms avg, 157 ms max |
| End-to-end upload → stored result | ~275 ms |
| Concurrency | serialized to 1; excess sheds with `503` |

Projected VPS budget: app ~330 MB + MySQL (tuned) ~450 MB + OS ~250 MB
≈ **1.0 GB of 2 GB**.

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
./.venv/bin/pip install torch torchvision transformers onnx onnxruntime pillow numpy certifi
./.venv/bin/python export.py     # writes vit_dr_int8.onnx, labels.json, preprocess.json
./.venv/bin/python eval.py       # verifies ordering + int8 vs fp32
```

### 3. Database

```bash
mysql -u root -e "CREATE DATABASE drml CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;"
mysql -u root drml < internal/database/migration/schema.sql
```

### 4. Configure and run

```bash
cp .env.example .env    # then edit DB_* and JWT_SECRET
go run ./cmd/seed -username admin -password 'ChangeMe123'
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
| `MODEL_HEATMAP_ENABLED` | `true` | see [Known limitations](#known-limitations) |
| `MODEL_GATE_PATH` | `model/gate.json` | non-fundus gate thresholds; absent ⇒ gate off. On/off is overridden at runtime by `/admin/settings` |
| `ORT_LIB_PATH` | auto-detected | onnxruntime shared library |
| `ORT_INTRA_OP_THREADS` | `2` | matches 2 vCPU |
| `ORT_MAX_QUEUE_DEPTH` | `4` | requests beyond this get `503` |
| `STORAGE_DRIVER` | `local` | or `r2` |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET` | — | required when driver is `r2` |

Scan rows store an opaque storage **key**, never a URL, so moving to R2 is a
config change rather than a data migration. R2 objects stay private and are
served through short-lived presigned URLs.

---

## Roles

| Role | Sees | Can |
|---|---|---|
| Clinician (`role=1`) | only their own scans | upload, view, delete own |
| Administrator (`role=2`) | all scans | + manage users |

Scoping is enforced **in SQL** — every scan query takes `(all_scans, created_by)`
— so a clinician cannot receive another clinician's rows even if a handler is
wrong. Requesting another user's scan returns `404`, not `403`, so the response
does not confirm the row exists.

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
cd /opt/drml && sudo -u drml ./bin/seed -username admin -password 'ChangeMe123'
```

The `cd` is load-bearing: `config.Load()` reads `.env` relative to the working
directory. `seed` refuses to run if an administrator already exists.

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

    # The app caps uploads at 12 MiB (MaxImageBytes). Sitting above it lets the
    # app return its own error page instead of a bare nginx 413.
    client_max_body_size 16m;

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

```bash
cd /opt/drml/src && sudo -u drml git pull
sudo -u drml env PATH=$PATH CGO_ENABLED=1 \
  go build -trimpath -ldflags="-s -w" -o /opt/drml/bin/drml ./cmd/server
sudo systemctl restart drml
```

A change to a template's Tailwind classes also needs `make css` run and
`static/css/app.css` committed from a dev machine — the VPS has no Tailwind CLI,
and the binary embeds whatever the stylesheet was at build time.

---

## Known limitations

**The attention heatmap is a weak localizer.** Measured on this checkpoint:

- The maps *are* input-dependent (random noise yields a different map), but two
  different fundus images correlated at **0.82**.
- Diseased images show a strong top-edge bias; on a grade-2 image with obvious
  central exudates, the peak landed on the top-right edge rather than the
  lesions.
- This survives every standard variant (mean/max head fusion, discard ratio
  0.9), so it is a property of the model, not the implementation — which is
  verified against a Python reference.
- Only 12% of saliency falls on dark border patches, so this is *not* a
  missing-fundus-crop bug.

This cannot be fixed by rendering, so the rendering is instead built not to
overstate it:

- **The overlay is confined to the illuminated disc.** `infer.FundusMask` drops
  grid cells on the black surround before normalization, and `RenderHeatmap`
  additionally refuses to paint any pixel below `ROILumaFloor` — both are
  needed, because upsampling a 14x14 grid interpolates a lit boundary cell out
  past the disc edge and the ramp crosses the paint threshold in the black.
  Saliency there is not weak evidence about the retina, it is none.
- **Normalization clips to p2/p98 of the ROI** rather than its min and max, so
  one runaway border patch no longer compresses the rest of the map into the
  bottom of the colour ramp.
- **`Saliency.Concentration`** (p98/mean of the raw rollout over the ROI;
  uniform attention scores 1.0) measures how much structure a map actually
  carries, and `HeatmapOptions.MinConcentration`/`FullConcentration` ramp the
  overlay's alpha by it, down to rendering no heatmap at all. Without this,
  normalization guarantees a saturated peak on every scan however diffuse the
  attention was.

  **The ramp ships disabled (both 0).** The only corpus available here is 12
  images, over which concentration runs p0 1.50, p50 1.67, p100 2.09 — and it
  does not track pathology on that sample (a grade-0 scan scored the highest
  and another the lowest). Two thresholds fitted to that would be overfitting,
  not calibration. Re-measure on a real corpus and set them from the printed
  distribution:

  ```bash
  DRML_SWEEP_DIR=/abs/path/to/scans DRML_SWEEP_OUT=/tmp/overlays \
    go test ./internal/infer/ -run TestSaliencyConcentrationSweep -v
  ```

The UI says plainly that the map is not a lesion marker and must not be used to
locate a lesion. Set `MODEL_HEATMAP_ENABLED=false` to remove it entirely. A
faithful alternative is occlusion sensitivity, but at ~196 forward passes per
image it is far too slow for synchronous inference on 2 vCPU.

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

**No batch upload** — dropped by request; uploads are synchronous, one at a time.

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

---

## Layout

```
cmd/{server,seed}          entrypoints
internal/
  app/scan                 screening workflow (inference → storage → DB)
  app/web                  routes + handlers + templates
  infer                    ONNX session, preprocessing, rollout, overlay
  storage                  blob interface: local | Cloudflare R2
  database                 schema, sqlc queries, store
  shared                   config, middleware, auth, pagination
template/                  layouts (base, guest) + pages + partials
static/css/app.src.css     design tokens + components (Tailwind source)
static/css/app.css         compiled output, committed
model/                     export.py, eval.py, ONNX artifacts
deploy/                    Dockerfile, compose, my.cnf, systemd unit
```
