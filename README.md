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
fundus image ──► Go: decode → resize 224² → normalize
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

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | localhost:3306, `drml` | required in production |
| `JWT_SECRET` | dev placeholder | **must** be changed in production |
| `MODEL_PATH` | `model/vit_dr_int8.onnx` | set to `model/vit_dr.onnx` for fp32 |
| `MODEL_HEATMAP_ENABLED` | `true` | see [Known limitations](#known-limitations) |
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

```bash
cd deploy
DB_PASSWORD=… DB_ROOT_PASSWORD=… JWT_SECRET=… docker compose up -d
```

The image bakes in both the ONNX model and `libonnxruntime.so`, so the VPS
never downloads either at boot. `deploy/my.cnf` holds MySQL to ~450 MB
(`performance_schema=OFF` is the single biggest win). `deploy/drml.service` is
the non-Docker alternative.

Add 2 GB of swap as a safety net for session init.

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

The UI labels it accurately ("not a lesion marker"). Set
`MODEL_HEATMAP_ENABLED=false` to remove it entirely. A faithful alternative is
occlusion sensitivity, but at ~196 forward passes per image it is far too slow
for synchronous inference on 2 vCPU.

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
model/                     export.py, eval.py, ONNX artifacts
deploy/                    Dockerfile, compose, my.cnf, systemd unit
```
