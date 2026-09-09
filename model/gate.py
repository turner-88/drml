#!/usr/bin/env python3
"""Calibrate the non-fundus gate that guards the classifier.

The served checkpoint is a closed-set 5-way softmax trained only on fundus
images. It has no "not a retina" class and its probabilities always sum to 1,
so it grades a selfie as confidently as a real scan. Nothing inside the model
can fix that; the rejection has to happen outside it.

This script measures where to put the thresholds for the two tiers implemented
in internal/infer/gate.go:

  Tier 1  image heuristics  — red dominance, circular ROI, texture. Runs on
                              every upload before the inference semaphore.
  Tier 2  energy score      — -logsumexp(logits), free-riding on a forward pass
                              that has already happened.

Both are calibrated against real data rather than guessed: fundus positives
from the same two datasets eval.py uses, non-fundus negatives from food, people
and general-scene datasets. Food is in there deliberately — warm-toned food
photography is the hardest natural negative for a red-dominance feature.

Thresholds are chosen at a target TRUE POSITIVE rate, not at best accuracy.
Rejecting a real fundus image is a much worse error than admitting a borderline
one: the clinician is left unable to use the tool at all, whereas an admitted
non-fundus image still gets a low-confidence warning.

Tier 2 stays disabled unless it earns its place. A dynamically quantized
community fine-tune is not guaranteed to produce logits whose magnitudes
separate in- from out-of-distribution inputs, so the energy threshold is only
written when its measured AUROC clears --energy-min-auroc.

Usage:
    ./.venv/bin/python gate.py [--fundus 120] [--negatives 210]
    ./.venv/bin/python gate.py --no-write        # report only, change nothing
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

# eval.py already solves fetching from the datasets server: retrying JSON GETs,
# reading a whole label column cheaply, stratified sampling and the export-time
# preprocessing. Reuse it rather than growing a second copy that can drift.
_ev = importlib.import_module("eval")

HERE = pathlib.Path(__file__).resolve().parent

# Mirrors gateThumbDim in internal/infer/gate.go.
THUMB = 64

# Non-fundus sources. food101 is the adversarial one: warm-toned, round-plated,
# often on a dark surface, so it stresses red dominance and the ROI vote at the
# same time. The other two supply skin tones and general scenes.
NEGATIVE_SETS = [
    {"name": "ethz/food101", "split": "validation", "note": "warm-toned food; hardest natural negative"},
    {"name": "jxie/flickr8k", "split": "test", "note": "everyday photos, people, skin tones"},
    {"name": "sasha/dog-food", "split": "test", "note": "animals and food, high resolution"},
    {"name": "blanchon/UC_Merced", "split": "train", "note": "aerial imagery; textured, no dark border"},
]

# All four serve images above min_dim, deliberately: a negative set that failed
# on size would say nothing about whether the content features work.

# Defaults must match DefaultGateConfig() in internal/infer/gate.go.
DEFAULTS = {
    "enabled": True,
    "min_dim": 224,
    "max_aspect": 2.0,
    "luma_floor": 0.12,
    "red_weight": 0.45,
    "roi_weight": 0.35,
    "texture_weight": 0.20,
    "score_min": 0.55,
    "energy_enabled": False,
    "energy_max": 0.0,
}


# --------------------------------------------------------------------------
# Feature extraction — a numpy mirror of gate.go's features().
#
# This does NOT need the bit-exactness the preprocessing parity test enforces:
# every feature is a mean or a variance over the whole thumbnail, so the small
# differences between PIL's and x/image/draw's bilinear kernels move them far
# below the threshold margins. What must match is the definitions.
# --------------------------------------------------------------------------


def plateau(x: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    if x <= lo0:
        return 0.0
    if x < lo1:
        return (x - lo0) / (lo1 - lo0)
    if x <= hi1:
        return 1.0
    if x < hi0:
        return (hi0 - x) / (hi0 - hi1)
    return 0.0


def block_mean(luma: np.ndarray, x0: int, y0: int, size: int) -> float:
    return float(luma[y0 : y0 + size, x0 : x0 + size].mean())


def laplacian_var(luma: np.ndarray) -> float:
    lap = (
        4 * luma[1:-1, 1:-1]
        - luma[1:-1, :-2]
        - luma[1:-1, 2:]
        - luma[:-2, 1:-1]
        - luma[2:, 1:-1]
    )
    return float(lap.var())


def features(img: Image.Image, cfg: dict) -> dict:
    """Return the same metric names gate.go's GateReport carries."""
    w, h = img.size
    out = {"width": float(w), "height": float(h)}
    out["short_side"] = float(min(w, h))
    out["aspect"] = max(w, h) / max(min(w, h), 1)

    t = np.asarray(img.convert("RGB").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float64) / 255.0
    r, g, b = t[..., 0], t[..., 1], t[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b

    mask = luma >= cfg["luma_floor"]
    out["roi_frac"] = float(mask.mean())
    if not mask.any():
        out.update(red_score=0.0, roi_score=0.0, texture_score=0.0, score=0.0)
        return out

    mean_r, mean_g, mean_b = float(r[mask].mean()), float(g[mask].mean()), float(b[mask].mean())
    out.update(mean_r=mean_r, mean_g=mean_g, mean_b=mean_b)

    # 1. Red dominance.
    red_ratio = mean_r / max(mean_g, 1e-6)
    out["red_ratio"] = red_ratio
    red_score = plateau(red_ratio, 1.15, 1.45, 5.0, 8.0)
    if mean_r <= mean_b:
        red_score = 0.0
    out["red_score"] = red_score

    # 2. Circular ROI: bright disc, dark corners, as bounded Michelson contrast.
    # A centre/corner ratio divides by the epsilon guard on a true black border
    # and reaches six figures; the contrast form is monotonically related, so
    # the arms are the images of the old ratio arms 1.2 and 3.0.
    corner = (
        block_mean(luma, 0, 0, 12)
        + block_mean(luma, THUMB - 12, 0, 12)
        + block_mean(luma, 0, THUMB - 12, 12)
        + block_mean(luma, THUMB - 12, THUMB - 12, 12)
    ) / 4
    centre = block_mean(luma, (THUMB - 20) // 2, (THUMB - 20) // 2, 20)
    contrast = (centre - corner) / max(centre + corner, 1e-6)
    out.update(corner_luma=corner, centre_luma=centre, centre_contrast=contrast)
    out["roi_score"] = plateau(contrast, 0.0909, 0.5, float("inf"), float("inf"))

    # 3. Texture as a band. A one-sided floor measured AUROC 0.46 — it voted for
    # the wrong class, because retinas are smooth and photographs are busy.
    stddev = float(luma[mask].std())
    out["roi_stddev"] = stddev
    out["texture_score"] = plateau(stddev, 0.015, 0.030, 0.13, 0.22)

    out["laplacian_var"] = laplacian_var(luma)

    total = cfg["red_weight"] + cfg["roi_weight"] + cfg["texture_weight"]
    out["score"] = (
        cfg["red_weight"] * out["red_score"]
        + cfg["roi_weight"] * out["roi_score"]
        + cfg["texture_weight"] * out["texture_score"]
    ) / total
    return out


def hard_reject(f: dict, cfg: dict) -> str | None:
    """The two prefilters gate.go applies before any pixel work."""
    if f["short_side"] < cfg["min_dim"]:
        return "too_small"
    if cfg["max_aspect"] > 0 and f["aspect"] > cfg["max_aspect"]:
        return "aspect"
    return None


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), tie-corrected.

    Hand-rolled because the export venv carries neither sklearn nor scipy, and
    adding either for one statistic is not worth the install.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def energy(logits: np.ndarray) -> np.ndarray:
    """-logsumexp(logits); in-distribution inputs score lower."""
    m = logits.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(logits - m).sum(axis=1)))


def entropy(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


CACHE = HERE / ".cache" / "gate"


def cached_image(key: str, url: str) -> Image.Image | None:
    """Download an image once and keep it on disk.

    The datasets server signs image URLs with a short expiry, and a full
    calibration run takes long enough that URLs fetched at the start are dead by
    the end. Caching means a re-run costs nothing and a partial failure can be
    resumed rather than restarted.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.png"
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001 - a truncated cache entry is just a miss
            path.unlink(missing_ok=True)
    try:
        img = _ev.load_image(url)
    except Exception as err:  # noqa: BLE001
        print(f"    skip {key}: {str(err)[:60]}")
        return None
    img.save(path)
    return img


def rows_page(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    qs = _ev.urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    return _ev.http_json(f"{_ev.ROWS_API}?{qs}").get("rows", [])


def dataset_config(dataset: str) -> str:
    """Read the config name rather than assuming 'default' — not every dataset
    uses it (cifar10's is 'plain_text')."""
    try:
        qs = _ev.urllib.parse.urlencode({"dataset": dataset})
        payload = _ev.http_json(f"{_ev.SIZE_API}?{qs}")
        splits = payload["size"].get("splits", [])
        if splits and splits[0].get("config"):
            return str(splits[0]["config"])
    except Exception:  # noqa: BLE001
        pass
    return "default"


def image_cell(row: dict) -> dict | None:
    for key in ("image", "img"):
        cell = row.get(key)
        if isinstance(cell, dict) and "src" in cell:
            return cell
    return None


def spread_indices(total: int, want: int) -> list[int]:
    """Evenly spaced row indices across the whole split.

    Both fundus datasets are stored sorted by class, so reading from offset 0
    returns a single grade — the mistake eval.py documents at length. Spreading
    uniformly hits every grade in the dataset's own proportions, which is all
    this script needs: the gate decides fundus vs not-fundus, and never looks at
    severity. Skipping the label scan that eval.py needs saves ~27 slow requests
    per dataset.
    """
    if want >= total:
        return list(range(total))
    step = total / want
    return sorted({int(i * step) for i in range(want)})


def fetch_rows(dataset: str, split: str, want_idx: list[int], key_prefix: str) -> list[Image.Image]:
    """Download the images at the given row indices, one page request per page.

    Images are fetched immediately after their page is returned: the datasets
    server signs image URLs with a short expiry, and a URL collected at the
    start of a long run is dead by the end of it.
    """
    config = dataset_config(dataset)
    wanted = set(want_idx)
    imgs: list[Image.Image] = []
    for page_start in sorted({(i // 100) * 100 for i in wanted}):
        # Skip the request entirely when this page's rows are already on disk;
        # it makes a re-run after a threshold change cost seconds, not an hour.
        page_idx = [i for i in wanted if page_start <= i < page_start + 100]
        cached = [(i, CACHE / f"{key_prefix}-{i}.png") for i in page_idx]
        if all(path.exists() for _, path in cached):
            for i, path in cached:
                try:
                    imgs.append(Image.open(path).convert("RGB"))
                except Exception:  # noqa: BLE001
                    path.unlink(missing_ok=True)
            print(f"    cached {len(imgs)}/{len(wanted)}", end="\r", flush=True)
            continue
        try:
            rows = rows_page(dataset, config, split, page_start, 100)
        except Exception as err:  # noqa: BLE001
            print(f"    page {page_start} failed, skipping: {str(err)[:60]}")
            continue
        for k, r in enumerate(rows):
            idx = page_start + k
            if idx not in wanted:
                continue
            cell = image_cell(r["row"])
            if cell is None:
                continue
            img = cached_image(f"{key_prefix}-{idx}", cell["src"])
            if img is not None:
                imgs.append(img)
                print(f"    fetched {len(imgs)}/{len(wanted)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return imgs


def fetch_fundus(want_each: int, rng) -> list[Image.Image]:
    """Fundus positives from the two datasets eval.py already verifies."""
    imgs: list[Image.Image] = []
    for spec in _ev.DATASETS:
        print(f"  {spec['name']} ({spec['note']})")
        try:
            total = _ev.num_rows(spec["name"], spec["split"])
            idx = spread_indices(total, want_each)
        except Exception as err:  # noqa: BLE001
            print(f"    unavailable, skipping: {str(err)[:80]}")
            continue
        got = fetch_rows(spec["name"], spec["split"], idx, spec["key"])
        imgs.extend(got)
        print(f"    {len(got)} images")
    return imgs


def fetch_negatives(count: int) -> list[Image.Image]:
    """Non-fundus images, spread across the negative sources."""
    per_set = max(1, count // len(NEGATIVE_SETS))
    imgs: list[Image.Image] = []
    for spec in NEGATIVE_SETS:
        print(f"  {spec['name']} ({spec['note']})")
        try:
            config = dataset_config(spec["name"])
        except Exception as err:  # noqa: BLE001
            print(f"    unavailable, skipping: {str(err)[:80]}")
            continue
        got, offset = 0, 0
        while got < per_set:
            try:
                rows = rows_page(spec["name"], config, spec["split"], offset, 50)
            except Exception as err:  # noqa: BLE001
                print(f"    unavailable, skipping: {str(err)[:80]}")
                break
            if not rows:
                break
            for k, r in enumerate(rows):
                if got >= per_set:
                    break
                cell = image_cell(r["row"])
                if cell is None:
                    continue
                key = spec["name"].replace("/", "_")
                img = cached_image(f"{key}-{offset + k}", cell["src"])
                if img is None:
                    continue
                imgs.append(img)
                got += 1
                print(f"    fetched {got}/{per_set}", end="\r", flush=True)
            offset += len(rows)
        print(" " * 40, end="\r")
        print(f"    {got} images")
    return imgs


def cache_samples(imgs: list[Image.Image], out: pathlib.Path, n: int) -> None:
    """Save a few real images so the Go tests have ground truth to assert on.

    Without this the Go side can only test synthetic shapes: the repo's only
    image fixture is seeded random noise from export.py, not a retina.
    """
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    for i, im in enumerate(imgs[:n]):
        # Downscaled before committing: these are test fixtures, not data. 384
        # keeps them clear of the min_dim prefilter while staying small enough
        # to live in the repo — the prefilters are covered by gate.py's report
        # on the true sizes, so the fixtures only need to carry the content.
        im = im.copy()
        im.thumbnail((384, 384), Image.LANCZOS)
        im.save(out / f"{i:02d}.png")
    print(f"  cached {min(n, len(imgs))} samples -> {out.relative_to(HERE.parent)}")


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fundus", type=int, default=120, help="fundus images per dataset")
    ap.add_argument("--negatives", type=int, default=210, help="total non-fundus images")
    ap.add_argument("--target-tpr", type=float, default=0.98, help="fraction of real fundus images that must pass")
    ap.add_argument("--energy-min-auroc", type=float, default=0.90, help="tier 2 stays off below this")
    ap.add_argument("--model", default="vit_dr_int8.onnx")
    ap.add_argument("--no-write", action="store_true", help="report only; do not write gate.json or fixtures")
    ap.add_argument("--cache-samples", type=int, default=6, help="real images to save per class for the Go tests")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    rng = np.random.default_rng(20240908)

    print("Fetching fundus positives ...")
    pos_imgs = fetch_fundus(args.fundus, rng)
    print(f"  {len(pos_imgs)} fundus images")
    print("Fetching non-fundus negatives ...")
    neg_imgs = fetch_negatives(args.negatives)
    print(f"  {len(neg_imgs)} non-fundus images")

    if not pos_imgs or not neg_imgs:
        print("ERROR: need both classes to calibrate", file=sys.stderr)
        return 1

    print("\nComputing tier 1 features ...")
    pos_f = [features(im, cfg) for im in pos_imgs]
    neg_f = [features(im, cfg) for im in neg_imgs]

    # ---- tier 1 ----
    print("\n== Tier 1: image heuristics ==")
    print(f"{'feature':<22} {'AUROC':>7}  {'fundus mean':>12}  {'non-fundus mean':>16}")
    for name in ("red_ratio", "red_score", "centre_contrast", "roi_score", "roi_stddev", "texture_score", "score"):
        p = np.array([f.get(name, 0.0) for f in pos_f], dtype=np.float64)
        n = np.array([f.get(name, 0.0) for f in neg_f], dtype=np.float64)
        print(f"{name:<22} {auroc(p, n):>7.4f}  {p.mean():>12.4f}  {n.mean():>16.4f}")

    # An AUROC near 0 means a feature is anti-correlated — it is voting for the
    # wrong class and must be redefined, not reweighted. The percentiles below
    # are what the plateau arms in gate.go should be placed from.
    print(f"\n{'raw feature':<22} {'class':<11}" + "".join(f"{q:>9}" for q in ("p1", "p5", "p50", "p95", "p99")))
    for name in ("red_ratio", "centre_contrast", "roi_stddev", "laplacian_var", "roi_frac"):
        for label, fs in (("fundus", pos_f), ("non-fundus", neg_f)):
            v = np.array([f.get(name, 0.0) for f in fs], dtype=np.float64)
            qs = np.quantile(v, [0.01, 0.05, 0.50, 0.95, 0.99])
            print(f"{name:<22} {label:<11}" + "".join(f"{q:>9.4f}" for q in qs))

    pos_scores = np.array([f["score"] for f in pos_f])
    neg_scores = np.array([f["score"] for f in neg_f])

    # The prefilters are reported separately: they are absolute rules, so their
    # cost is a count of real fundus images wrongly excluded, not an AUROC.
    pos_hard = [hard_reject(f, cfg) for f in pos_f]
    neg_hard = [hard_reject(f, cfg) for f in neg_f]
    print(f"\nprefilters (min_dim={cfg['min_dim']}, max_aspect={cfg['max_aspect']}):")
    print(f"  fundus rejected     {sum(h is not None for h in pos_hard):>4}/{len(pos_f)}"
          f"   too_small={sum(h == 'too_small' for h in pos_hard)}"
          f" aspect={sum(h == 'aspect' for h in pos_hard)}")
    print(f"  non-fundus rejected {sum(h is not None for h in neg_hard):>4}/{len(neg_f)}"
          f"   too_small={sum(h == 'too_small' for h in neg_hard)}"
          f" aspect={sum(h == 'aspect' for h in neg_hard)}")

    # Threshold on the images the prefilters let through; it never sees the rest.
    surviving_pos = pos_scores[[h is None for h in pos_hard]]
    score_min = float(np.quantile(surviving_pos, 1 - args.target_tpr)) if len(surviving_pos) else cfg["score_min"]
    cfg["score_min"] = round(score_min, 4)

    passed_pos = sum(1 for f, h in zip(pos_f, pos_hard) if h is None and f["score"] >= score_min)
    passed_neg = sum(1 for f, h in zip(neg_f, neg_hard) if h is None and f["score"] >= score_min)
    print(f"\nscore_min = {score_min:.4f}  (chosen at {args.target_tpr:.0%} TPR on surviving fundus images)")
    print(f"  full gate: fundus admitted {passed_pos}/{len(pos_f)} ({passed_pos / len(pos_f):.1%})"
          f" — this is the false-rejection cost")
    print(f"  full gate: non-fundus admitted {passed_neg}/{len(neg_f)} ({passed_neg / len(neg_f):.1%})"
          f" — these still reach the classifier")

    # ---- tier 2 ----
    print("\n== Tier 2: energy score ==")
    model_path = HERE / args.model
    pre = json.loads((HERE / "preprocess.json").read_text())
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    logits_idx = out_names.index("logits") if "logits" in out_names else 0

    def run(imgs: list[Image.Image]) -> np.ndarray:
        acc = []
        for k, im in enumerate(imgs):
            acc.append(sess.run([out_names[logits_idx]], {"pixel_values": _ev.preprocess(im, pre)})[0][0])
            print(f"    {k + 1}/{len(imgs)}", end="\r", flush=True)
        print(" " * 30, end="\r")
        return np.asarray(acc, dtype=np.float64)

    print(f"  running {model_path.name} over both sets ...")
    pos_logits, neg_logits = run(pos_imgs), run(neg_imgs)

    pos_e, neg_e = energy(pos_logits), energy(neg_logits)
    # Higher energy should mean more out-of-distribution, so the positive class
    # for AUROC is the negative set: score = energy, label = "is non-fundus".
    e_auroc = auroc(neg_e, pos_e)
    ml_auroc = auroc(-neg_logits.max(axis=1), -pos_logits.max(axis=1))
    ent_auroc = auroc(entropy(neg_logits), entropy(pos_logits))
    print(f"  energy    AUROC {e_auroc:.4f}   fundus mean {pos_e.mean():+.3f}  non-fundus mean {neg_e.mean():+.3f}")
    print(f"  max-logit AUROC {ml_auroc:.4f}")
    print(f"  entropy   AUROC {ent_auroc:.4f}")

    # Tier 2 only ever sees what tier 1 admitted, and its false rejections
    # compound with tier 1's rather than replacing them. Budgeting both at the
    # target independently would miss it: 0.98 x 0.98 is 0.96. So tier 2 is
    # calibrated on the surviving fundus images, for the share of the budget
    # tier 1 has not already spent.
    pos_survives_t1 = np.array([h is None and f["score"] >= score_min for f, h in zip(pos_f, pos_hard)])
    neg_survives_t1 = np.array([h is None and f["score"] >= score_min for f, h in zip(neg_f, neg_hard)])
    t1_tpr = float(pos_survives_t1.mean())

    if e_auroc < args.energy_min_auroc:
        cfg["energy_enabled"] = False
        cfg["energy_max"] = 0.0
        print(f"  DISABLED: AUROC {e_auroc:.4f} < {args.energy_min_auroc} — the served checkpoint's")
        print("            logits do not separate cleanly enough to threshold on. Tier 1 carries the gate.")
        energy_max = 0.0
    else:
        budget = min(1.0, args.target_tpr / t1_tpr) if t1_tpr > 0 else 1.0
        surviving_e = pos_e[pos_survives_t1]
        energy_max = float(np.quantile(surviving_e, budget)) if len(surviving_e) else 0.0
        if budget >= 1.0 and len(surviving_e):
            # Tier 1 has already spent the whole false-rejection budget, so the
            # quantile lands exactly on the worst fundus image in the sample.
            # Sitting on that maximum would reject any real scan even slightly
            # more unusual than the 220 seen here, which is calibrating to the
            # sample rather than to the population. Extend by a quarter of the
            # observed upper spread.
            spread = float(surviving_e.max() - np.median(surviving_e))
            energy_max += 0.25 * spread
        leaked = int(neg_survives_t1.sum())
        caught = int(((neg_e > energy_max) & neg_survives_t1).sum())
        if caught == 0:
            # AUROC is not sufficient grounds to ship a threshold. Energy
            # separates the two classes well on its own, but tier 1 turned out
            # strong enough to spend the whole false-rejection budget, which
            # leaves tier 2 no room to act: at a threshold that rejects no
            # additional fundus image it rejects no additional non-fundus one
            # either. An enabled cut that catches nothing is not defence in
            # depth, it is an unexercised way to lose a real scan.
            cfg["energy_enabled"] = False
            cfg["energy_max"] = 0.0
            print(f"  DISABLED: energy_max would be {energy_max:.4f}, which catches 0 of the "
                  f"{leaked} non-fundus images tier 1 admitted.")
            print(f"            Energy separates well in isolation (AUROC {e_auroc:.4f}), but tier 1")
            print(f"            already spends the full {1 - args.target_tpr:.0%} budget, so there is no room left.")
        else:
            cfg["energy_enabled"] = True
            cfg["energy_max"] = round(energy_max, 4)
            print(f"  ENABLED: energy_max = {energy_max:.4f} (tier 1 spent {1 - t1_tpr:.1%} of the "
                  f"{1 - args.target_tpr:.0%} budget, tier 2 gets the rest)")
            print(f"  of the {leaked} non-fundus images tier 1 admitted, tier 2 catches {caught}")

    # ---- combined ----
    t2_ok = np.ones(len(pos_f), dtype=bool) if not cfg["energy_enabled"] else (pos_e <= energy_max)
    t2_ok_neg = np.ones(len(neg_f), dtype=bool) if not cfg["energy_enabled"] else (neg_e <= energy_max)
    final_pos = int((pos_survives_t1 & t2_ok).sum())
    final_neg = int((neg_survives_t1 & t2_ok_neg).sum())
    print("\n== Both tiers ==")
    print(f"  fundus admitted     {final_pos}/{len(pos_f)} ({final_pos / len(pos_f):.1%})"
          f"   <- false rejections cost {1 - final_pos / len(pos_f):.1%}")
    print(f"  non-fundus admitted {final_neg}/{len(neg_f)} ({final_neg / len(neg_f):.1%})"
          f"   <- still reach the classifier")

    # ---- write ----
    if args.no_write:
        print("\n--no-write: nothing written")
        return 0

    cfg["calibration"] = {
        "fundus_images": len(pos_imgs),
        "non_fundus_images": len(neg_imgs),
        "fundus_sources": [d["name"] for d in _ev.DATASETS],
        "non_fundus_sources": [d["name"] for d in NEGATIVE_SETS],
        "target_tpr": args.target_tpr,
        "measured_fundus_admitted": round(final_pos / len(pos_f), 4),
        "measured_non_fundus_admitted": round(final_neg / len(neg_f), 4),
        "measured_tier1_only_fundus_admitted": round(passed_pos / len(pos_f), 4),
        "score_auroc": round(auroc(pos_scores, neg_scores), 4),
        "energy_auroc": round(e_auroc, 4),
        "model": args.model,
        "note": "Written by gate.py. Thresholds are set at a target true-positive "
                "rate on real fundus images, not at best accuracy: a false rejection "
                "is worse than a false admission, which the low-confidence warning "
                "already covers.",
    }
    out = HERE / "gate.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(HERE.parent)}")

    if args.cache_samples > 0:
        cache_samples(pos_imgs, HERE / "testdata" / "fundus", args.cache_samples)
        cache_samples(neg_imgs, HERE / "testdata" / "negatives", args.cache_samples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
