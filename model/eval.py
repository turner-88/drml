#!/usr/bin/env python3
"""Verify the exported model before the Go server is allowed to trust it.

Three checks, in order of how badly they can hurt:

1. ONNX parity      — do vit_dr.onnx / vit_dr_int8.onnx agree with PyTorch?
2. Class ordering   — does output index i actually mean DR grade i?
3. int8 vs fp32     — did dynamic quantization cost more than ~2 points?

Check 2 is the one that matters clinically. The checkpoint ships a placeholder
id2label of {"0":"0",...,"4":"4"}, so "index 2 == Moderate" is an assumption
until measured. This script measures it by brute-forcing all 120 permutations
of output index -> severity grade and reporting which fits labelled data best.

Two datasets are used, because each alone is insufficient:

  aptos_test  — all five grades present, but labels are bare integers, so on
                its own it could only confirm a numbering against a numbering.
  eslam       — labels are spelled out ("No DR", "Mild", ...), which pins the
                semantics, but the split only contains grades 0-2.

Agreement between them is the actual evidence: the named dataset fixes what
grades 0-2 mean, and the numeric one extends the check to 3 and 4 while
reproducing APTOS's characteristic class balance.

IMPORTANT: sampling must be stratified. Both datasets are stored sorted by
class, so reading from offset 0 returns a single class. An earlier version of
this script did exactly that and "verified" the ordering against 200 images
that were all Mild.

CAVEAT: APTOS's public labels are its train split, which this checkpoint was
almost certainly fine-tuned on. Accuracy here is optimistic and must not be
quoted as held-out performance. Ordering is unaffected by leakage.

Usage:
    ./.venv/bin/python eval.py [--per-class 30]
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import pathlib
import ssl
import sys
import time
import urllib.parse
import urllib.request

import certifi
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROWS_API = "https://datasets-server.huggingface.co/rows"
SIZE_API = "https://datasets-server.huggingface.co/size"

# Python installs on macOS frequently lack a system trust store, so pin urllib
# to certifi's bundle rather than silently disabling verification.
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Maps a dataset's spelled-out ClassLabel names to canonical severity grades.
NAME_TO_GRADE = {
    "no dr": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "proliferative dr": 4,
}

DATASETS = [
    {
        "key": "aptos_test",
        "name": "sngsfydy/aptos_test",
        "split": "train",
        # Bare integer ClassLabels; assumed to be the APTOS diagnosis column.
        "numeric": True,
        "note": "all 5 grades; numeric labels",
    },
    {
        "key": "eslam",
        "name": "EslamHasan/APTOS2019DiabeticRetinopathy",
        "split": "train",
        # Spelled-out names, mapped via NAME_TO_GRADE.
        "numeric": False,
        "note": "grades 0-2 only; named labels pin the semantics",
    },
]


def http_json(url: str, attempts: int = 4) -> dict:
    """GET with retries — the datasets server returns sporadic 500/502s."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=90, context=SSL_CTX) as resp:
                return json.load(resp)
        except Exception as err:  # noqa: BLE001 - retry any transport/server error
            last = err
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed after {attempts} attempts: {url}\n  last error: {last}")


def num_rows(dataset: str, split: str) -> int:
    payload = http_json(f"{SIZE_API}?{urllib.parse.urlencode({'dataset': dataset})}")
    for s in payload["size"].get("splits", []):
        if s["split"] == split:
            return int(s["num_rows"])
    return int(payload["size"]["dataset"]["num_rows"])


def scan_labels(dataset: str, split: str, total: int) -> tuple[list[int], list[str], list[str]]:
    """Read the whole label column plus each row's image URL.

    The rows endpoint returns image URLs rather than bytes, so scanning every
    label costs a few JSON pages and downloads nothing.
    """
    labels: list[int] = []
    urls: list[str] = []
    names: list[str] = []
    offset = 0
    while offset < total:
        qs = urllib.parse.urlencode(
            {"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": 100}
        )
        payload = http_json(f"{ROWS_API}?{qs}")
        if not names:
            for feat in payload.get("features", []):
                if feat["name"] == "label":
                    names = [str(x) for x in feat["type"]["names"]]
        rows = payload.get("rows", [])
        if not rows:
            break
        for r in rows:
            labels.append(int(r["row"]["label"]))
            urls.append(r["row"]["image"]["src"])
        offset += len(rows)
        print(f"    scanned {offset}/{total} labels", end="\r", flush=True)
    print(" " * 40, end="\r")
    return labels, urls, names


def index_to_grade(names: list[str], numeric: bool) -> dict[int, int]:
    """Map a dataset's ClassLabel index to a canonical severity grade."""
    mapping: dict[int, int] = {}
    for i, nm in enumerate(names):
        key = nm.strip().lower()
        if numeric:
            mapping[i] = int(key)
        else:
            if key not in NAME_TO_GRADE:
                raise ValueError(f"unmapped class name {nm!r}")
            mapping[i] = NAME_TO_GRADE[key]
    return mapping


def stratified_indices(grades: list[int], per_class: int, rng: np.random.Generator) -> list[int]:
    """Pick up to per_class rows for each grade actually present."""
    chosen: list[int] = []
    by_grade: dict[int, list[int]] = {}
    for i, g in enumerate(grades):
        by_grade.setdefault(g, []).append(i)
    for g in sorted(by_grade):
        pool = by_grade[g]
        take = min(per_class, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    return chosen


def load_image(url: str) -> Image.Image:
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90, context=SSL_CTX) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception:
            if i == 2:
                raise
            time.sleep(1.0 * (i + 1))
    raise RuntimeError("unreachable")


def preprocess(img: Image.Image, pre: dict) -> np.ndarray:
    """Mirror the export-time processor exactly."""
    resample = {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR, 3: Image.BICUBIC}.get(
        pre.get("resample_code", 2), Image.BILINEAR
    )
    img = img.resize((pre["width"], pre["height"]), resample)
    arr = np.asarray(img, dtype=np.float32) * pre["rescale_factor"]
    arr = (arr - np.array(pre["image_mean"], dtype=np.float32)) / np.array(pre["image_std"], dtype=np.float32)
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


def quadratic_weighted_kappa(actual: np.ndarray, pred: np.ndarray, n: int = 5) -> float:
    """The APTOS competition metric — being off by 3 grades is penalized far
    more than off by 1, which is what matters on an ordinal severity scale."""
    obs = np.zeros((n, n))
    for a, p in zip(actual, pred):
        obs[a, p] += 1
    w = np.array([[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)])
    exp = np.outer(np.bincount(actual, minlength=n), np.bincount(pred, minlength=n)).astype(float)
    if exp.sum() == 0:
        return 0.0
    exp *= obs.sum() / exp.sum()
    denom = (w * exp).sum()
    return 1.0 - (w * obs).sum() / denom if denom else 0.0


def print_confusion(actual: np.ndarray, pred: np.ndarray, present: list[int]) -> None:
    m = np.zeros((5, 5), dtype=int)
    for a, p in zip(actual, pred):
        m[a, p] += 1
    print("                    predicted")
    print("               " + "".join(f"{i:>6}" for i in range(5)))
    for i in range(5):
        mark = " " if i in present else "-"
        print(f"  true {i}{mark}({GRADE_NAMES[i][:4]:>4}) " + "".join(f"{v:>6}" for v in m[i]))
    print("  (rows marked '-' had no samples)")


def check_onnx_parity() -> bool:
    golden_path = HERE / "testdata" / "golden.json"
    if not golden_path.exists():
        print("  SKIP: no golden fixture; run export.py first")
        return True
    golden = json.loads(golden_path.read_text())
    pv = np.array(golden["pixel_values"], dtype=np.float32).reshape(golden["pixel_values_shape"])
    torch_logits = np.array(golden["logits"], dtype=np.float32)

    ok = True
    for name in ("vit_dr.onnx", "vit_dr_int8.onnx"):
        path = HERE / name
        if not path.exists():
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logits, attns = sess.run(["logits", "attentions"], {"pixel_values": pv})
        diff = float(np.abs(logits.flatten() - torch_logits).max())
        tol = 1.0 if "int8" in name else 1e-3
        if diff > tol:
            ok = False
        print(f"  {'OK ' if diff <= tol else 'FAIL'} {name:20s} max|logit diff| = {diff:.6f} (tol {tol})")
        print(f"       attentions {tuple(attns.shape)} (expect (layers,1,heads,197,197))")
    return ok


def evaluate(sessions: dict, urls: list[str], grades: list[int], pre: dict) -> tuple[np.ndarray, dict]:
    truth: list[int] = []
    preds: dict[str, list[int]] = {k: [] for k in sessions}
    for n, (url, g) in enumerate(zip(urls, grades)):
        try:
            img = load_image(url)
        except Exception:
            continue
        truth.append(g)
        x = preprocess(img, pre)
        for name, sess in sessions.items():
            preds[name].append(int(np.argmax(sess.run(["logits"], {"pixel_values": x})[0])))
        if (n + 1) % 20 == 0:
            print(f"    {n + 1}/{len(urls)} images", end="\r", flush=True)
    print(" " * 40, end="\r")
    return np.array(truth), {k: np.array(v) for k, v in preds.items()}


def ordering_verdict(actual: np.ndarray, pred: np.ndarray) -> tuple[bool, tuple, float, float]:
    """Return (identity_is_best, best_perm, identity_acc, best_acc).

    Only grades present in `actual` can constrain the mapping, so ties are
    resolved in favour of identity to avoid over-claiming from a partial split.
    """
    scored = []
    for perm in itertools.permutations(range(5)):
        mapped = np.array([perm[v] for v in pred])
        scored.append((float((mapped == actual).mean()), perm))
    scored.sort(key=lambda t: (-t[0], t[1] != (0, 1, 2, 3, 4)))
    identity_acc = float((pred == actual).mean())
    best_acc, best_perm = scored[0]
    return best_perm == (0, 1, 2, 3, 4), best_perm, identity_acc, best_acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30, help="images sampled per grade, per dataset")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pre = json.loads((HERE / "preprocess.json").read_text())
    rng = np.random.default_rng(args.seed)

    print("[1/3] ONNX parity vs PyTorch")
    parity_ok = check_onnx_parity()

    sessions = {}
    for name in ("vit_dr.onnx", "vit_dr_int8.onnx"):
        p = HERE / name
        if p.exists():
            sessions[name] = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    if not sessions:
        print("ERROR: no ONNX model found; run export.py first", file=sys.stderr)
        return 2

    print("\n[2/3] Class ordering")
    verdicts: dict[str, bool] = {}
    accuracy: dict[str, dict[str, float]] = {}

    for ds in DATASETS:
        print(f"\n  --- {ds['name']} ({ds['note']}) ---")
        try:
            total = num_rows(ds["name"], ds["split"])
            labels, urls, names = scan_labels(ds["name"], ds["split"], total)
        except Exception as err:
            print(f"  SKIP: {err}", file=sys.stderr)
            continue

        try:
            idx2grade = index_to_grade(names, ds["numeric"])
        except ValueError as err:
            print(f"  SKIP: {err}", file=sys.stderr)
            continue

        grades = [idx2grade[l] for l in labels]
        dist = np.bincount(np.array(grades), minlength=5)
        present = [g for g in range(5) if dist[g] > 0]
        print(f"  {total} rows; ClassLabel names={names}")
        print(f"  index->grade {idx2grade}; full-split grade distribution {dist.tolist()}")

        pick = stratified_indices(grades, args.per_class, rng)
        sel_urls = [urls[i] for i in pick]
        sel_grades = [grades[i] for i in pick]
        print(f"  stratified sample: {len(pick)} images, per-grade "
              f"{np.bincount(np.array(sel_grades), minlength=5).tolist()}")

        actual, preds = evaluate(sessions, sel_urls, sel_grades, pre)
        if len(actual) == 0:
            print("  SKIP: no images could be downloaded", file=sys.stderr)
            continue

        for model_name, pred in preds.items():
            is_identity, best_perm, id_acc, best_acc = ordering_verdict(actual, pred)
            kappa = quadratic_weighted_kappa(actual, pred)
            print(f"\n  [{model_name}] on {len(actual)} images covering grades {present}")
            print(f"    identity : accuracy={id_acc:.4f}  quadratic-weighted kappa={kappa:.4f}")
            print(f"    best perm: accuracy={best_acc:.4f}  perm={best_perm}")
            if is_identity:
                print("    => identity is the best mapping. Ordering supported.")
            elif best_acc - id_acc < 0.05:
                print(f"    => {best_perm} ties identity within 5 points; not evidence against it.")
            else:
                print(f"    => WARNING: {best_perm} beats identity by {best_acc - id_acc:.3f}.")
            print_confusion(actual, pred, present)

            key = f"{ds['key']}:{model_name}"
            verdicts[key] = is_identity or (best_acc - id_acc < 0.05)
            accuracy.setdefault(model_name, {})[ds["key"]] = id_acc

    print("\n[3/3] int8 vs fp32")
    if "vit_dr.onnx" in accuracy and "vit_dr_int8.onnx" in accuracy:
        for key in accuracy["vit_dr.onnx"]:
            a = accuracy["vit_dr.onnx"][key]
            b = accuracy["vit_dr_int8.onnx"].get(key)
            if b is None:
                continue
            print(f"  {key}: fp32={a:.4f} int8={b:.4f} drop={a - b:+.4f}")
            if a - b > 0.02:
                print("  => int8 costs >2 points. Serve fp32: MODEL_PATH=model/vit_dr.onnx")
            else:
                print("  => int8 within tolerance; safe to serve the quantized model.")

    all_ok = bool(verdicts) and all(verdicts.values())
    print("\n" + "=" * 66)
    print("ORDERING VERDICT:", "SUPPORTED" if all_ok else "NOT ESTABLISHED")
    for k, v in verdicts.items():
        print(f"  {k}: {'ok' if v else 'FAILED'}")
    print("=" * 66)
    print("NOTE: APTOS's public labels are its train split, which this checkpoint was")
    print("      almost certainly fine-tuned on. The accuracies above are optimistic;")
    print("      what this run establishes is the class ordering, not held-out skill.")

    labels_path = HERE / "labels.json"
    meta = json.loads(labels_path.read_text())
    meta["ordering_verified"] = all_ok
    meta["ordering_evidence"] = {
        "datasets": {k: bool(v) for k, v in verdicts.items()},
        "per_class_sampled": args.per_class,
        "method": "stratified sample; argmax over all 120 index->grade permutations",
        "caveat": "APTOS public labels are the train split; accuracy is optimistic, ordering is not affected",
    }
    labels_path.write_text(json.dumps(meta, indent=2))
    print(f"\nlabels.json: ordering_verified = {all_ok}")

    return 0 if (parity_ok and all_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
