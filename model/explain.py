#!/usr/bin/env python3
"""Render rollout and gradient-relevance overlays side by side for eyeballing.

    ./.venv/bin/python explain.py [--model vit_dr_int8.onnx] [--images testdata/fundus] --out /tmp/overlays

For each image it writes <name>.png with three panels — the input, attention
rollout, and gradient-weighted relevance — and prints the predicted class, the
rank correlation between the two maps, and where each peaks. Human review only:
there is no lesion ground truth in this repository to score against.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import onnxruntime as ort
from PIL import Image

from export import attention_rollout, chefer_relevance

HERE = pathlib.Path(__file__).resolve().parent


def preprocess(img: Image.Image, pre: dict) -> np.ndarray:
    resample = {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR, 3: Image.BICUBIC, 4: Image.BOX, 5: Image.HAMMING}
    img = img.convert("RGB").resize((pre["width"], pre["height"]), resample[pre["resample_code"]])
    arr = np.asarray(img).astype(np.float32) * pre["rescale_factor"]
    arr = (arr - np.array(pre["image_mean"], np.float32)) / np.array(pre["image_std"], np.float32)
    return arr.transpose(2, 0, 1)[None].copy()


def normalize(grid: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(grid, 2), np.percentile(grid, 98)
    if hi - lo <= 1e-6 * max(abs(hi), abs(lo), 1e-30):
        return np.zeros_like(grid)
    return np.clip((grid - lo) / (hi - lo), 0, 1)


def jet(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def overlay(img: Image.Image, grid: np.ndarray, alpha: float = 0.45, threshold: float = 0.25) -> Image.Image:
    side = int(round(np.sqrt(grid.size)))
    small = Image.fromarray((normalize(grid).reshape(side, side) * 255).astype(np.uint8))
    big = np.asarray(small.resize(img.size, Image.BICUBIC)).astype(np.float32) / 255
    base = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    luma = base @ np.array([0.299, 0.587, 0.114])
    w = np.clip((big - threshold) / (1 - threshold), 0, 1) * alpha * (luma > 0.12)
    out = base * (1 - w[..., None]) + jet(big) * w[..., None]
    return Image.fromarray((out * 255).astype(np.uint8))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "vit_dr_int8.onnx"))
    ap.add_argument("--images", default=str(HERE / "testdata" / "fundus"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pre = json.load(open(HERE / "preprocess.json"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    if "attn_grads" not in names:
        print(f"{args.model} has outputs {names}: no gradients, nothing to compare")
        return 1

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(pathlib.Path(args.images).iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        img = Image.open(path)
        logits, attn, grads = sess.run(["logits", "attentions", "attn_grads"], {"pixel_values": preprocess(img, pre)})
        roll, grad = attention_rollout(attn), chefer_relevance(attn, grads)
        side = int(round(np.sqrt(grad.size)))
        print(
            f"{path.name}: class={int(logits.argmax())} rho(rollout,grad)={spearman(roll, grad):.3f} "
            f"peak rollout=(r{roll.argmax() // side},c{roll.argmax() % side}) grad=(r{grad.argmax() // side},c{grad.argmax() % side})"
        )
        img = img.convert("RGB")
        w, h = img.size
        panel = Image.new("RGB", (3 * w, h))
        panel.paste(img, (0, 0))
        panel.paste(overlay(img, roll), (w, 0))
        panel.paste(overlay(img, grad), (2 * w, 0))
        panel.save(out_dir / f"{path.stem}.png")
    print(f"wrote panels to {out_dir} (input | rollout | grad-relevance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
