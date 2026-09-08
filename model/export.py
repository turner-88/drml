#!/usr/bin/env python3
"""Export a pretrained DR classifier to ONNX for the Go server.

Run once on a dev machine; never on the VPS. Produces:

    vit_dr.onnx        fp32 export (kept for the int8-vs-fp32 accuracy check)
    vit_dr_int8.onnx   dynamically quantized, the artifact the server loads
    labels.json        grade -> display name
    preprocess.json    normalization constants, so Go never hardcodes them
    testdata/golden.json  fixture for the Go/Python parity test

The ONNX graph has two outputs: `logits` and `attentions`. The attentions are
what make the heatmap possible — Grad-CAM needs a backward pass, which ONNX
Runtime cannot do, but attention rollout is forward-only.

Usage:
    ./.venv/bin/python export.py [--model-id ID] [--skip-quantize]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor

HERE = pathlib.Path(__file__).resolve().parent

DEFAULT_MODEL_ID = "Kontawat/vit-diabetic-retinopathy-classification"

# The APTOS 5-class severity scale. The chosen checkpoint ships a placeholder
# id2label of {"0":"0",...,"4":"4"}, so these names encode an ASSUMPTION about
# ordering that eval.py must confirm against labelled data before the UI is
# trusted. See the class-ordering check in the plan's Verification section.
DR_GRADES = [
    {"grade": 0, "en": "No DR", "id": "Tidak ada DR"},
    {"grade": 1, "en": "Mild", "id": "Ringan"},
    {"grade": 2, "en": "Moderate", "id": "Sedang"},
    {"grade": 3, "en": "Severe", "id": "Berat"},
    {"grade": 4, "en": "Proliferative DR", "id": "Proliferatif"},
]


class WrappedViT(torch.nn.Module):
    """Exposes attention maps as a second graph output.

    The HF forward returns a dataclass, which torch.onnx.export cannot trace
    into multiple named outputs; this flattens it to a plain tuple.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values, output_attentions=True)
        # tuple of L tensors [B, heads, tokens, tokens] -> [L, B, heads, T, T]
        attentions = torch.stack(out.attentions, dim=0)
        return out.logits, attentions


def load_processor(model_id: str):
    """Load the image processor, tolerating pre-v5 checkpoints.

    Checkpoints published before transformers v5 declare
    `image_processor_type: ViTFeatureExtractor`, an alias that v5 removed, so
    AutoImageProcessor refuses them. The stored config is still a valid ViT
    processor config, so fall back to instantiating ViTImageProcessor directly.
    """
    try:
        return AutoImageProcessor.from_pretrained(model_id)
    except ValueError as err:
        if "Unrecognized image processor" not in str(err):
            raise
        print("  note: legacy processor config, falling back to ViTImageProcessor")
        return ViTImageProcessor.from_pretrained(model_id)


# PIL resample codes, recorded numerically so Go matches the exact filter.
PIL_RESAMPLE_NAMES = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_preprocess_config(processor) -> dict:
    """Read normalization constants off the processor rather than guessing.

    Go reads this file at startup, so a model swap that changes mean/std or
    input size does not require touching Go code.
    """
    size = getattr(processor, "size", None) or {}
    if "height" in size and "width" in size:
        height, width = int(size["height"]), int(size["width"])
    elif "shortest_edge" in size:
        height = width = int(size["shortest_edge"])
    else:
        height = width = 224

    resample = int(getattr(processor, "resample", 2) or 2)

    return {
        "height": height,
        "width": width,
        "image_mean": [float(v) for v in getattr(processor, "image_mean", [0.5, 0.5, 0.5])],
        "image_std": [float(v) for v in getattr(processor, "image_std", [0.5, 0.5, 0.5])],
        "rescale_factor": float(getattr(processor, "rescale_factor", 1.0 / 255.0)),
        "do_normalize": bool(getattr(processor, "do_normalize", True)),
        "do_rescale": bool(getattr(processor, "do_rescale", True)),
        # Go must reproduce this exact filter or the parity test fails. PIL
        # antialiases when downscaling, so Go needs a kernel scaler that also
        # widens its support (x/image/draw.BiLinear), not a naive sampler.
        "resample_code": resample,
        "resample": PIL_RESAMPLE_NAMES.get(resample, "bilinear"),
    }


def make_golden_fixture(processor, model, pre: dict, out_dir: pathlib.Path) -> None:
    """Emit a deterministic fixture the Go parity test asserts against.

    A synthetic image keeps the repo free of patient data while still exercising
    resize, channel order, rescale and normalize — the usual sources of silent
    mismatch when a preprocessing pipeline is reimplemented.
    """
    rng = np.random.default_rng(1234)
    h, w = 137, 173  # deliberately non-square and not a multiple of the patch size
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    fixture_png = out_dir / "fixture.png"
    img.save(fixture_png)

    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        logits, attentions = WrappedViT(model)(pixel_values)

    (out_dir / "golden.json").write_text(
        json.dumps(
            {
                "note": "Generated by export.py. Go must reproduce pixel_values within 1e-4 and logits within 1e-3.",
                "image": "fixture.png",
                "image_height": h,
                "image_width": w,
                "preprocess": pre,
                "pixel_values_shape": list(pixel_values.shape),
                "pixel_values": pixel_values.flatten().tolist(),
                "logits": logits.flatten().tolist(),
                "attentions_shape": list(attentions.shape),
            },
            indent=2,
        )
    )
    print(f"  golden fixture: pixel_values{list(pixel_values.shape)} logits{list(logits.shape)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--skip-quantize", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    testdata = HERE / "testdata"
    testdata.mkdir(exist_ok=True)

    print(f"Loading {args.model_id} ...")
    processor = load_processor(args.model_id)
    # eager attention is REQUIRED: the default SDPA kernel does not return
    # attention probabilities, and rollout has nothing to work with without them.
    model = AutoModelForImageClassification.from_pretrained(
        args.model_id, attn_implementation="eager"
    ).eval()

    num_labels = int(model.config.num_labels)
    if num_labels != len(DR_GRADES):
        print(
            f"ERROR: model exposes {num_labels} classes, expected {len(DR_GRADES)}. "
            "labels.json would be wrong — aborting.",
            file=sys.stderr,
        )
        return 1

    pre = build_preprocess_config(processor)
    print(f"  preprocess: {pre['height']}x{pre['width']} mean={pre['image_mean']} std={pre['image_std']}")

    fp32_path = HERE / "vit_dr.onnx"
    dummy = torch.randn(1, 3, pre["height"], pre["width"])

    print(f"Exporting ONNX (opset {args.opset}) ...")
    torch.onnx.export(
        WrappedViT(model),
        (dummy,),
        str(fp32_path),
        input_names=["pixel_values"],
        output_names=["logits", "attentions"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {fp32_path.name} ({fp32_path.stat().st_size / 1e6:.1f} MB)")

    served_path = fp32_path
    if not args.skip_quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = HERE / "vit_dr_int8.onnx"
        print("Quantizing (dynamic, int8) ...")
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"  wrote {int8_path.name} ({int8_path.stat().st_size / 1e6:.1f} MB)")
        served_path = int8_path

    (HERE / "labels.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "model_file": served_path.name,
                "model_sha256": sha256_file(served_path),
                "ordering_verified": False,
                "ordering_note": (
                    "The checkpoint's id2label is a placeholder ({'0':'0',...}). These names "
                    "assume the standard APTOS ordering. Run eval.py against labelled data and "
                    "set ordering_verified=true before relying on these labels clinically."
                ),
                "labels": DR_GRADES,
            },
            indent=2,
        )
    )
    (HERE / "preprocess.json").write_text(json.dumps(pre, indent=2))

    print("Building golden fixture ...")
    make_golden_fixture(processor, model, pre, testdata)

    print("\nDone. Next: run eval.py to confirm class ordering before trusting labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
