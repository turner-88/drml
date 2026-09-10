#!/usr/bin/env python3
"""Export a pretrained DR classifier to ONNX for the Go server.

Run once on a dev machine; never on the VPS. Produces:

    vit_dr.onnx        fp32 export (kept for the int8-vs-fp32 accuracy check)
    vit_dr_int8.onnx   dynamically quantized, the artifact the server loads
    labels.json        grade -> display name
    preprocess.json    normalization constants, so Go never hardcodes them
    testdata/golden.json  fixture for the Go/Python parity tests

The ONNX graph has three outputs: `logits`, `attentions` and `attn_grads`. The
last is d(predicted-class logit)/d(attention probabilities) for every layer,
which is what a class-specific explanation (Chefer et al., ICCV 2021) needs.
ONNX Runtime cannot differentiate, so the backward pass is baked into the graph
here: the wrapper is differentiated with torch.func.grad, the resulting
forward+backward computation is captured with make_fx, and that is what gets
exported. ORT then computes the gradients in the same Run as the logits.

`--forward-only` exports the old two-output graph (logits, attentions) for a
deployment that cannot afford the backward pass; the server then falls back to
attention rollout, which is class-agnostic and much weaker.

Usage:
    ./.venv/bin/python export.py [--model-id ID] [--skip-quantize] [--forward-only]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import sys
from collections import Counter

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor
from transformers.models.vit import modeling_vit

HERE = pathlib.Path(__file__).resolve().parent

DEFAULT_MODEL_ID = "Kontawat/vit-diabetic-retinopathy-classification"

# ONNX Runtime 1.20 (deploy/Dockerfile) rejects newer IR versions.
MAX_IR_VERSION = 10

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


# ---------------------------------------------------------------------------
# Attention probing
# ---------------------------------------------------------------------------

_ORIGINAL_EAGER_ATTENTION = modeling_vit.eager_attention_forward


@contextlib.contextmanager
def probed_attention(probes: tuple[torch.Tensor, ...], recorder: list[torch.Tensor]):
    """Temporarily replace ViT's eager attention kernel with one that exposes
    the attention probabilities to autograd.

    Each layer adds its zero-valued probe to the probabilities right after the
    softmax (and the eval-mode no-op dropout), so the gradient of anything
    downstream with respect to that probe *is* the gradient with respect to the
    probabilities. The same post-add tensor is recorded, so the `attentions`
    output and the gradient refer to exactly the same quantity.

    transformers resolves the kernel by looking up the module-level name at
    call time, which is what makes a monkeypatch sufficient. SDPA never
    materializes the probabilities, so the model must be loaded with
    attn_implementation="eager".
    """

    def patched(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if scaling is None:
            scaling = query.size(-1) ** -0.5
        weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = nn.functional.dropout(weights, p=dropout, training=module.training)
        weights = weights + probes[module.drml_layer_idx]
        recorder.append(weights)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        return output, weights

    modeling_vit.eager_attention_forward = patched
    try:
        yield
    finally:
        modeling_vit.eager_attention_forward = _ORIGINAL_EAGER_ATTENTION


class ExplainableViT(nn.Module):
    """Forward and backward in one graph.

    Outputs (logits, attentions, attn_grads), the last two both shaped
    [layers, batch, heads, tokens, tokens]. attn_grads is the gradient of the
    predicted class's logit with respect to the attention probabilities.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        layers = self.model.vit.layers
        for i, layer in enumerate(layers):
            # ViTAttention has no layer index of its own; the patched kernel
            # reads this attribute to pick its probe.
            layer.attention.drml_layer_idx = i
        cfg = self.model.config
        tokens = (cfg.image_size // cfg.patch_size) ** 2 + 1
        self.num_layers = len(layers)
        self.probe_shape = (1, cfg.num_attention_heads, tokens, tokens)

    def objective(self, pixel_values: torch.Tensor, probes: tuple[torch.Tensor, ...]):
        recorder: list[torch.Tensor] = []
        with probed_attention(probes, recorder):
            logits = self.model(pixel_values=pixel_values).logits
        # The predicted class's logit, selected with a detached one-hot so the
        # backward is a plain multiply rather than a data-dependent scatter.
        onehot = (logits == logits.amax(dim=-1, keepdim=True)).to(logits.dtype).detach()
        return (logits * onehot).sum(), (logits, torch.stack(recorder, dim=0))

    def zero_probes(self, like: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # One tensor per layer rather than one stacked tensor indexed per
        # layer: indexing would put a 22 MB scatter into every layer's backward.
        return tuple(torch.zeros(self.probe_shape, dtype=like.dtype) for _ in range(self.num_layers))

    def forward(self, pixel_values: torch.Tensor):
        grads, (logits, attentions) = torch.func.grad(self.objective, argnums=1, has_aux=True)(
            pixel_values, self.zero_probes(pixel_values)
        )
        return logits, attentions, torch.stack(grads, dim=0)


class ForwardOnlyViT(nn.Module):
    """The two-output graph: logits and attention probabilities."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.inner = ExplainableViT(model)

    def forward(self, pixel_values: torch.Tensor):
        _, (logits, attentions) = self.inner.objective(pixel_values, self.inner.zero_probes(pixel_values))
        return logits, attentions


# ---------------------------------------------------------------------------
# Reference saliency, shared with the Go parity test and explain.py
# ---------------------------------------------------------------------------


def chefer_relevance(attn: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """Gradient-weighted attention relevance (Chefer et al., ICCV 2021).

    R = I; for each layer: R += mean_heads(relu(grad * A)) @ R. Returns the CLS
    row over the patch tokens, un-normalized. Go's GradRelevance must match.
    """
    layers, _, _, tokens, _ = attn.shape
    r = np.eye(tokens, dtype=np.float64)
    for l in range(layers):
        cam = np.clip(grads[l, 0].astype(np.float64) * attn[l, 0].astype(np.float64), 0, None).mean(axis=0)
        r = r + cam @ r
    return r[0, 1:]


def attention_rollout(attn: np.ndarray) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020), matching Go's Rollout."""
    layers, _, _, tokens, _ = attn.shape
    r = np.eye(tokens, dtype=np.float64)
    for l in range(layers):
        a = 0.5 * attn[l, 0].astype(np.float64).mean(axis=0) + 0.5 * np.eye(tokens)
        a /= a.sum(axis=1, keepdims=True)
        r = a @ r
    return r[0, 1:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_onnx(wrapper: nn.Module, dummy: torch.Tensor, output_names: list[str], opset: int) -> onnx.ModelProto:
    """Capture the wrapper with make_fx and hand the result to the ONNX exporter.

    torch.export cannot run the autograd engine on fake tensors, which is what
    torch.func.grad needs, so the capture uses make_fx in real mode: it runs
    the module once on real tensors and records every op, backward included,
    as a plain forward graph. Parameters become constants in that graph.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    graph = make_fx(wrapper, tracing_mode="real")(dummy).eval()
    program = torch.onnx.export(
        graph,
        (dummy,),
        input_names=["pixel_values"],
        output_names=output_names,
        opset_version=opset,
        dynamo=True,
        external_data=False,
        optimize=True,
        verify=False,
    )
    model = program.model_proto
    if model.ir_version > MAX_IR_VERSION:
        model.ir_version = MAX_IR_VERSION
    return model


def fold_transposed_weights(model: onnx.ModelProto) -> int:
    """Turn MatMul(x, Transpose(W)) into MatMul(x, W_T) for constant W.

    The backward pass multiplies by every weight's transpose. The exporter
    emits that as a Transpose node in front of the MatMul, which the dynamic
    quantizer does not treat as a constant weight, so those MatMuls would stay
    fp32 — measured at about 60% of the whole backward's time. Materializing
    the transposed copies doubles the weight bytes on disk and in memory but
    lets every backward MatMul quantize.
    """
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    producer = {out: node for node in graph.node for out in node.output}
    made: dict[tuple, str] = {}
    folded = 0
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        transpose = producer.get(node.input[1])
        if transpose is None or transpose.op_type != "Transpose" or transpose.input[0] not in inits:
            continue
        perm = next((list(a.ints) for a in transpose.attribute if a.name == "perm"), None)
        key = (transpose.input[0], tuple(perm) if perm else None)
        if key not in made:
            arr = numpy_helper.to_array(inits[transpose.input[0]])
            arr_t = np.ascontiguousarray(arr.transpose(perm) if perm else arr.T)
            new = numpy_helper.from_array(arr_t, transpose.input[0] + "_T")
            graph.initializer.append(new)
            made[key] = new.name
        node.input[1] = made[key]
        folded += 1

    used = {i for node in graph.node for i in node.input} | {o.name for o in graph.output}
    keep = [n for n in graph.node if not (n.op_type == "Transpose" and n.output[0] not in used)]
    del graph.node[:]
    graph.node.extend(keep)
    return folded


def check_gradient(wrapper: ExplainableViT, pixel_values: torch.Tensor, grads: torch.Tensor) -> None:
    """Central finite differences on the largest gradient entries.

    A wrong injection point (before the softmax, say) would still export and
    still produce plausible-looking tensors; this is the check that the graph's
    gradient really is d(logit)/d(probability).
    """
    flat = grads.abs().flatten()
    top = torch.topk(flat, 3).indices
    delta = 1e-3
    for idx in top:
        pos = np.unravel_index(int(idx), grads.shape)
        layer, rest = pos[0], pos[1:]

        def target(sign: float) -> float:
            probes = list(wrapper.zero_probes(pixel_values))
            probes[layer] = probes[layer].clone()
            probes[layer][rest] = sign * delta
            with torch.no_grad():
                value, _ = wrapper.objective(pixel_values, tuple(probes))
            return float(value)

        fd = (target(+1) - target(-1)) / (2 * delta)
        analytic = float(grads[pos])
        rel = abs(fd - analytic) / max(abs(analytic), 1e-9)
        print(f"  gradient check {pos}: analytic={analytic:.5f} finite-diff={fd:.5f} rel-err={rel:.3g}")
        if rel > 0.05:
            raise SystemExit("ERROR: exported gradient disagrees with finite differences — probe injection is wrong")


def onnx_session(path: pathlib.Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def compare_int8_relevance(fp32_path: pathlib.Path, int8_path: pathlib.Path, processor, fundus_dir: pathlib.Path) -> None:
    """Report how much dynamic quantization moves the saliency map."""
    images = sorted(fundus_dir.glob("*.png")) if fundus_dir.is_dir() else []
    if not images:
        print("  (no testdata/fundus images; skipping int8 saliency comparison)")
        return
    fp32, int8 = onnx_session(fp32_path), onnx_session(int8_path)
    worst = 1.0
    for path in images:
        pv = processor(images=Image.open(path).convert("RGB"), return_tensors="np")["pixel_values"]
        lg32, a32, g32 = fp32.run(None, {"pixel_values": pv})
        lg8, a8, g8 = int8.run(None, {"pixel_values": pv})
        r32, r8 = chefer_relevance(a32, g32), chefer_relevance(a8, g8)
        cos = float(r32 @ r8 / (np.linalg.norm(r32) * np.linalg.norm(r8) + 1e-12))
        worst = min(worst, cos)
        print(
            f"  {path.name}: class fp32={lg32.argmax()} int8={lg8.argmax()} "
            f"relevance cosine={cos:.3f} peak fp32={r32.argmax()} int8={r8.argmax()}"
        )
    print(f"  worst int8-vs-fp32 relevance cosine: {worst:.3f}")
    if worst < 0.95:
        print("  WARNING: quantization is distorting the saliency map noticeably; consider --skip-quantize")


def make_golden_fixture(processor, wrapper: ExplainableViT, pre: dict, out_dir: pathlib.Path) -> None:
    """Emit deterministic fixtures the Go parity tests assert against.

    A synthetic image keeps the repo free of patient data while still exercising
    resize, channel order, rescale and normalize — the usual sources of silent
    mismatch when a preprocessing pipeline is reimplemented. A class-specific
    saliency map on noise is a weak target, though, so when the fundus samples
    are present one of them is recorded too, with its tensor pre-computed so
    the saliency comparison does not depend on preprocessing.
    """
    rng = np.random.default_rng(1234)
    h, w = 137, 173  # deliberately non-square and not a multiple of the patch size
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    fixture_png = out_dir / "fixture.png"
    img.save(fixture_png)

    def run(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        logits, attentions, grads = wrapper(pixel_values)
        relevance = chefer_relevance(attentions.detach().numpy(), grads.detach().numpy())
        return pixel_values, logits.detach(), relevance

    pixel_values, logits, relevance = run(img)
    golden = {
        "note": (
            "Generated by export.py. Go must reproduce pixel_values within 1e-4, logits within 1e-3 "
            "and, for the fp32 graph, relevance within 1e-3 after normalization."
        ),
        "image": "fixture.png",
        "image_height": h,
        "image_width": w,
        "preprocess": pre,
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values": pixel_values.flatten().tolist(),
        "logits": logits.flatten().tolist(),
        "attentions_shape": [wrapper.num_layers, *wrapper.probe_shape],
        "attn_grads_shape": [wrapper.num_layers, *wrapper.probe_shape],
        "relevance": relevance.tolist(),
    }

    fundus = out_dir / "fundus" / "00.png"
    if fundus.exists():
        f_pixels, f_logits, f_relevance = run(Image.open(fundus).convert("RGB"))
        golden["fundus_fixture"] = {
            "image": "fundus/00.png",
            "pixel_values": f_pixels.flatten().tolist(),
            "logits": f_logits.flatten().tolist(),
            "relevance": f_relevance.tolist(),
        }

    (out_dir / "golden.json").write_text(json.dumps(golden, indent=2))
    print(f"  golden fixture: pixel_values{list(pixel_values.shape)} logits{list(logits.shape)} relevance[{len(relevance)}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--skip-quantize", action="store_true")
    ap.add_argument("--forward-only", action="store_true", help="export logits+attentions only (no gradients)")
    ap.add_argument(
        "--no-fold-transposes",
        action="store_true",
        help="keep backward weight transposes as nodes: half the file size, but the backward MatMuls stay fp32",
    )
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    testdata = HERE / "testdata"
    testdata.mkdir(exist_ok=True)

    print(f"Loading {args.model_id} ...")
    processor = load_processor(args.model_id)
    # eager attention is REQUIRED: the default SDPA kernel does not materialize
    # attention probabilities, and there would be nothing to probe.
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
    explainable = ExplainableViT(model)

    if args.forward_only:
        wrapper: nn.Module = ForwardOnlyViT(model)
        output_names = ["logits", "attentions"]
    else:
        wrapper = explainable
        output_names = ["logits", "attentions", "attn_grads"]

    print(f"Exporting ONNX (opset {args.opset}, outputs {output_names}) ...")
    onnx_model = export_onnx(wrapper, dummy, output_names, args.opset)
    if not args.forward_only and not args.no_fold_transposes:
        print(f"  folded {fold_transposed_weights(onnx_model)} transposed weight MatMuls")
    onnx.save(onnx_model, str(fp32_path))
    ops = Counter(n.op_type for n in onnx_model.graph.node)
    print(
        f"  wrote {fp32_path.name} ({fp32_path.stat().st_size / 1e6:.1f} MB, ir {onnx_model.ir_version}, "
        f"{len(onnx_model.graph.node)} nodes, {ops['MatMul'] + ops['Gemm']} matmuls)"
    )
    got = [o.name for o in onnx_model.graph.output]
    if got != output_names:
        print(f"ERROR: graph outputs are {got}, expected {output_names}", file=sys.stderr)
        return 1

    if not args.forward_only:
        print("Checking the exported gradient against finite differences ...")
        _, _, grads = explainable(dummy)
        check_gradient(explainable, dummy, grads.detach())

        sess = onnx_session(fp32_path)
        _, ort_attn, ort_grads = sess.run(None, {"pixel_values": dummy.numpy()})
        err = float(np.abs(ort_grads - grads.detach().numpy()).max())
        print(f"  ORT vs torch attn_grads max|diff| = {err:.2e}")
        if err > 1e-4:
            print("ERROR: ONNX Runtime's gradient does not match torch", file=sys.stderr)
            return 1

    served_path = fp32_path
    if not args.skip_quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = HERE / "vit_dr_int8.onnx"
        print("Quantizing (dynamic, int8) ...")
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"  wrote {int8_path.name} ({int8_path.stat().st_size / 1e6:.1f} MB)")
        served_path = int8_path
        if not args.forward_only:
            print("Comparing int8 vs fp32 saliency ...")
            compare_int8_relevance(fp32_path, int8_path, processor, testdata / "fundus")

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
    make_golden_fixture(processor, explainable, pre, testdata)

    print("\nDone. Next: run eval.py to confirm class ordering before trusting labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
