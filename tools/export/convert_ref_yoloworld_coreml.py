"""Convert the Ref-YOLO-World text-cache ONNX model to a Core ML ML Program.

The ONNX graph from ``export_ref_yoloworld_text_cache_onnx.py`` is converted to
PyTorch with onnx2torch, traced to TorchScript, and converted with coremltools.
The text-side tensors stay model inputs, so the Core ML model runs only the
per-frame detector. On macOS the script also compares the Core ML outputs with
ONNX Runtime on a dummy input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx2torch import convert
from onnx2torch.node_converters.comparisons import OnnxCompare
from onnx2torch.node_converters.isinf import OnnxIsInf
from onnx2torch.node_converters.isnan import OnnxIsNaN
from onnx2torch.node_converters.split import OnnxSplit13
from PIL import Image

INPUTS = [
    ("images", (1, 3, 640, 640)),
    ("pooled_text_features", (1, 1, 512)),
    ("token_features", (1, 1, 77, 512)),
    ("token_attention_mask", (1, 1, 77)),
    ("text_mask", (1, 1)),
]
OUTPUTS = [
    "cls_score_s0", "cls_score_s1", "cls_score_s2",
    "bbox_pred_s0", "bbox_pred_s1", "bbox_pred_s2",
]
DEPLOYMENT_TARGETS = {17: ct.target.iOS17, 18: ct.target.iOS18}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Ref-YOLO-World text-cache ONNX model to Core ML.")
    parser.add_argument("--onnx", type=Path, required=True,
                        help="Text-cache ONNX model path")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .mlpackage path")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON conversion report path")
    parser.add_argument("--compute-precision", choices=("fp32", "fp16"), default="fp16",
                        help="Core ML internal compute precision; inputs and outputs stay FP32")
    parser.add_argument("--minimum-deployment-target", type=int,
                        choices=sorted(DEPLOYMENT_TARGETS), default=17,
                        help="Minimum iOS version of the converted model")
    parser.add_argument("--image-input", action="store_true",
                        help="Expose images as an RGB ImageType with 1/255 scaling")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def constant_values(model: onnx.ModelProto) -> dict[str, list]:
    values = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                values[node.output[0]] = onnx.numpy_helper.to_array(
                    onnx.helper.get_attribute_value(attribute)).tolist()
    return values


def compare(self: OnnxCompare, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if self.compare_function is torch.greater:
        return x > y
    if self.compare_function is torch.less:
        return x < y
    if self.compare_function is torch.eq:
        return x == y
    if self.compare_function is torch.greater_equal:
        return x >= y
    return x <= y


def make_torch_model(model: onnx.ModelProto) -> torch.nn.Module:
    """Convert ONNX to PyTorch with ops that TorchScript and coremltools accept."""
    values = constant_values(model)
    split_sizes = [values[node.input[1]]
                   for node in model.graph.node if node.op_type == "Split"]

    # onnx2torch needs explicit Clip bounds.
    clips = (node for node in model.graph.node if node.op_type == "Clip")
    for clip_index, node in enumerate(clips):
        for input_index in range(1, len(node.input)):
            if node.input[input_index]:
                continue
            name = f"__coreml_clip_{clip_index}_{input_index}"
            value = -np.inf if input_index == 1 else np.inf
            model.graph.initializer.append(onnx.numpy_helper.from_array(
                np.array(value, dtype=np.float32), name))
            node.input[input_index] = name

    module = convert(model).eval()

    # Fix Split sizes so that tracing does not depend on runtime tensors.
    splits = [m for m in module.modules() if isinstance(m, OnnxSplit13)]
    if len(splits) != len(split_sizes):
        raise RuntimeError(f"Split mismatch: {len(splits)} != {len(split_sizes)}")
    for split, sizes in zip(splits, split_sizes):
        split.fixed_split = sizes
    OnnxSplit13.forward = (
        lambda self, x, split=None: torch.split(x, self.fixed_split, dim=self.axis))
    OnnxCompare.forward = compare
    OnnxIsNaN.forward = lambda self, x: x != x
    OnnxIsInf.forward = lambda self, x: torch.abs(x) > 3.4028234e38
    return module


def dummy_inputs() -> tuple[torch.Tensor, ...]:
    tensors = [torch.zeros(shape, dtype=torch.float32) for _, shape in INPUTS]
    tensors[3].fill_(1)
    tensors[4].fill_(1)
    return tuple(tensors)


def compare_with_onnx(onnx_path: Path,
                      coreml_path: Path,
                      examples: tuple[torch.Tensor, ...],
                      image_input: bool) -> dict[str, dict]:
    reference_inputs = {name: x.numpy() for (name, _), x in zip(INPUTS, examples)}
    coreml_inputs = dict(reference_inputs)
    if image_input:
        coreml_inputs["images"] = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = session.run(OUTPUTS, reference_inputs)
    model = ct.models.MLModel(str(coreml_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    candidate = model.predict(coreml_inputs)
    comparison = {}
    for name, expected in zip(OUTPUTS, reference):
        actual = candidate[name]
        difference = np.abs(expected - actual)
        comparison[name] = {
            "shape": list(actual.shape),
            "finite": bool(np.isfinite(actual).all()),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        }
    return comparison


def main() -> None:
    args = parse_args()
    examples = dummy_inputs()
    traced = torch.jit.trace(make_torch_model(onnx.load(args.onnx)), examples, strict=False)

    if args.image_input:
        image_type = ct.ImageType(name="images", shape=examples[0].shape,
                                  scale=1 / 255.0, color_layout=ct.colorlayout.RGB)
    else:
        image_type = ct.TensorType(name="images", shape=examples[0].shape, dtype=np.float32)
    inputs = [image_type] + [
        ct.TensorType(name=name, shape=x.shape, dtype=np.float32)
        for (name, _), x in zip(INPUTS[1:], examples[1:])
    ]
    compute_precision = (ct.precision.FLOAT16 if args.compute_precision == "fp16"
                         else ct.precision.FLOAT32)

    started = time.perf_counter()
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=DEPLOYMENT_TARGETS[args.minimum_deployment_target],
        compute_precision=compute_precision,
        inputs=inputs,
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in OUTPUTS])
    conversion_seconds = time.perf_counter() - started

    source_sha256 = sha256(args.onnx)
    model.short_description = (
        f"Ref-YOLO-World 640x640, {args.compute_precision.upper()} compute")
    model.user_defined_metadata["source_onnx_sha256"] = source_sha256
    model.user_defined_metadata["compute_precision"] = args.compute_precision.upper()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))

    report = {
        "onnx": str(args.onnx),
        "source_onnx_sha256": source_sha256,
        "output": str(args.output),
        "compute_precision": args.compute_precision.upper(),
        "minimum_deployment_target": f"iOS {args.minimum_deployment_target}",
        "image_input": args.image_input,
        "conversion_seconds": conversion_seconds,
        "coremltools_version": ct.__version__,
        "torch_version": torch.__version__,
    }
    # Core ML prediction is only available on macOS.
    if sys.platform == "darwin":
        report["cpu_only_dummy_comparison"] = compare_with_onnx(
            args.onnx, args.output, examples, args.image_input)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
