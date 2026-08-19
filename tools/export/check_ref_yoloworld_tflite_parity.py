"""Check ONNX vs TFLite parity for Ref-YOLO-World text-cache exports."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Ref-YOLO-World text-cache ONNX and TFLite outputs.")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--tflite", required=True, help="TFLite model path")
    parser.add_argument("--summary-output", default=None, help="Optional JSON output path")
    parser.add_argument("--seed", type=int, default=788)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def normalized_random(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.normal(size=shape).astype(np.float32)
    return values / (np.linalg.norm(values, axis=-1, keepdims=True) + 1e-6)


def make_onnx_inputs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "images": rng.random((1, 3, 640, 640), dtype=np.float32),
        "pooled_text_features": normalized_random(rng, (1, 1, 512)),
        "token_features": normalized_random(rng, (1, 1, 77, 512)),
        "token_attention_mask": np.ones((1, 1, 77), dtype=np.float32),
        "text_mask": np.ones((1, 1), dtype=np.float32),
    }


def to_tflite_input(name: str, onnx_inputs: dict[str, np.ndarray]) -> np.ndarray:
    value = onnx_inputs[name]
    if name == "images":
        return np.transpose(value, (0, 2, 3, 1))
    if name == "pooled_text_features":
        return np.transpose(value, (0, 2, 1))
    if name == "token_features":
        return np.transpose(value, (0, 2, 3, 1))
    if name == "token_attention_mask":
        return np.transpose(value, (0, 2, 1))
    if name == "text_mask":
        return value
    raise KeyError(f"Unsupported TFLite input name: {name}")


def run_onnx(path: str, inputs: dict[str, np.ndarray]) -> tuple[list[np.ndarray], float]:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    outputs = session.run(None, inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return outputs, elapsed_ms


def run_tflite(path: str,
               inputs: dict[str, np.ndarray],
               num_threads: int) -> tuple[list[np.ndarray], float, list[dict[str, Any]]]:
    interpreter = tf.lite.Interpreter(model_path=path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    for detail in input_details:
        value = to_tflite_input(detail["name"], inputs).astype(detail["dtype"])
        interpreter.set_tensor(detail["index"], value)
    start = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    outputs = [
        interpreter.get_tensor(detail["index"])
        for detail in interpreter.get_output_details()
    ]
    serializable_inputs = [
        {
            "name": detail["name"],
            "shape": detail["shape"].tolist(),
            "dtype": str(detail["dtype"]),
        }
        for detail in input_details
    ]
    return outputs, elapsed_ms, serializable_inputs


def compare_outputs(onnx_outputs: list[np.ndarray],
                    tflite_outputs: list[np.ndarray],
                    atol: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (onnx_output, tflite_output) in enumerate(zip(onnx_outputs, tflite_outputs)):
        shape_match = onnx_output.shape == tflite_output.shape
        finite = bool(np.isfinite(tflite_output).all())
        max_abs = None
        mean_abs = None
        pass_output = False
        if shape_match and finite:
            diff = np.abs(onnx_output - tflite_output)
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            pass_output = max_abs <= atol
        rows.append({
            "idx": idx,
            "shape_match": bool(shape_match),
            "onnx_shape": list(onnx_output.shape),
            "tflite_shape": list(tflite_output.shape),
            "finite": finite,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "pass": pass_output,
        })
    return rows


def main() -> None:
    args = parse_args()
    inputs = make_onnx_inputs(args.seed)
    onnx_outputs, onnx_ms = run_onnx(args.onnx, inputs)
    tflite_outputs, tflite_ms, input_details = run_tflite(
        args.tflite, inputs, args.num_threads)
    comparisons = compare_outputs(onnx_outputs, tflite_outputs, args.atol)
    summary = {
        "onnx": args.onnx,
        "tflite": args.tflite,
        "seed": args.seed,
        "atol": args.atol,
        "onnx_ms": onnx_ms,
        "tflite_ms": tflite_ms,
        "tflite_inputs": input_details,
        "outputs": comparisons,
        "pass": all(row["pass"] for row in comparisons),
    }
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
