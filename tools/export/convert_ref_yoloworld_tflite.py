"""Convert the Ref-YOLO-World text-cache ONNX model to TFLite with onnx2tf.

Export the ONNX model with ``--tflite-strip-cls-score-where`` first. onnx2tf
writes NHWC FP32 and FP16 TFLite models and keeps the ONNX input and output
names, which ``check_ref_yoloworld_tflite_parity.py`` relies on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx2tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Ref-YOLO-World text-cache ONNX model to TFLite.")
    parser.add_argument("--onnx", type=Path, required=True,
                        help="Text-cache ONNX model exported with --tflite-strip-cls-score-where")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for the SavedModel and the TFLite models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=str(args.onnx),
        output_folder_path=str(args.output_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True)
    for precision in ("float32", "float16"):
        path = args.output_dir / f"{args.onnx.stem}_{precision}.tflite"
        if not path.exists():
            raise FileNotFoundError(f"onnx2tf did not write {path}")
        print(path)


if __name__ == "__main__":
    main()
