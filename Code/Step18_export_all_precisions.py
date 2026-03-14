#!/usr/bin/env python3
"""
Export best.pt to model/v6-stage3-best-model/:
  - 6 precision variants: best_float8/16/32 .onnx and .tflite (raw head, 1,64,8400).
  - best_edgeimpulse.tflite: TFLite with NMS (output 1,max_det,6) for Edge Impulse / Jetson / RPi.

TFLite int8 uses onnx2tf+concrete-func when Ultralytics int8 export fails.
Edge Impulse model has NMS baked in so EI can show bounding boxes (max_det=100).
"""

import os
import shutil
import tempfile
from pathlib import Path

# Use legacy Keras for TFLite/onnx2tf
os.environ.pop("TF_USE_LEGACY_KERAS", None)

from ultralytics import YOLO


WEIGHTS = Path("/Users/jvarghese/Documents/TrafficSignProject/model/v6-stage3-best-model/best.pt")
TARGET_DIR = WEIGHTS.parent
IMGSZ = 640
MAX_DET_EI = 100  # Edge Impulse NMS export: max detections per image
DATA_YAML = Path("/Users/jvarghese/Documents/TrafficSignProject/yolo_dataset/dataset.yaml")


def _quantize_onnx_int8(onnx_path: Path, out_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print(f"  Quantizing to int8 -> {out_path.name}")
    quantize_dynamic(
        str(onnx_path),
        str(out_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
    )


def _export_tflite_int8_standalone(onnx_path: Path, out_tflite: Path) -> bool:
    """Produce best_float8.tflite via onnx2tf (disable_model_save) + TFLiteConverter. Returns True if successful."""
    import numpy as np
    import tensorflow as tf

    def representative_dataset_gen():
        for _ in range(100):
            yield [np.random.rand(1, IMGSZ, IMGSZ, 3).astype(np.float32)]

    try:
        import onnx2tf
    except ImportError:
        print("  Standalone int8 skipped: onnx2tf not installed")
        return False

    print("  Standalone int8: ONNX -> Keras (onnx2tf, disable_model_save=True)...")
    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as d:
        model = onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=d,
            not_use_onnxsim=True,
            verbosity="error",
            output_signaturedefs=False,
            output_integer_quantized_tflite=False,
            enable_batchmatmul_unfold=True,
            disable_model_save=True,
        )
        run_model = tf.function(lambda *inputs: model(inputs))
        concrete_func = run_model.get_concrete_function(
            *[tf.TensorSpec(t.shape, t.dtype) for t in model.inputs]
        )
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(f"  Full int8 failed ({e}), trying dynamic-range...")
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()
    out_tflite.write_bytes(tflite_model)
    print(f"  -> best_float8.tflite ({out_tflite.stat().st_size / 1e6:.2f} MB)")
    return True


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Export from a temp dir so best.onnx and best_saved_model never appear in TARGET_DIR
    with tempfile.TemporaryDirectory(prefix="yolo_export_") as tmp:
        tmp = Path(tmp)
        shutil.copy2(WEIGHTS, tmp / "best.pt")
        print(f"Loading {WEIGHTS}")
        model = YOLO(str(tmp / "best.pt"))

        # --- ONNX: float32, float16, float8 (int8) ---
        print("\n--- ONNX ---")
        print("  Exporting FP32...")
        p = model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True)
        shutil.copy2(p, TARGET_DIR / "best_float32.onnx")
        print(f"  -> best_float32.onnx")

        print("  Exporting FP16...")
        try:
            p = model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True, half=True)
            shutil.copy2(p, TARGET_DIR / "best_float16.onnx")
            print(f"  -> best_float16.onnx")
        except Exception as e:
            print(f"  FP16 skipped (needs GPU): {e}")

        print("  Exporting int8 (best_float8.onnx)...")
        _quantize_onnx_int8(TARGET_DIR / "best_float32.onnx", TARGET_DIR / "best_float8.onnx")
        print(f"  -> best_float8.onnx")

        # --- TFLite: int8 (standalone fallback), float32, float16 ---
        print("\n--- TFLite ---")
        print("  Exporting int8 (best_float8.tflite)...")
        export_kw = dict(format="tflite", imgsz=IMGSZ, int8=True)
        if DATA_YAML.exists():
            export_kw["data"] = str(DATA_YAML)
        int8_ok = False
        try:
            p = model.export(**export_kw)
            shutil.copy2(p, TARGET_DIR / "best_float8.tflite")
            print(f"  -> best_float8.tflite")
            int8_ok = True
        except Exception as e:
            print(f"  Ultralytics int8 failed: {e}")
        if not int8_ok and (TARGET_DIR / "best_float32.onnx").is_file():
            int8_ok = _export_tflite_int8_standalone(
                TARGET_DIR / "best_float32.onnx", TARGET_DIR / "best_float8.tflite"
            )
        if not int8_ok:
            print("  best_float8.tflite skipped")

        print("  Exporting FP32...")
        p = model.export(format="tflite", imgsz=IMGSZ)
        shutil.copy2(p, TARGET_DIR / "best_float32.tflite")
        print(f"  -> best_float32.tflite")

        print("  Exporting FP16...")
        p = model.export(format="tflite", imgsz=IMGSZ, half=True)
        shutil.copy2(p, TARGET_DIR / "best_float16.tflite")
        print(f"  -> best_float16.tflite")

        # --- Edge Impulse: TFLite with NMS (decoded boxes for EI / Jetson / RPi) ---
        print("\n--- Edge Impulse (NMS) ---")
        try:
            p = model.export(
                format="tflite",
                imgsz=IMGSZ,
                nms=True,
                conf=0.25,
                iou=0.45,
                max_det=MAX_DET_EI,
            )
            src = Path(p)
            if not src.is_file():
                for f in Path(p).rglob("*.tflite"):
                    if "int16" not in f.name:
                        src = f
                        break
            if src.is_file():
                shutil.copy2(src, TARGET_DIR / "best_edgeimpulse.tflite")
                print(f"  -> best_edgeimpulse.tflite ({(TARGET_DIR / 'best_edgeimpulse.tflite').stat().st_size / 1e6:.2f} MB)")
            else:
                print("  best_edgeimpulse.tflite skipped (no tflite under export dir)")
        except Exception as e:
            print(f"  best_edgeimpulse.tflite skipped: {e}")

    # Temp dir removed here: no best.onnx, no best_saved_model in TARGET_DIR

    print(f"\nDone. Files in {TARGET_DIR}")
    for name in [
        "best_float8.tflite", "best_float16.tflite", "best_float32.tflite",
        "best_float8.onnx", "best_float16.onnx", "best_float32.onnx",
        "best_edgeimpulse.tflite",
    ]:
        f = TARGET_DIR / name
        print(f"  {name}  {f.stat().st_size / 1e6:.2f} MB" if f.exists() else f"  {name}  (missing)")


if __name__ == "__main__":
    main()
