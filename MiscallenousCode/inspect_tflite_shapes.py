#!/usr/bin/env python3
"""Print input/output details and output shape for each of the three .tflite models."""

import numpy as np
import tensorflow as tf
from pathlib import Path

TARGET_DIR = Path("/Users/jvarghese/Documents/TrafficSignProject/model/v6-stage3-best-model")
NAMES = ["best_float8.tflite", "best_float16.tflite", "best_float32.tflite", "best_edgeimpulse.tflite"]


def inspect(path: Path) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input:", input_details)
    print("Output:", output_details)

    # Dummy input: use model's expected dtype (float32 for fp16/fp32, int8 for int8 model)
    inp = input_details[0]
    shape = inp["shape"]
    dtype = inp["dtype"]
    if np.issubdtype(dtype, np.integer):
        img = np.random.randint(-128, 128, size=shape, dtype=np.int8)
    else:
        img = np.random.rand(*shape).astype(np.float32)
    interpreter.set_tensor(inp["index"], img)
    interpreter.invoke()

    for i, out in enumerate(output_details):
        data = interpreter.get_tensor(out["index"])
        print(f"Output[{i}] shape: {data.shape}, dtype: {data.dtype}")
        print(f"  Sample (first 10): {data.ravel()[:10]}")
    print()


def main():
    for name in NAMES:
        path = TARGET_DIR / name
        if not path.is_file():
            print(f"SKIP (not found): {path}\n")
            continue
        print("=" * 60)
        print(path.name)
        print("=" * 60)
        inspect(path)


if __name__ == "__main__":
    main()
