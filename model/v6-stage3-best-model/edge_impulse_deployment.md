# Edge Impulse deployment (float32 & float16)

**best_float32.tflite** (~10.2 MB). Deployed for **Nvidia Jetson Nano** (v2), **Nvidia Jetson Orin Nano** (v5), **Raspberry Pi 5** (v3), and **Raspberry Pi 4** (v4). **best_float16.tflite** (~5.2 MB) on Raspberry Pi 5; **best_float8.tflite** (Quantized int8) tested on Pi 5 — see results below.

---

## Edge Impulse configuration

| Setting | Value |
|---------|--------|
| Model input | Image |
| Input shape | (640, 640, 3) |
| Input scaling | Pixels 0..1 (not normalized) |
| Resize mode | Fit longest axis |
| Model output | Object detection |
| Output shape | (64, 8400) |
| Output layer | YOLOv11 (coordinates scaled 0..1) |
| Score threshold | 0.2 |
| Output labels | 60 (traffic sign classes) |

Labels: see `labels_60.txt` in this folder.

### Target device and application budget

| Field | Nvidia Jetson Nano | Nvidia Jetson Orin Nano | Raspberry Pi 5 | Raspberry Pi 4 |
|-------|--------------------|-------------------------|----------------|----------------|
| Target device | Nvidia Jetson Nano | Nvidia Jetson Orin Nano | Raspberry Pi 5 | Raspberry Pi 4 |
| Processor family | Cortex-A | Cortex-A | Cortex-A | Cortex-A |
| Clock rate (max) | — | — | 2.4 GHz | 1.8 GHz |
| RAM (max) | 8 GB | 8 GB | 8 GB | 8 GB |
| ROM (max) | 8 GB | 8 GB | 8 GB | 8 GB |
| Latency (max) | 100 ms | 100 ms | 100 ms | 100 ms |

---

## Model testing results (best_float32.tflite, best_float16.tflite, best_float8.tflite)

Run: 2026-03-04. Test set: 3,914 images (yolo_dataset/images/test). Status: Job completed (success).

| Field | Nvidia Jetson Nano (v2) | Nvidia Jetson Orin Nano (v5) | Raspberry Pi 5 (v3) | Raspberry Pi 4 (v4) | Raspberry Pi 5 (float16) | Raspberry Pi 5 (int8) |
|-------|------------------------|-----------------------------|---------------------|---------------------|---------------------------|------------------------|
| Deployment | Linux (AARCH64) / EIM v2 | Linux (AARCH64) / EIM v5 | Linux (AARCH64) / EIM v3 | Linux (AARCH64) / EIM v4 | — | — |
| Model | best_float32.tflite | best_float32.tflite | best_float32.tflite | best_float32.tflite | best_float16.tflite | best_float8.tflite (Quantized int8) |
| Target device | Nvidia Jetson Nano | Nvidia Jetson Orin Nano | Raspberry Pi 5 | Raspberry Pi 4 | Raspberry Pi 5 | Raspberry Pi 5 |
| Latency | 7,378 ms | 2,715 ms | 564 ms | 10,006 ms | 473 ms | N/A |
| FLASH | 10.2M | 10.2M | 10.2M | 10.2M | 5.2M | — |
| Accuracy | 57.40% | 57.40% | 57.40% | 57.40% | 57.37% | 0.03% |

### Metrics (Pretrained learn)

| Metric | float32 / float16 | best_float8 (int8) |
|--------|--------------------|---------------------|
| **MAP@50** | 0.56 | -1.00 |
| mAP | 0.44 | -1.00 |
| mAP@[IoU=50] | 0.56 | -1.00 |
| mAP@[IoU=75] | 0.52 | -1.00 |
| mAP@[area=small] | 0.19 | -1.00 |
| mAP@[area=medium] | 0.13 | -1.00 |
| mAP@[area=large] | 0.47 | -1.00 |
| Recall@[max_detections=1] | 0.50 | -1.00 |
| Recall@[max_detections=10] | 0.50 | -1.00 |
| Recall@[max_detections=100] | 0.50 | -1.00 |
| Recall@[area=small] | 0.19 | -1.00 |
| Recall@[area=medium] | 0.16 | -1.00 |
| Recall@[area=large] | 0.52 | -1.00 |
| Precision score (legacy) | 53.2% | 0.0% |

**best_float8 (int8):** No valid detections; **not recommended for deployment**. float32/float16 metrics match across devices.

---

## Deployment target

- **Nvidia Jetson Nano (Linux AARCH64):** v2 (float32).
  - Float32 → `trafficsigndetection-linux-aarch64-v2.eim`
- **Nvidia Jetson Orin Nano (Linux AARCH64):** v5 (float32 only, same **best_float32.tflite**).
  - Float32 → `trafficsigndetection-linux-aarch64-v5.eim`
- **Raspberry Pi 5 (Linux AARCH64):** v3 (float32 only, same **best_float32.tflite**).
  - Float32 → `trafficsigndetection-linux-aarch64-v3.eim`
- **Raspberry Pi 4 (Linux AARCH64):** v4 (float32 only, same **best_float32.tflite**).
  - Float32 → `trafficsigndetection-linux-aarch64-v4.eim`

---

## How to use the .eim file (Jetson Nano, Jetson Orin Nano, Raspberry Pi 4 / 5 / Linux AARCH64)

Use **v2** (Jetson Nano), **v5** (Jetson Orin Nano), **v3** (Raspberry Pi 5), or **v4** (Raspberry Pi 4). Replace `<model>.eim` with the file you copied.

1. **Copy the file to the device**  
   Copy the appropriate EIM to your device (e.g. `trafficsigndetection-linux-aarch64-v2.eim` for Jetson Nano float32, `trafficsigndetection-linux-aarch64-v5.eim` for Jetson Orin Nano, `trafficsigndetection-linux-aarch64-v3.eim` for Raspberry Pi 5, `trafficsigndetection-linux-aarch64-v4.eim` for Raspberry Pi 4).

2. **Run it directly** (standalone executable):
   ```bash
   chmod +x <model>.eim
   ./<model>.eim
   ```
   The EIM runs as a process and listens for inference requests over a Unix socket or stdio (JSON in/out).

3. **Or use Edge Impulse Linux runner** (if you installed the EI CLI on the Nano):
   ```bash
   edge-impulse-linux-runner --model-file <model>.eim
   ```

4. **In your own app (Python/C++/Go/Node):**  
   Use the [Edge Impulse Linux SDK](https://docs.edgeimpulse.com/docs/run-inference/linux-eim-executable) for your language. The SDK starts the .eim as a subprocess and sends raw data (e.g. image bytes) and receives JSON with `bounding_boxes` (label, x, y, width, height, value).

5. **Print model info** (input shape, labels, etc.):
   ```bash
   ./<model>.eim --print-info
   ```
   or  
   `edge-impulse-linux-runner --model-file <model>.eim --print-info`

---

## Notes

- Deployed with EI’s YOLOv11 decoder and 60 labels.
- Input: RGB image, values 0..1, resized with “Fit longest axis” to 640×640.
