# Progressive Distillation for Indian Traffic Sign Detection Using YOLOv12

Staged progressive knowledge distillation through the YOLOv12 architecture family (x→l→m→s→n) for edge-deployable Indian traffic sign detection. Achieves **22.8:1 compression** while retaining **88.9% teacher accuracy**, producing a **5.4 MB** model validated on embedded platforms.

## Key Results

| Metric | Value |
|--------|-------|
| Final model (V6) | YOLOv12n, 2.6M params, 5.4 MB |
| Test mAP@0.5 (1024px) | 76.3% |
| Teacher retention | 88.9% of YOLOv12x (85.8%) |
| vs. single-step compression | +5.8pp (76.3% vs 70.5%) |
| Best edge latency | 473 ms on Raspberry Pi 5 (float16) |

## Dataset

30,000 images across **60 IRC-compliant** Indian traffic sign classes (27 mandatory + 33 cautionary), augmented from 604 base images with Indian road degradation conditions (monsoon, sign damage, night, occlusion).

| Split | Images |
|-------|--------|
| Train | 23,754 |
| Val | 2,332 |
| Test | 3,914 |

## Training Strategy Progression

Six versions isolate the contribution of each technique:

| Version | Strategy | Test mAP@0.5 |
|---------|----------|:------------:|
| V1 | Baseline (640px) | 61.6% |
| V2 | + Augmentation (800px) | 71.4% |
| V3 | + Same-arch distillation + multi-res | 75.6% |
| V4 | YOLOv12x teacher | 85.8% |
| V5 | Single-step x→n (22.8×) | 70.5% |
| V6 | **Progressive x→l→m→s→n** | **76.3%** |

## Edge Deployment

V6 exported to TFLite via Edge Impulse, evaluated on 3,914 test images at 640×640:

| Platform | Precision | Latency | mAP@0.5 |
|----------|-----------|---------|---------|
| Raspberry Pi 5 | float16 | 473 ms | 56.0% |
| Raspberry Pi 5 | float32 | 564 ms | 56.0% |
| Jetson Orin Nano | float32 | 2,715 ms | 56.0% |
| Raspberry Pi 4 | float32 | 10,006 ms | 56.0% |

## Project Structure

```
├── Code/
│   ├── Step0–6         # Data preparation pipeline
│   ├── Step7–9         # V1–V3 training (baseline → distillation)
│   ├── Step10–12       # V1–V3 multi-resolution testing
│   ├── Step13–14       # V4 teacher training & testing
│   ├── Step15–16       # V5 single-step distillation & testing
│   ├── Step17_Step1–4  # V6 progressive distillation (4 steps)
│   ├── Step18          # TFLite/ONNX export
│   └── analysis/       # Dataset analysis artifacts
├── model/
│   ├── pre-trained-models/    # YOLOv12x/l/m/s/n weights
│   └── v6-stage3-best-model/  # V6 exports (PyTorch, TFLite, ONNX)
├── organized_dataset/         # Class-organized images & annotations
├── OriginalDataSet/           # 604 base IRC-compliant sign images
└── yolo_dataset/
    ├── dataset.yaml
    └── images/{train,val,test}/
```

## Requirements

- Python 3.9+
- PyTorch 2.9+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv12)
- [Git LFS](https://git-lfs.github.com/) (for model artifacts)
- Google Colab Pro with NVIDIA A100 (for training)

## Authors

**Shiney Thomas**, **John Varghese**, **Juby Mathew**
Department of Computer Science and Engineering, Amal Jyothi College of Engineering (Autonomous), Kanjirappally, Kerala, India
