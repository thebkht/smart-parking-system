import time
import numpy as np
from ultralytics import YOLO

PT = "runs/acpds_cls/yolov8n_stage2/weights/best.pt"
ONNX = "artifacts/models/best.onnx"
ONNX_INT8 = "artifacts/models/best_int8.onnx"

patch = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


def bench(label, weights, device, warmup=10, iters=200):
    model = YOLO(weights, task="classify")
    for _ in range(warmup):
        model(patch, device=device, imgsz=128, verbose=False)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(patch, device=device, imgsz=128, verbose=False)
        times.append(time.perf_counter() - t0)
    ms = sum(times) / len(times) * 1000
    print(f"{label}: {ms:.1f} ms -> {1000 / ms:.0f} FPS")


bench("YOLO MPS", PT, "mps")
bench("YOLO CPU", PT, "cpu")
bench("ONNX FP32", ONNX, "cpu")
bench("ONNX INT8", ONNX_INT8, "cpu")
