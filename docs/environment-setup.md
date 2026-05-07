# Environment Setup

This project now uses a shared Python environment for ML work, edge work, and the minimal FastAPI backend.

## Recommended Local Setup

- macOS on Apple Silicon or Intel
- Python 3.9+ available as `python3`
- one project virtual environment in `.venv`

Current machine check from this repo:

- Python: `3.9.6`
- Architecture: `arm64`

## Quick Start

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Or with `make`:

```bash
make install-dev
```

The `Makefile` calls the virtual environment's Python directly, so the same `make` targets work on Windows without a manual `activate` step.

## Dependency Groups

`requirements.txt`

- runtime packages for dataset prep, training, inference, FastAPI logging, plotting, and local logging

`requirements-dev.txt`

- runtime packages plus notebook, test, and formatting tools

## Project Conventions

- use one shared `.venv` at the repo root
- save trained model artifacts under `artifacts/models/`
- save edge and evaluation logs under `logs/`
- keep local configuration in a copied file such as `edge/config.yaml`
- use [edge/config.example.yaml](https://github.com/thebkht/smart-parking-system/blob/main/edge/config.example.yaml) as the starting point

## Notes for This Project

- `opencv-python` is used instead of `opencv-python-headless` because the edge track needs webcam support
- `onnxruntime` is included for optional ONNX inference paths
- the backend is intentionally minimal and the frontend is out of scope

## First Tasks After Install

1. Copy `edge/config.example.yaml` to `edge/config.yaml`
2. Create `artifacts/models/` and `logs/` as needed during local work
3. Verify the environment:

```bash
python -c "import cv2, ultralytics, yaml; print('env ok')"
```

4. Start with dataset preparation and the first YOLO baseline

## One-Off Prediction

Use the standalone prediction CLI when you want a quick model output without running the full edge pipeline:

```bash
make predict STAGE2_VARIANT=n PREDICT_SOURCE=samples/demo.jpg
```
