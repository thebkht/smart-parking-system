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
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or with `make`:

```bash
make install
```

The `Makefile` calls the virtual environment's Python directly, so the same `make` targets work on Windows without a manual `activate` step.

## Dependency File

`requirements.txt`

- the single source of truth for runtime, test, and formatting dependencies used by this repo

## Project Conventions

- use one shared `.venv` at the repo root
- save trained model artifacts under `artifacts/models/`
- save edge and evaluation logs under `logs/`
- keep local configuration in a copied file such as `edge/config.yaml`
- use [edge/config.example.yaml](https://github.com/thebkht/smart-parking-system/blob/main/edge/config.example.yaml) as the starting point

## Notes for This Project

- `opencv-python` is used instead of `opencv-python-headless` because the edge track needs webcam support
- `onnxruntime` is included for optional ONNX inference paths
- the FastAPI backend, Vite web app, and Expo mobile app are all part of the current demo path

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

## Backend And Mobile Demo

Start the backend from the repo root:

```bash
make backend
```

By default this binds uvicorn to `0.0.0.0:8000`. That is intentional for Expo
testing: a physical phone must call the Mac's LAN IP, not `127.0.0.1`.

Set the matching URL in `frontend/mobile/api.js`:

```js
const API_BASE = "http://<mac-lan-ip>:8000";
```

Then run Expo:

```bash
cd frontend/mobile
npx expo start
```

If the mobile app shows `Network Error`, check:

- backend log says `Uvicorn running on http://0.0.0.0:8000`
- phone and Mac are on the same Wi-Fi
- `frontend/mobile/api.js` uses the Mac LAN IP
- macOS Firewall allows inbound connections to Python/uvicorn
