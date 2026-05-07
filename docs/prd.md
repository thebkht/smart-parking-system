**Smart Parking System**

Full Project PRD v6

Two-Stage Edge Inference  ·  ACPDS  ·  Quadrilateral Pooling  ·  Find My Car  ·  App

Intelligent Edge Computing  ·  Weeks 3–8  ·  4-Person Team

| Project | Smart Parking System — Two-Stage \+ ACPDS \+ Quad Pooling \+ Find My Car \+ App |
| :---- | :---- |
| **Version** | v6 — adds quadrilateral pooling (method a), corner ordering fix, pooling comparison |
| **Duration** | 6 weeks (Week 3 → Week 8\) |
| **Current week** | Week 4 — Dataset \+ ML model presentation |
| **Edge device** | MacBook (Apple Silicon MPS / Intel CPU) |
| **Stage 1** | ACPDS quadrilateral annotations as ROIs; SfM for new cameras |
| **Stage 2** | YOLOv8n-cls — occupied vs free per 128×128 warped patch |
| **Pooling method** | Quadrilateral pooling (ACPDS paper method a) — getPerspectiveTransform \+ warpPerspective |
| **Corner ordering** | Clockwise TL→TR→BR→BL enforced via order\_corners() before every warp |
| **Primary dataset** | ACPDS (arXiv 2107.12207) — MIT license, 293 images, 11,236 spot annotations |
| **Baseline target** | 98% on unseen lots (ACPDS paper ResNet50 baseline) |
| **Backend** | FastAPI \+ SQLite — 7 endpoints |
| **Frontend** | React (web) \+ React Native (mobile) — 3 screens |
| **Team split** | ML team (A \+ B)  ·  Edge \+ App team (C \+ D) |
| **Final deadline** | Week 8 — technical report (email) \+ presentation (class) |

# **1\. Problem statement**

Existing parking detection approaches either require expensive per-spot hardware sensors, or stream raw video to a cloud server for processing — raising bandwidth costs and privacy concerns. Prior datasets (PKLot, CNRPark-EXT) use same-lot train/test splits that inflate accuracy. The professor has also requested a Find My Car feature requiring a web/mobile app.

| Problem | Impact |
| :---- | :---- |
| No real-time occupancy data | Drivers waste time and fuel circling |
| Cloud video streaming | High bandwidth \+ privacy risk |
| Per-spot sensor hardware | Too expensive for small or temporary lots |
| PKLot / CNRPark same-lot splits | Accuracy overestimated — not real generalization |
| Rectangular patch crops | Include pixels from neighboring spots — degrades accuracy (ACPDS paper Fig. 4b) |
| No car-finding feature | Drivers lose track of spot in large lots |
| Manual ROI definition | Tedious, not scalable across camera angles |

# **2\. Why two stages**

ACPDS provides full parking lot images annotated with quadrilateral polygons per spot. Stage 1 uses these polygons to extract each spot region; Stage 2 classifies the warped patch. The quadrilateral warp is the key technical decision: it ensures each patch contains only one spot's pixels, corrects perspective distortion, and matches the exact pooling method from the ACPDS paper.

| Stage | Task | Dataset | Method |
| :---- | :---- | :---- | :---- |
| **Stage 1** | Locate spot regions | ACPDS quadrilateral annotations | order\_corners() → getPerspectiveTransform → warpPerspective |
| **Stage 2** | Classify each patch | ACPDS — 11,236 unique spot views, unique lots per split | YOLOv8n-cls (128×128 warped patch, binary: occupied / free) |

# **3\. Quadrilateral pooling — the patch extraction method**

The ACPDS paper (Section 4.3, Figure 4\) discusses two ways to pool features for each annotated parking space. That comparison is useful background for this project, but our implementation is not the paper's R-CNN-based model family. This project uses YOLOv8-cls for Stage 2 and adopts method (a): quadrilateral pooling. This section documents the paper's two methods, explains why method (a) is the right fit for this YOLO pipeline, and records the corner-ordering fix required for correct implementation.

## **3.1 The two methods (ACPDS paper Figure 4\)**

|  | Method (a) — quadrilateral | Method (b) — square | Impact |
| :---- | :---- | :---- | :---- |
| **Region used** | Exact perspective quadrilateral matching physical spot edges | Minimum bounding square around the quadrilateral | Quad uses only the spot; square bleeds into neighbors |
| **Pixels included** | Only pixels inside the 4-corner polygon | Includes parts of adjacent spots and road markings | Square adds noise, especially in dense lots |
| **Perspective** | Corrected — warped to a flat top-down 128×128 view | Not corrected — still perspective-distorted | Warp removes angle distortion, making features consistent |
| **Implementation** | getPerspectiveTransform \+ warpPerspective | getRectSubPix or simple array slice | Quad requires corner ordering; square does not |
| **ACPDS result** | Better accuracy, especially with occlusions | Simpler but less accurate | Quad is the recommended method in the paper |

| Why quadrilateral pooling matters for this project |
| :---- |
| The ACPDS dataset captures lots from \~12 m height with strong perspective distortion. |
| Each spot appears as a trapezoid or parallelogram in the image, not a rectangle. |
| A bounding-square crop (method b) always includes pixels from neighboring spots — visible in the paper's Figure 4b as colored overlaps. |
| The perspective warp (method a) maps the exact 4 corners to a 128×128 square, removing distortion and excluding neighboring spots entirely. |
| Square pooling can add surrounding context, which may sometimes help when a spot is heavily occluded or near an image edge, but it also mixes in irrelevant neighboring pixels. |
| This is especially important for crowded ACPDS scenes with heavy occlusions between adjacent cars. |
| For this project's YOLOv8 patch classifier, cleaner per-spot inputs are preferred over extra context, so method (a) remains the chosen approach. |

## **3.2 The corner-ordering problem**

getPerspectiveTransform requires the 4 source points in a consistent order. If ACPDS annotation corners are stored in a different order (e.g., arbitrary click order during labeling), the transform maps wrong corners to wrong destinations — producing twisted or folded patches that look correct in shape but contain the wrong pixels.

| Corner index | Expected (clockwise from TL) | Wrong order result |
| :---- | :---- | :---- |
| **corners\[0\]** | Top-left | Random corner — warp is twisted |
| **corners\[1\]** | Top-right | Patch pixels from wrong region |
| **corners\[2\]** | Bottom-right | Classifier sees garbled image |
| **corners\[3\]** | Bottom-left | Silent failure — no error raised |

| Critical: this is a silent failure |
| :---- |
| OpenCV will not raise an error if corners are in the wrong order. |
| The patch will look like a valid image but contain pixels from neighboring spots or road surface. |
| The classifier will train on corrupted patches and report lower accuracy than expected. |
| Always visualize 10–20 sample patches before starting a training run to catch this early. |

## **3.3 order\_corners() — the fix**

Add order\_corners() as the first step in every patch extraction call. It enforces clockwise TL → TR → BR → BL ordering regardless of how the ACPDS JSON stores the corners.

```python
import numpy as np
import cv2


def order_corners(corners):
    """
    Reorder 4 corner points to clockwise: TL, TR, BR, BL.
    Works regardless of the original storage order in ACPDS JSON.
    """
    pts = np.array(corners, dtype=np.float32)

    # Split into top 2 (smallest y) and bottom 2 (largest y)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2]       # smaller y = higher in image
    bottom = sorted_by_y[2:]    # larger y = lower in image

    # Within each pair, sort by x to get left/right
    tl, tr = top[np.argsort(top[:, 0])]         # left = smaller x
    bl, br = bottom[np.argsort(bottom[:, 0])]   # left = smaller x

    return np.array([tl, tr, br, bl], dtype=np.float32)
    # Result: [top-left, top-right, bottom-right, bottom-left]


def warp_spot(img, corners, size=128):
    """
    Quadrilateral pooling (ACPDS paper method a, Figure 4).
    Warps the perspective quadrilateral to a flat size x size square.
    Pixels come only from inside the spot boundary -- no bleed from neighbors.
    """
    src = order_corners(corners)    # enforce TL, TR, BR, BL
    dst = np.array([                # destination: flat square
        [0, 0],
        [size, 0],
        [size, size],
        [0, size],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size))


# Validation: run this before any training
def validate_patches(data, n=20):
    """
    Visualize n random patches to confirm corner ordering is correct.
    Each patch should show exactly one parking spot, flat and upright.
    Run this before starting a training run.
    """
    import random

    samples = random.sample(data, min(n, len(data)))
    for entry in samples:
        img = cv2.imread(entry["image_path"])
        for spot in entry["spots"][:3]:  # first 3 spots per image
            patch = warp_spot(img, spot["corners"])
            label = "occ" if spot["occupied"] else "free"
            cv2.imshow(f"{entry['id']}_{spot['id']}_{label}", patch)
            cv2.waitKey(300)
    cv2.destroyAllWindows()
```

## **3.4 Full patch extraction pipeline**

Complete script to extract all ACPDS patches into the stage2 folder structure. Uses order\_corners() on every spot before warping.

```python
import cv2
import json
import numpy as np
import os


def order_corners(corners):
    pts = np.array(corners, dtype=np.float32)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top, bottom = sorted_by_y[:2], sorted_by_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_spot(img, corners, size=128):
    src = order_corners(corners)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size))


with open("acpds/annotations.json") as f:
    data = json.load(f)

counts = {
    "train": {"occupied": 0, "free": 0},
    "val": {"occupied": 0, "free": 0},
    "test": {"occupied": 0, "free": 0},
}

for entry in data:
    img = cv2.imread(entry["image_path"])
    split = entry["split"]  # train, val, or test
    for spot in entry["spots"]:
        patch = warp_spot(img, spot["corners"])   # quad pooling, method (a)
        label = "occupied" if spot["occupied"] else "free"
        out_dir = f"acpds_stage2/{split}/{label}"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{entry['id']}_{spot['id']}.jpg"
        cv2.imwrite(path, patch)
        counts[split][label] += 1

# Print dataset statistics
for split, c in counts.items():
    total = c["occupied"] + c["free"]
    print(
        f"{split:6s}: {total:5d} patches  "
        f"occupied={c['occupied']} ({100 * c['occupied'] // total}%)  "
        f"free={c['free']} ({100 * c['free'] // total}%)"
    )
```

## **3.5 Expected folder structure after extraction**

```text
acpds_stage2/
├── train/
│   ├── occupied/   (~2,580 patches, 128x128, perspective-corrected)
│   └── free/       (~2,796 patches, 128x128, perspective-corrected)
├── val/
│   ├── occupied/   (~700 patches, different lots from train)
│   └── free/       (~720 patches)
└── test/
    ├── occupied/   (~700 patches, never-seen lots - the key evaluation set)
    └── free/       (~740 patches)

Total: ~11,236 patches across all splits
Class balance: ~48% occupied / 52% free (near-balanced, no weighting needed)
```

# **4\. Primary dataset — ACPDS**

## **4.1 Why ACPDS over PKLot \+ CNRPark-EXT**

| Property | PKLot \+ CNRPark-EXT | ACPDS (v6 — chosen) |
| :---- | :---- | :---- |
| **Total images** | 12,417 \+ \~300 scene images | 293 full parking lot images |
| **Spot annotations** | 695K \+ 150K pre-cropped patches | 11,236 unique quadrilateral polygons |
| **Train/val/test split** | Same lots across splits | Unique parking lots per split — true generalization |
| **Views** | Fixed overhead cameras | Every image unique view (GoPro on 12 m pole) |
| **Annotation format** | Axis-aligned bounding boxes | Quadrilateral per spot — handles perspective |
| **Patch method** | Simple rectangular crop | Perspective warp from quadrilateral (method a) |
| **Occlusions** | Minimal / moderate | Heavy occlusions — realistic lamppost camera |
| **Generalization test** | Manual cross-dataset split | Built-in: val/test lots never seen in training |
| **Baseline accuracy** | \~98% same-lot (overestimated) | 98% on unseen lots — meaningful benchmark |
| **License** | CC Attribution / research-only | MIT — dataset, code, and trained models |

## **4.2 Dataset details**

| Property | Value |
| :---- | :---- |
| **Paper** | arXiv:2107.12207 — Image-Based Parking Space Occupancy Classification: Dataset and Baseline |
| **Author** | Martin Marek (ParkDots, PosAm), 2021 |
| **License** | MIT — dataset, code, and pretrained models |
| **GitHub** | github.com/martin-marek/parking-space-occupancy |
| **Total images** | 293 full parking lot images (4000×3000 px, GoPro Hero 6 wide FOV) |
| **Spot annotations** | 11,236 unique views of a parking space |
| **Occupied / free** | 5,376 occupied (48%) — near-balanced classes |
| **Train images** | 231 images |
| **Val images** | 35 images (\>1,400 unique spot views, different lots from train) |
| **Test images** | 27 images (\>1,400 unique spot views, never-seen lots) |
| **Camera height** | \~12 m (telescoping pole) — same height as real lampposts |
| **Annotation format** | Quadrilateral per spot (4 corner points) — handles perspective distortion |
| **Pooling method** | Method (a): getPerspectiveTransform \+ warpPerspective to 128×128 |

# **5\. System architecture**

Three parallel data flows share one FastAPI backend and one SQLite database.

| Flow | Data path |
| :---- | :---- |
| **Setup (owner)** | 4–5 photos → Layout AI (SfM \+ BEV) → 2D map JSON \+ quadrilateral polygons → SQLite (layout table) |
| **Inference (edge)** | Webcam frame → Stage 1 (load ACPDS / SfM polygons) → order\_corners() → warpPerspective 128×128 → Stage 2 (YOLOv8n-cls) → temporal smooth → JSON → POST /update |
| **Consumer (app)** | Driver photo → POST /park → SIFT / embedding match → spot\_id → GET /find/{id} → 2D map highlight → app UI |

# **6\. ML models**

## **6.1 Stage 2 — patch classifier**

| Property | Value |
| :---- | :---- |
| **Model** | YOLOv8n-cls (classify mode) |
| **Dataset** | ACPDS — 293 images, 11,236 annotations, MIT license |
| **Task** | Binary classification: occupied vs free |
| **Input** | 128×128 perspective-warped patch (quadrilateral pooling, method a) |
| **Class balance** | \~48% occupied / 52% free — no class weighting needed |
| **Expected accuracy** | \~98% on unseen lots (target: match or beat ResNet50 paper baseline) |
| **Training time** | \~25 min on MPS |
| **Paper baseline** | ResNet50 (25.6M params) — 98% accuracy on unseen lots |
| **Paper weights** | Available at github.com/martin-marek/parking-space-occupancy (MIT) |

## **6.2 Training command**

```bash
yolo classify train \
  model=yolov8n-cls.pt \
  data=acpds_stage2/ \
  epochs=30 \
  imgsz=128 \
  device=mps \
  batch=32

# imgsz=128 matches ACPDS quadrilateral warp output size
# Validate:
yolo classify val model=acpds_cls/weights/best.pt data=acpds_stage2/
```

## **6.3 Model comparison plan**

| Model | Params | Expected acc. | Expected FPS | Input | Size |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **ResNet50 (paper baseline)** | 25.6M | \~98% | — | 128×128 | — |
| YOLOv8n-cls | 2.7M | \~97% | 120–160 | 128×128 | \~5 MB |
| YOLOv8s-cls | 6.4M | \~98% | 70–100 | 128×128 | \~12 MB |
| YOLOv8m-cls | 17M | \~98.5% | 35–55 | 128×128 | \~35 MB |
| YOLOv8n-cls INT8 | 2.7M | \~97% | 200+ | 128×128 | \~1.5 MB |

| Core academic contribution |
| :---- |
| ResNet50 (paper baseline): 25.6M parameters, 98% on unseen lots. |
| YOLOv8n-cls: 2.7M parameters (9× fewer). If it matches 98% accuracy on the same unseen test split, |
| the result is: a 9× smaller model achieving the same generalization — directly relevant for edge deployment. |
| Both models use identical quadrilateral pooling (method a), making the comparison fair. |

## **6.4 Car localization model**

| Property | Value |
| :---- | :---- |
| **Method (primary)** | SIFT feature extraction \+ FLANN matcher (OpenCV, no GPU) |
| **Method (upgrade)** | MobileNetV3 embedding \+ cosine similarity (PyTorch, 2 MB) |
| **Input** | Driver query photo \+ stored reference photos per spot |
| **Output** | Best-match spot\_id \+ similarity score |
| **Training required** | None for SIFT; MobileNetV3 uses pretrained ImageNet weights |

# **7\. Inference pipeline — detect.py**

The complete edge inference pipeline. Key change from v3: rectangular crop replaced by quadrilateral warp using order\_corners() \+ getPerspectiveTransform. Polygon ROIs loaded from GET /map at startup.

```python
import cv2
import requests
import time
import numpy as np
from ultralytics import YOLO
from collections import deque

STAGE2_MODEL = "acpds_cls/weights/best.pt"
API_URL = "http://localhost:8000/update"
INTERVAL = 2      # seconds between inference cycles
SMOOTH_N = 5      # temporal smoothing window

# Load spot polygons from backend
map_data = requests.get("http://localhost:8000/map").json()
SPOT_POLYGONS = {s["spot_id"]: s["corners"] for s in map_data["spots"]}

stage2 = YOLO(STAGE2_MODEL)
history = {}


# Quadrilateral pooling (ACPDS paper method a)
def order_corners(corners):
    pts = np.array(corners, dtype=np.float32)
    sorted_y = pts[np.argsort(pts[:, 1])]
    top, bot = sorted_y[:2], sorted_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bot[np.argsort(bot[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_spot(frame, corners, size=128):
    src = order_corners(corners)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (size, size))


# Classification
def classify_patch(frame, corners):
    patch = warp_spot(frame, corners)
    result = stage2(patch, device="mps", verbose=False)[0]
    return result.names[result.probs.top1], float(result.probs.top1conf)


# Temporal smoothing
def smooth(spot_id, occupied):
    if spot_id not in history:
        history[spot_id] = deque(maxlen=SMOOTH_N)
    history[spot_id].append(occupied)
    return sum(history[spot_id]) > len(history[spot_id]) / 2


# Main loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    status, confidences = {}, {}
    for spot_id, corners in SPOT_POLYGONS.items():
        label, conf = classify_patch(frame, corners)
        smoothed = smooth(spot_id, label == "occupied")
        status[spot_id] = "occupied" if smoothed else "free"
        confidences[spot_id] = round(conf, 3)
    payload = {
        **status,
        "confidence": confidences,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(payload)
    try:
        requests.post(API_URL, json=payload, timeout=2)
    except Exception:
        pass
    time.sleep(INTERVAL)
```

# **8\. Backend API**

| Method | Endpoint | Owner | Description |
| :---- | :---- | :---- | :---- |
| **POST** | /update | Edge device | Submit occupancy JSON from detect.py |
| **GET** | /status | Any client | Latest occupancy by spot\_id |
| **GET** | /history | Any client | Last N occupancy records |
| **POST** | /layout | Lot owner app | 4–5 photos → SfM → store 2D map \+ quadrilateral polygons in DB |
| **GET** | /map | Any client | 2D layout JSON with spot\_id \+ quadrilateral corners array |
| **POST** | /park | Driver app | Driver photo → localization → return session\_id |
| **GET** | /find/{id} | Driver app | Return spot\_id \+ coordinates for session |

The /map endpoint must return corners as an array of \[x, y\] pairs in the same coordinate system as the live camera frame. For the ACPDS evaluation setup, corners are loaded from the ACPDS annotation JSON scaled to the display resolution. For new lot owner cameras, corners come from the SfM pipeline output.

# **9\. Web / Mobile app**

## **9.1 Tech stack**

| Layer | Technology | Rationale |
| :---- | :---- | :---- |
| **Web app** | React \+ Vite | Fast setup, browser demo |
| **Mobile app** | React Native (Expo) | Same JS codebase, native camera access |
| **Map rendering** | Leaflet.js (custom CRS) | Render 2D parking map \+ spot polygon overlays |
| **Backend** | FastAPI \+ SQLite | 7 endpoints |
| **HTTP client** | Axios | Simple REST calls |

## **9.2 Three screens**

* Owner setup: upload 4–5 photos → POST /layout → spinner while SfM runs → show 2D map with quadrilateral overlays → allow label correction.

* Live occupancy map: render 2D layout (GET /map) with spot polygons colored green / red → poll GET /status every 2–5 s → free / occupied count in header.

* Find My Car: camera → POST /park → store session\_id → GET /find/{id} → highlight spot polygon in amber on map.

# **10\. Team structure**

| ML team (A \+ B) | Responsibilities |
| :---- | :---- |
| **Core** | ACPDS download; quadrilateral patch extraction with order\_corners() \+ warpPerspective |
| **Core** | Run validate\_patches() before training; confirm 20 sample warps look correct |
| **Core** | YOLOv8n/s/m-cls training on acpds\_stage2/; target 98% on test split (unseen lots) |
| **Core** | Compare YOLOv8n-cls vs ResNet50 paper baseline — accuracy and parameter count |
| **Core** | Evaluate val accuracy vs test accuracy gap (generalization check) |
| **New** | SfM pipeline (COLMAP / OpenCV) → BEV map \+ quad polygons for new cameras |
| **New** | SIFT car localization; MobileNetV3 embedding (optional upgrade) |

| Edge \+ App team (C \+ D) | Responsibilities |
| :---- | :---- |
| **Core** | detect.py: order\_corners() \+ warpPerspective replacing rectangular crop |
| **Core** | Load quad polygons from GET /map at startup; no hardcoded FIXED\_ROIS |
| **Core** | Temporal smoothing, FPS benchmark (MPS / CPU / ONNX FP32 / ONNX INT8) |
| **Core** | Bandwidth measurement and analysis |
| **New** | React web app: 3 screens (owner setup, live map, Find My Car) |
| **New** | React Native (Expo) wrapper for mobile camera access |
| **New** | Leaflet.js: render quadrilateral polygon overlays per spot; real-time color updates |
| **New** | FastAPI 7 endpoints \+ SQLite schema (layout, spot\_references, park\_sessions tables) |

| Handoff points |
| :---- |
| End of Week 5: ML team delivers acpds\_cls/weights/best.pt \+ sample validated patches to Edge team. |
| End of Week 5: ML team delivers SfM pipeline script \+ BEV map sample to App team. |
| Week 6: Edge team integrates trained model into detect.py; full pipeline end-to-end. |
| Week 7: Car localization tested end-to-end with 20+ real driver photos. |

# **11\. Week-by-week delivery plan**

## **Week 4 — present in class (current)**

| ML team (A \+ B) | Edge \+ App team (C \+ D) |
| :---- | :---- |
| Present ACPDS: unique-lot splits, 11K quad annotations, MIT license Explain quadrilateral pooling (method a vs b, Figure 4\) Explain corner ordering requirement and order\_corners() fix Introduce SfM layout AI \+ Find My Car concept Run live demo with pre-trained YOLOv8n-cls (placeholder) | detect.py: order\_corners() \+ warpPerspective warping Run demo on parking image with ACPDS-style quad polygons Mock FastAPI POST /update endpoint Sketch wireframes for 3 app screens |

## **Week 5**

| ML team (A \+ B) | Edge \+ App team (C \+ D) |
| :---- | :---- |
| Download ACPDS; run quad extraction script Run validate\_patches() — confirm 20 sample warps look correct Build acpds\_stage2/ folder structure Train YOLOv8n-cls → target 98% test accuracy Implement SfM pipeline → deliver BEV map sample Deliver best.pt \+ validated sample patches to Edge team | detect.py: load polygon ROIs from GET /map at startup Add temporal smoothing FPS benchmark on pre-trained classifier (128×128) FastAPI: 7 endpoints stubbed, SQLite schema complete React app scaffold: 3 screens \+ routing |

## **Week 6**

| ML team (A \+ B) | Edge \+ App team (C \+ D) |
| :---- | :---- |
| Train YOLOv8s-cls \+ YOLOv8m-cls on ACPDS INT8 quantization ResNet50 vs YOLOv8 comparison table SIFT car localization (OpenCV) Localization accuracy on sample ACPDS photos | Load trained acpds\_cls/weights/best.pt into detect.py Full pipeline end-to-end running FPS benchmark: MPS / CPU / ONNX FP32 / ONNX INT8 Bandwidth measurement App: live occupancy map screen complete |

## **Week 7**

| ML team (A \+ B) | Edge \+ App team (C \+ D) |
| :---- | :---- |
| Val accuracy vs test accuracy gap analysis Per-weather breakdown (sunny / overcast / low-light) Confusion matrix \+ PR curve Full model comparison table Localization accuracy (20+ real photos) | Find My Car flow: end-to-end test FPS \+ latency table (all backends) Bandwidth savings analysis System stability test (30 min) App: Find My Car screen complete |

## **Week 8 — submit \+ present**

| ML team (A \+ B) | Edge \+ App team (C \+ D) |
| :---- | :---- |
| Complete ML \+ dataset \+ results sections Finalize accuracy tables and figures Present ML findings \+ quad pooling justification | Complete edge \+ system \+ app sections Compile full technical report Run live demo: occupancy \+ Find My Car Submit report via email |

# **12\. Evaluation plan**

## **12.1 Classification accuracy (three splits)**

| Split | Lots | What it measures |
| :---- | :---- | :---- |
| **Train accuracy** | Training lots only | Sanity check / overfitting detection |
| **Val accuracy** | Different lots, same overall dataset | In-distribution generalization |
| **Test accuracy** | Completely unseen parking lots — key result | True generalization — the headline number |

| Model | Train | Val | Test (key) | FPS (MPS) | Size |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **ResNet50 (paper)** | — | — | \~98% | — | — |
| YOLOv8n-cls | TBD | TBD | TBD | TBD | \~5 MB |
| YOLOv8s-cls | TBD | TBD | TBD | TBD | \~12 MB |
| YOLOv8m-cls | TBD | TBD | TBD | TBD | \~35 MB |
| YOLOv8n-cls INT8 | TBD | TBD | TBD | TBD | \~1.5 MB |

## **12.2 Pooling method comparison (optional bonus result)**

If time allows in Week 7, run a direct comparison of the two ACPDS pooling methods on the same model to report the accuracy impact. This matches the experiment in Section 5 of the ACPDS paper.

| Model | Method (a) quad test acc. | Method (b) square test acc. | Accuracy gap |
| :---- | :---- | :---- | :---- |
| **YOLOv8n-cls** | TBD | TBD | TBD |
| **ResNet50 (paper)** | \~98% | Lower (paper reports gap) | See arXiv:2107.12207 Table 2 |

## **12.3 Inference speed**

| Backend | FPS | Latency (ms) |
| :---- | :---- | :---- |
| **MPS (Apple Silicon)** | TBD | TBD |
| **CPU (Intel Mac)** | TBD | TBD |
| **ONNX FP32** | TBD | TBD |
| **ONNX INT8** | TBD | TBD |

## **12.4 Car localization accuracy**

| Metric | Target | Method |
| :---- | :---- | :---- |
| **Top-1 accuracy** | \>85% | Query photo matched to correct spot\_id |
| **Top-3 accuracy** | \>95% | Correct spot in top-3 matches |
| **Avg. match time** | \<500 ms | SIFT extraction \+ FLANN matching on MacBook CPU |
| **Robustness** | Report | Test: daytime, evening, overcast conditions |

# **13\. Deliverables**

| Deliverable | Owner | Week | Status |
| :---- | :---- | :---- | :---- |
| ACPDS downloaded \+ order\_corners() crop script | A+B | W4 | **In progress** |
| Two-stage pipeline \+ quadrilateral pooling slides | A+B | W4 | **In progress** |
| detect.py with order\_corners() \+ warp \+ pre-trained | C+D | W4 | **In progress** |
| Mock FastAPI POST /update | C+D | W4 | **In progress** |
| validate\_patches() confirms correct warps | A+B | W5 | Upcoming |
| acpds\_stage2/ folder structure \+ \~11K patches | A+B | W5 | Upcoming |
| YOLOv8n-cls trained → 98% test accuracy target | A+B | W5 | Upcoming |
| best.pt \+ sample validated patches delivered | A+B | W5 | Upcoming |
| SfM pipeline \+ BEV map sample | A+B | W5 | Upcoming |
| FastAPI 7 endpoints \+ SQLite schema | C+D | W5 | Upcoming |
| React app scaffold (3 screens) | C+D | W5 | Upcoming |
| YOLOv8s \+ YOLOv8m trained; ResNet50 comparison table | A+B | W6 | Upcoming |
| SIFT localization implementation | A+B | W6 | Upcoming |
| Full pipeline end-to-end running | C+D | W6 | Upcoming |
| Live occupancy map screen (quad polygon overlays) | C+D | W6 | Upcoming |
| Pooling method (a) vs (b) comparison (optional W7) | A+B | W7 | Upcoming |
| Val vs test accuracy gap analysis | A+B | W7 | Upcoming |
| Localization accuracy (20+ photos) | A+B | W7 | Upcoming |
| Find My Car flow end-to-end | C+D | W7 | Upcoming |
| System stability test (30 min) | C+D | W7 | Upcoming |
| Technical report — all sections | All | W8 | Upcoming |
| Final presentation \+ live demo | All | W8 | Upcoming |
| Report submitted via email | A or D | W8 | Upcoming |

# **14\. Technical report outline**

| Section | Owner | Content |
| :---- | :---- | :---- |
| **Abstract** | D | 150 words: ACPDS, quadrilateral pooling, 98% unseen-lot target, Find My Car, FPS |
| **1\. Introduction** | A | Parking problem, edge computing, why two stages, Find My Car requirement |
| **2\. Related work** | B | PKLot, CNRPark, ACPDS (arXiv:2107.12207), YOLOv8, SfM / visual localization |
| **3\. Architecture** | C | Three-actor system, three data flows, edge computing justification |
| **4\. Dataset** | A | ACPDS: collection method, unique-lot splits, quad annotations, class balance |
| **5\. Stage 1** | C | order\_corners() \+ warpPerspective; quad vs square pooling comparison; SfM for new cameras |
| **6\. Stage 2** | B | YOLOv8-cls, 128×128 input, training config, curves, accuracy vs ResNet50 |
| **7\. Layout AI** | A+B | SfM pipeline, BEV projection, spot polygon extraction for new cameras |
| **8\. Find My Car** | A+B | SIFT localization, accuracy table, failure analysis |
| **9\. Inference** | C+D | detect.py walkthrough, quad warp, temporal smoothing, map-driven ROIs |
| **10\. App** | C+D | 3 screens, Leaflet.js polygon overlays, real-time updates |
| **11\. Evaluation** | A+B | Train/val/test accuracy, pooling comparison, model comparison, FPS, bandwidth, localization |
| **12\. Discussion** | All | What worked, limitations, production considerations |
| **13\. Conclusion** | D | Contributions, future work |
| **References** | D | arXiv:2107.12207, PKLot, CNRPark, YOLOv8, COLMAP, OpenCV, Leaflet.js |

# **15\. Risks and mitigations**

| Severity | Risk | Mitigation |
| :---- | :---- | :---- |
| **High** | Quad corners in wrong order — twisted patches, silent failure | Always run order\_corners() before getPerspectiveTransform; run validate\_patches() on 20 samples before training |
| **High** | YOLOv8n-cls test accuracy below 95% on unseen lots | Confirm patch quality first; increase epochs to 50; try YOLOv8s-cls; check 48/52 class balance |
| **High** | Demo fails at Week 4 — trained model not ready | Use pre-trained yolov8n-cls.pt with hardcoded ACPDS quad polygons; works today with zero training |
| **Medium** | ACPDS download slow or fails | \~26 MB dataset; download once, share via team Google Drive |
| **Medium** | SfM fails on owner photos | Require \>60% photo overlap; fallback: manual polygon drawing as in v3 |
| **Medium** | SIFT localization accuracy too low | Use better-lit reference photos; try MobileNetV3 embedding; report as metric |
| **Medium** | React Native camera blocked on test device | Fall back to React web with file upload for exam demo |
| **Medium** | ML team finishes late | Edge team uses pre-trained yolov8n-cls.pt \+ hardcoded ACPDS polygons as placeholder |
| **Low** | INT8 quantization accuracy drop | Report as finding — valid academic result |
| **Low** | FastAPI not running during demo | detect.py catches ConnectionError; terminal output is the fallback |

# **16\. Environment setup**

## **16.1 Python / ML**

```bash
pip install ultralytics opencv-python requests fastapi uvicorn
pip install torch torchvision   # for MobileNetV3 localization (optional)
pip install open3d pycolmap     # for SfM layout AI

git clone https://github.com/martin-marek/parking-space-occupancy
cd parking-space-occupancy
# Follow README to download ACPDS dataset (~26 MB)
```

## **16.2 Web / Mobile app**

```bash
npm create vite@latest parking-app -- --template react
cd parking-app && npm install axios leaflet react-leaflet

npx create-expo-app parking-mobile
cd parking-mobile && npx expo install expo-camera axios
```

## **16.3 Verify GPU and patch extraction**

```bash
# Verify MPS backend
python -c "import torch; print(torch.backends.mps.is_available())"

# Run patch extraction
python extract_patches.py
# Expected output:
# train:  5376 patches  occupied=2580 (48%)  free=2796 (52%)
# val:    ~1420 patches  ...
# test:   ~1440 patches  ...  (unseen lots)

# Validate patches visually before training
python -c "from extract_patches import validate_patches; validate_patches(data)"

# Train
yolo classify train model=yolov8n-cls.pt data=acpds_stage2/ imgsz=128 epochs=30 device=mps
```

Smart Parking System — PRD v6  ·  ACPDS \+ Quadrilateral Pooling \+ Find My Car \+ App  ·  Weeks 3–8
