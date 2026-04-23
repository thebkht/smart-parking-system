#!/usr/bin/env python3
"""
Applies --stage1-only mode to edge/detect.py.

Changes:
  1. Adds --stage1-only CLI flag (parse_args)
  2. Adds get_spot_boxes_with_scores() that returns (boxes, scores) from stage1
  3. Patches run_pipeline to skip stage2 when stage1_only=True,
     using stage1 detection confidence directly as occupancy signal
"""

import re, sys, pathlib

TARGET = pathlib.Path("edge/detect.py")
if not TARGET.exists():
    sys.exit(f"ERROR: {TARGET} not found. Run from project root.")

src = TARGET.read_text()

# ── 1. Add --stage1-only flag after --stage1-detector ──────────────────────
OLD_FLAG = '''    parser.add_argument(
        "--stage1-detector",
        action="store_true",
        help="Use the Stage 1 parking-space detector instead of fixed ROIs.",
    )'''

NEW_FLAG = '''    parser.add_argument(
        "--stage1-detector",
        action="store_true",
        help="Use the Stage 1 parking-space detector instead of fixed ROIs.",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help=(
            "Skip Stage 2 classifier entirely. A Stage 1 detection box is "
            "treated as occupied; undetected spots are free. "
            "Useful when Stage 2 domain-shifts badly (e.g. steep overhead angle)."
        ),
    )'''

if "--stage1-only" in src:
    print("--stage1-only flag already present, skipping flag insertion.")
else:
    assert OLD_FLAG in src, "Could not find --stage1-detector block to patch after"
    src = src.replace(OLD_FLAG, NEW_FLAG, 1)
    print("✓ Added --stage1-only CLI flag")

# ── 2. Add get_spot_boxes_with_scores() helper before run_pipeline ──────────
HELPER = '''
def get_spot_boxes_with_scores(
    frame: np.ndarray,
    stage1_model: "YOLO",
    device: str,
    use_sahi: bool = False,
    stage1_imgsz: int = 1280,
    slice_size: int = 640,
    overlap: float = 0.2,
    min_box_area: int = DEFAULT_STAGE1_MIN_BOX_AREA,
    filter_mode: str = DEFAULT_STAGE1_FILTER_MODE,
    postprocess_type: str = DEFAULT_STAGE1_POSTPROCESS_TYPE,
    match_threshold: float = DEFAULT_STAGE1_MATCH_THRESHOLD,
    fixed_rois: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
) -> Tuple[Dict[str, Tuple[int, int, int, int]], Dict[str, float]]:
    """Run stage1 and return (spot_boxes, confidences_per_spot).

    Confidence per spot is the max stage1 score of detections consolidated
    into that spot.  Used by --stage1-only mode to infer occupancy directly.
    """
    yolo_device = device if device != "coreml" else "cpu"
    lot_mask = roi_bounds(fixed_rois) if fixed_rois else None
    roi_boxes_list = list(fixed_rois.values()) if fixed_rois else []

    raw_candidates: list[Tuple[Tuple[int, int, int, int], float]] = []

    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            import tempfile, os as _os

            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=stage1_model.ckpt_path,
                confidence_threshold=DEFAULT_STAGE1_CONFIDENCE,
                device=yolo_device,
            )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            import cv2 as _cv2
            _cv2.imwrite(tmp_path, frame)
            result = get_sliced_prediction(
                tmp_path, sahi_model,
                slice_height=slice_size, slice_width=slice_size,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                verbose=0,
            )
            _os.unlink(tmp_path)
            for pred in result.object_prediction_list:
                b = pred.bbox
                kept = filter_stage1_box(
                    frame.shape,
                    (int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)),
                    lot_mask=lot_mask, roi_boxes=roi_boxes_list,
                    min_box_area=min_box_area, filter_mode=filter_mode,
                )
                if kept is not None:
                    raw_candidates.append((kept, float(getattr(pred.score, "value", 1.0))))
        except ImportError:
            print("SAHI not installed; falling back to standard inference.")
            use_sahi = False

    if not use_sahi:
        results = stage1_model(
            frame, device=yolo_device, verbose=False,
            imgsz=stage1_imgsz, conf=DEFAULT_STAGE1_CONFIDENCE,
        )[0]
        raw_boxes  = results.boxes.xyxy.cpu().numpy()
        raw_scores = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else None
        for idx, box in enumerate(raw_boxes):
            kept = filter_stage1_box(
                frame.shape, tuple(box.astype(int)),
                lot_mask=lot_mask, roi_boxes=roi_boxes_list,
                min_box_area=min_box_area, filter_mode=filter_mode,
            )
            if kept is not None:
                score = float(raw_scores[idx]) if raw_scores is not None else 1.0
                raw_candidates.append((kept, score))

    spot_boxes = consolidate_stage1_boxes(
        raw_candidates,
        postprocess_type=postprocess_type,
        match_threshold=match_threshold,
    )

    # Build per-spot confidence: max raw score whose box overlaps the consolidated box
    spot_scores: Dict[str, float] = {}
    for spot_id, sbox in spot_boxes.items():
        best = 0.0
        for raw_box, raw_score in raw_candidates:
            if box_iou(sbox, raw_box) > 0.05:
                best = max(best, raw_score)
        spot_scores[spot_id] = round(best, 3)

    return spot_boxes, spot_scores

'''

ANCHOR = "def run_pipeline("
if "get_spot_boxes_with_scores" in src:
    print("get_spot_boxes_with_scores already present, skipping helper insertion.")
else:
    assert ANCHOR in src, "Could not find run_pipeline to insert helper before"
    src = src.replace(ANCHOR, HELPER + ANCHOR, 1)
    print("✓ Added get_spot_boxes_with_scores() helper")

# ── 3. Patch run_pipeline to handle stage1_only ─────────────────────────────
OLD_PIPELINE_BODY = '''    spot_boxes = get_spot_boxes(
        frame=frame,
        fixed_rois=fixed_rois,
        stage1_model=stage1_model,
        device=args.device,
        use_stage1_detector=args.stage1_detector,
        use_sahi=args.stage1_sahi,
        stage1_imgsz=args.stage1_imgsz,
        slice_size=args.stage1_slice_size,
        overlap=args.stage1_overlap,
        min_box_area=args.stage1_min_box_area,
        filter_mode=args.stage1_filter_mode,
        postprocess_type=args.stage1_postprocess_type,
        match_threshold=args.stage1_match_threshold,
    )

    raw_statuses: Dict[str, str] = {}
    confidences: Dict[str, float] = {}
    for spot_id, box in sorted(spot_boxes.items()):
        status, confidence = classify_patch(
            frame=frame,
            box=box,
            model=stage2_model,
            device=args.device,
            threshold=args.stage2_threshold,
        )
        raw_statuses[spot_id] = status
        confidences[spot_id] = confidence'''

NEW_PIPELINE_BODY = '''    stage1_only = getattr(args, "stage1_only", False)

    if stage1_only and args.stage1_detector and stage1_model is not None:
        # Skip stage2: treat every stage1 detection as "occupied"
        spot_boxes, spot_scores = get_spot_boxes_with_scores(
            frame=frame,
            stage1_model=stage1_model,
            device=args.device,
            use_sahi=args.stage1_sahi,
            stage1_imgsz=args.stage1_imgsz,
            slice_size=args.stage1_slice_size,
            overlap=args.stage1_overlap,
            min_box_area=args.stage1_min_box_area,
            filter_mode=args.stage1_filter_mode,
            postprocess_type=args.stage1_postprocess_type,
            match_threshold=args.stage1_match_threshold,
            fixed_rois=fixed_rois,
        )
        raw_statuses: Dict[str, str] = {sid: "occupied" for sid in spot_boxes}
        confidences: Dict[str, float] = spot_scores
    else:
        spot_boxes = get_spot_boxes(
            frame=frame,
            fixed_rois=fixed_rois,
            stage1_model=stage1_model,
            device=args.device,
            use_stage1_detector=args.stage1_detector,
            use_sahi=args.stage1_sahi,
            stage1_imgsz=args.stage1_imgsz,
            slice_size=args.stage1_slice_size,
            overlap=args.stage1_overlap,
            min_box_area=args.stage1_min_box_area,
            filter_mode=args.stage1_filter_mode,
            postprocess_type=args.stage1_postprocess_type,
            match_threshold=args.stage1_match_threshold,
        )

        raw_statuses: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        for spot_id, box in sorted(spot_boxes.items()):
            status, confidence = classify_patch(
                frame=frame,
                box=box,
                model=stage2_model,
                device=args.device,
                threshold=args.stage2_threshold,
            )
            raw_statuses[spot_id] = status
            confidences[spot_id] = confidence'''

if "stage1_only = getattr(args" in src:
    print("run_pipeline patch already applied, skipping.")
else:
    assert OLD_PIPELINE_BODY in src, "Could not find run_pipeline body to patch"
    src = src.replace(OLD_PIPELINE_BODY, NEW_PIPELINE_BODY, 1)
    print("✓ Patched run_pipeline with stage1_only branch")

# ── Write out ───────────────────────────────────────────────────────────────
TARGET.write_text(src)
print(f"\n✅ Patch complete → {TARGET}")
print("\nUsage:")
print("  python3 edge/detect.py --image samples \\")
print("    --stage1-detector --stage1-only --no-stage1-sahi \\")
print("    --stage1-model runs/detect/runs/stage1_finetune/yolov8s_stage1/weights/best.pt")