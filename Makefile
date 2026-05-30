PYTHON ?= python3
VENV ?= .venv

UNAME_S := $(shell uname -s 2>/dev/null)

ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV)/Scripts/python.exe
DEVICE ?= cpu
else
VENV_PYTHON := $(VENV)/bin/python
ifeq ($(UNAME_S),Darwin)
DEVICE ?= mps
else
DEVICE ?= cpu
endif
endif

PKLOT_DIR ?= datasets/pklot_raw
CNRPARK_DIR ?=
ACPDS_ROOT ?= datasets/acpds_raw
ACPDS_MANIFEST ?=
ACPDS_STAGE2_DIR ?= datasets/acpds_stage2
STAGE1_VARIANT ?= s
STAGE2_VARIANT ?= n
STAGE1_WEIGHTS ?= runs/stage1_det/yolov8$(STAGE1_VARIANT)_stage1/weights/best.pt
STAGE2_WEIGHTS ?= acpds_cls/weights/best.pt
STAGE2_SPLIT ?= val
BENCHMARK_IMAGE ?= samples/demo.jpg
BENCHMARK_ROI ?= 50 100 200 250
BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 8000
EDGE_ARGS ?=
PREDICT_SOURCE ?= samples/demo.jpg
PREP_STAGE1_ARGS ?=
PREP_STAGE2_ARGS ?=
PREP_SINGLE_MODEL_ARGS ?=
VALIDATE_STAGE2_ARGS ?=
TRAIN_STAGE1_ARGS ?=
TRAIN_STAGE2_ARGS ?=
EVALUATE_STAGE1_ARGS ?=
EVALUATE_STAGE2_ARGS ?=
COMPARE_STAGE2_ARGS ?=
SWEEP_STAGE2_ARGS ?=
EXPORT_STAGE2_ARGS ?=
EXPORT_EVAL_DEVICE ?= cpu
LOCALIZE_ARGS ?=
WEEK6_LOG_DIR ?= logs/week6
WEEK7_LOG_DIR ?= logs/week7
STABILITY_DURATION ?= 1800
STABILITY_ARGS ?=

.PHONY: venv install install-dev check-python \
	prepare-stage1 prepare-stage2 validate-stage2 prepare-single-model layout-sample \
	train-stage1 train-stage2 train-stage2-all \
	evaluate-stage1 evaluate-stage2 compare-stage2 sweep-stage2 \
	export-stage2 benchmark-stage2 bandwidth stability test lint finalize \
	localize-car week6-stage2 week6-export week7-eval \
	backend edge predict

check-python:
	@$(PYTHON) --version

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

install-dev: install

prepare-stage1:
	$(VENV_PYTHON) ml/prepare_dataset.py --stage1 --pklot-dir $(PKLOT_DIR) $(PREP_STAGE1_ARGS)

prepare-stage2:
	$(VENV_PYTHON) ml/extract_patches.py --dataset-root $(ACPDS_ROOT) --output $(ACPDS_STAGE2_DIR) $(if $(ACPDS_MANIFEST),--manifest $(ACPDS_MANIFEST),) $(PREP_STAGE2_ARGS)

validate-stage2:
	$(VENV_PYTHON) ml/extract_patches.py --dataset-root $(ACPDS_ROOT) --output $(ACPDS_STAGE2_DIR) $(if $(ACPDS_MANIFEST),--manifest $(ACPDS_MANIFEST),) --validate-only $(VALIDATE_STAGE2_ARGS)

prepare-single-model:
	$(VENV_PYTHON) ml/prepare_dataset.py --single-model --pklot-dir $(PKLOT_DIR) $(PREP_SINGLE_MODEL_ARGS)

train-stage1:
	$(VENV_PYTHON) ml/train.py --stage1 --variant $(STAGE1_VARIANT) --device $(DEVICE) $(TRAIN_STAGE1_ARGS)

train-stage2:
	$(VENV_PYTHON) ml/train.py --stage2 --variant $(STAGE2_VARIANT) --device $(DEVICE) $(TRAIN_STAGE2_ARGS)

train-stage2-all:
	$(VENV_PYTHON) ml/train.py --stage2 --variant n --device $(DEVICE)
	$(VENV_PYTHON) ml/train.py --stage2 --variant s --device $(DEVICE)
	$(VENV_PYTHON) ml/train.py --stage2 --variant m --device $(DEVICE)

evaluate-stage1:
	$(VENV_PYTHON) ml/evaluate.py --stage1 --weights $(STAGE1_WEIGHTS) --split val --device $(DEVICE) $(EVALUATE_STAGE1_ARGS)

evaluate-stage2:
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights $(STAGE2_WEIGHTS) --split $(STAGE2_SPLIT) --device $(DEVICE) $(EVALUATE_STAGE2_ARGS)

compare-stage2:
	$(VENV_PYTHON) ml/evaluate.py --stage2 --split $(STAGE2_SPLIT) --device $(DEVICE) --compare \
		runs/acpds_cls/yolov8n_stage2/weights/best.pt \
		runs/acpds_cls/yolov8s_stage2/weights/best.pt \
		runs/acpds_cls/yolov8m_stage2/weights/best.pt \
		$(COMPARE_STAGE2_ARGS)

sweep-stage2:
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights $(STAGE2_WEIGHTS) --split val --device $(DEVICE) --sweep $(SWEEP_STAGE2_ARGS)

export-stage2:
	$(VENV_PYTHON) ml/export.py --weights $(STAGE2_WEIGHTS) --imgsz 128 $(EXPORT_STAGE2_ARGS)

localize-car:
	$(VENV_PYTHON) ml/localize.py $(LOCALIZE_ARGS)

week6-stage2:
	$(VENV_PYTHON) ml/train.py --stage2 --variant s --device $(DEVICE) $(TRAIN_STAGE2_ARGS)
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights runs/acpds_cls/yolov8s_stage2/weights/best.pt --split val --device $(DEVICE) --output-json $(WEEK6_LOG_DIR)/stage2_s_val.json $(EVALUATE_STAGE2_ARGS)
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights runs/acpds_cls/yolov8s_stage2/weights/best.pt --split test --device $(DEVICE) --output-json $(WEEK6_LOG_DIR)/stage2_s_test.json $(EVALUATE_STAGE2_ARGS)
	$(VENV_PYTHON) ml/train.py --stage2 --variant m --device $(DEVICE) $(TRAIN_STAGE2_ARGS)
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights runs/acpds_cls/yolov8m_stage2/weights/best.pt --split val --device $(DEVICE) --output-json $(WEEK6_LOG_DIR)/stage2_m_val.json $(EVALUATE_STAGE2_ARGS)
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights runs/acpds_cls/yolov8m_stage2/weights/best.pt --split test --device $(DEVICE) --output-json $(WEEK6_LOG_DIR)/stage2_m_test.json $(EVALUATE_STAGE2_ARGS)
	$(VENV_PYTHON) ml/evaluate.py --stage2 --split val --device $(DEVICE) --output-json $(WEEK6_LOG_DIR)/stage2_compare_val.json --compare \
		acpds_cls/weights/best.pt \
		runs/acpds_cls/yolov8s_stage2/weights/best.pt \
		runs/acpds_cls/yolov8m_stage2/weights/best.pt \
		$(COMPARE_STAGE2_ARGS)
	$(VENV_PYTHON) ml/evaluate.py --stage2 --split test --device $(DEVICE) --output-json $(WEEK6_LOG_DIR)/stage2_compare_test.json --compare \
		acpds_cls/weights/best.pt \
		runs/acpds_cls/yolov8s_stage2/weights/best.pt \
		runs/acpds_cls/yolov8m_stage2/weights/best.pt \
		$(COMPARE_STAGE2_ARGS)

week6-export:
	$(VENV_PYTHON) ml/export.py --weights acpds_cls/weights/best.pt --imgsz 128 --summary-json artifacts/models/export_summary.json $(EXPORT_STAGE2_ARGS)
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights artifacts/models/best.onnx --split val --device $(EXPORT_EVAL_DEVICE) --output-json $(WEEK6_LOG_DIR)/stage2_export_onnx_val.json $(EVALUATE_STAGE2_ARGS)
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights artifacts/models/best.onnx --split test --device $(EXPORT_EVAL_DEVICE) --output-json $(WEEK6_LOG_DIR)/stage2_export_onnx_test.json $(EVALUATE_STAGE2_ARGS)

week7-eval:
	$(VENV_PYTHON) ml/analyze_generalization.py --output $(WEEK7_LOG_DIR)/val_test_gap.json
	$(VENV_PYTHON) ml/bucket_acpds_weather.py --summary-json $(WEEK7_LOG_DIR)/acpds_weather_buckets.json
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights acpds_cls/weights/best.pt \
		--data datasets/acpds_stage2_weather --per-weather \
		--weather-labels sunny,overcast,low_light \
		--device $(DEVICE) --output-json $(WEEK7_LOG_DIR)/stage2_acpds_weather.json

benchmark-stage2:
	$(VENV_PYTHON) edge/benchmark.py \
		--task classify \
		--image $(BENCHMARK_IMAGE) \
		--model $(STAGE2_WEIGHTS) \
		--imgsz 64 \
		--roi $(BENCHMARK_ROI)

bandwidth:
	$(VENV_PYTHON) ml/bandwidth.py

backend:
	$(VENV_PYTHON) -m uvicorn backend.main:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT)

edge:
	$(VENV_PYTHON) edge/detect.py $(EDGE_ARGS)

predict:
	$(VENV_PYTHON) ml/predict.py --weights $(STAGE2_WEIGHTS) --source $(PREDICT_SOURCE)

layout-sample:
	$(VENV_PYTHON) ml/sfm_layout.py --images samples --output artifacts/layout_sample

stability:
	$(VENV_PYTHON) edge/stability_test.py \
		--image $(BENCHMARK_IMAGE) \
		--stage1-detector \
		--stage1-model $(STAGE1_WEIGHTS) \
		--stage2-model $(STAGE2_WEIGHTS) \
		--duration $(STABILITY_DURATION) \
		$(STABILITY_ARGS)

finalize:
	$(VENV_PYTHON) ml/finalize.py

test:
	$(VENV_PYTHON) -m pytest -q

lint:
	$(VENV_PYTHON) -m ruff check .
	$(VENV_PYTHON) -m black --check .
