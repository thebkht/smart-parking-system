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
STAGE1_VARIANT ?= s
STAGE2_VARIANT ?= n
STAGE1_WEIGHTS ?= runs/stage1_det/yolov8$(STAGE1_VARIANT)_stage1/weights/best.pt
STAGE2_WEIGHTS ?= runs/stage2_cls/yolov8$(STAGE2_VARIANT)_stage2/weights/best.pt
BENCHMARK_IMAGE ?= samples/demo.jpg
BENCHMARK_ROI ?= 50 100 200 250
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
EDGE_ARGS ?=
PREDICT_SOURCE ?= samples/demo.jpg
PREP_STAGE1_ARGS ?=
PREP_STAGE2_ARGS ?=
PREP_SINGLE_MODEL_ARGS ?=
TRAIN_STAGE1_ARGS ?=
TRAIN_STAGE2_ARGS ?=
EVALUATE_STAGE1_ARGS ?=
EVALUATE_STAGE2_ARGS ?=
COMPARE_STAGE2_ARGS ?=
SWEEP_STAGE2_ARGS ?=
EXPORT_STAGE2_ARGS ?=
STABILITY_DURATION ?= 1800
STABILITY_ARGS ?=

.PHONY: venv install install-dev check-python \
	prepare-stage1 prepare-stage2 prepare-single-model \
	train-stage1 train-stage2 train-stage2-all \
	evaluate-stage1 evaluate-stage2 compare-stage2 sweep-stage2 \
	export-stage2 benchmark-stage2 bandwidth stability test lint finalize \
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
	$(VENV_PYTHON) ml/prepare_dataset.py --stage2 --pklot-dir $(PKLOT_DIR) $(if $(CNRPARK_DIR),--cnrpark-dir $(CNRPARK_DIR),) $(PREP_STAGE2_ARGS)

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
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights $(STAGE2_WEIGHTS) --split val --device $(DEVICE) $(EVALUATE_STAGE2_ARGS)

compare-stage2:
	$(VENV_PYTHON) ml/evaluate.py --stage2 --split val --device $(DEVICE) --compare \
		runs/stage2_cls/yolov8n_stage2/weights/best.pt \
		runs/stage2_cls/yolov8s_stage2/weights/best.pt \
		runs/stage2_cls/yolov8m_stage2/weights/best.pt \
		$(COMPARE_STAGE2_ARGS)

sweep-stage2:
	$(VENV_PYTHON) ml/evaluate.py --stage2 --weights $(STAGE2_WEIGHTS) --split val --device $(DEVICE) --sweep $(SWEEP_STAGE2_ARGS)

export-stage2:
	$(VENV_PYTHON) ml/export.py --weights $(STAGE2_WEIGHTS) --imgsz 64 $(EXPORT_STAGE2_ARGS)

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
