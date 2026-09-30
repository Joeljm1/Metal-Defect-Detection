.PHONY: help install setup-data eda preprocess visualize test benchmark dry-run train-m1 train-m2 train-m3 train-m4 clean

PYTHON ?= uv run python
PYTEST ?= uv run pytest

help:
	@echo "=========================================================================="
	@echo "Metal Surface Defect Detection (CSE411 Task 4 - Team 6)"
	@echo "=========================================================================="
	@echo "Available make commands:"
	@echo "  make install        - Install all project dependencies via uv"
	@echo "  make setup-data     - Verify and setup the NEU-DET dataset"
	@echo "  make eda            - Run Exploratory Data Analysis & generate charts"
	@echo "  make preprocess     - Run CLAHE + Bilateral filtering pipeline on dataset"
	@echo "  make visualize      - Generate side-by-side visual enhancement comparisons"
	@echo "  make test           - Run full test suite with pytest"
	@echo "  make benchmark      - Benchmark inference latency & FPS across M1-M4 models"
	@echo "  make dry-run        - Run sanity training verification on M4 model"
	@echo "  make train-m1       - Train M1 Baseline (YOLOv5s)"
	@echo "  make train-m2       - Train M2 Baseline + CLAHE/Bilateral Preprocessing"
	@echo "  make train-m3       - Train M3 Baseline + ECA/Spatial Attention"
	@echo "  make train-m4       - Train M4 Proposed Integrated Architecture"
	@echo "  make clean          - Remove temporary build, test, and cache files"
	@echo "=========================================================================="

install:
	uv sync --extra dev

setup-data:
	$(PYTHON) scripts/download_dataset.py

eda: setup-data
	$(PYTHON) scripts/run_eda.py --data-dir data/NEU-DET --out-dir reports

preprocess: setup-data
	$(PYTHON) scripts/preprocess_dataset.py --src-dir data/NEU-DET --dst-dir data/preprocessed/NEU-DET

visualize: setup-data
	$(PYTHON) scripts/visualize_preprocessing.py

test:
	$(PYTEST) tests/ -v

benchmark:
	$(PYTHON) scripts/benchmark_fps.py

evaluate:
	$(PYTHON) scripts/evaluate.py --checkpoint checkpoints/M4_Proposed_Integrated_best.pt

pdf-task4:
	typst compile ../CSE411_Project_Task_4_Group_6.typ ../CSE411_Project_Task_4_Group_6.pdf
	cp ../CSE411_Project_Task_4_Group_6.pdf ./CSE411_Project_Task_4_Group_6.pdf

pdf-task5:
	typst compile ../CSE411_Project_Task_5_Group_6.typ ../CSE411_Project_Task_5_Group_6.pdf
	cp ../CSE411_Project_Task_5_Group_6.pdf ./CSE411_Project_Task_5_Group_6.pdf

pdf-task6:
	typst compile ../CSE411_Project_Task_6_Group_6.typ ../CSE411_Project_Task_6_Group_6.pdf
	cp ../CSE411_Project_Task_6_Group_6.pdf ./CSE411_Project_Task_6_Group_6.pdf

pdf-task7:
	typst compile ../CSE411_Project_Task_7_Group_6.typ ../CSE411_Project_Task_7_Group_6.pdf
	cp ../CSE411_Project_Task_7_Group_6.pdf ./CSE411_Project_Task_7_Group_6.pdf

dashboard:
	$(PYTHON) scripts/run_dashboard.py

gc10-eval:
	$(PYTHON) scripts/evaluate_gc10.py

export-onnx:
	$(PYTHON) scripts/export_and_benchmark_onnx.py

gradcam:
	$(PYTHON) scripts/visualize_gradcam.py

train-ablation:
	$(PYTHON) scripts/train_ablation.py

dry-run: setup-data
	$(PYTHON) scripts/train.py --model M4 --dry-run --epochs 1

train-m1: setup-data
	$(PYTHON) scripts/train.py --model M1 --epochs 30 --batch-size 16

train-m2: setup-data
	$(PYTHON) scripts/train.py --model M2 --epochs 30 --batch-size 16

train-m3: setup-data
	$(PYTHON) scripts/train.py --model M3 --epochs 30 --batch-size 16

train-m4: setup-data
	$(PYTHON) scripts/train.py --model M4 --epochs 30 --batch-size 16

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ src/**/__pycache__ tests/__pycache__ scripts/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
