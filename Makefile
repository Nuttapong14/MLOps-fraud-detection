# Override with your GCP project: make cluster-create PROJECT_ID=my-project
# Or: export PROJECT_ID=my-project before running make
PROJECT_ID ?= $(shell grep -s PROJECT_ID .env | cut -d= -f2)
REGION     := asia-southeast1
CLUSTER    := mlops-cluster
GCS_BUCKET := $(PROJECT_ID)-mlops-data
AR_REPO    := mlops
IMAGE_TAG  := $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")

.PHONY: help setup-gcp cluster-create cluster-delete \
        install .venv/bin/dvc-init .venv/bin/dvc-pull data-download \
        pipeline-run train serve lint test

help:
	@echo "Phase 0: Foundation"
	@echo "  make setup-gcp       — Enable APIs + create GCS bucket"
	@echo "  make cluster-create  — Create GKE Autopilot cluster"
	@echo "  make cluster-delete  — Delete GKE cluster (save cost)"
	@echo "  make install         — Install Python deps with uv"
	@echo "  make .venv/bin/dvc-init        — Init DVC with GCS remote"
	@echo "  make data-download   — Download Kaggle dataset"
	@echo ""
	@echo "Phase 1+: Pipeline"
	@echo "  make pipeline-run    — Run full DVC pipeline"
	@echo "  make train           — Train model only"
	@echo "  make lint            — Run ruff linter"
	@echo "  make test            — Run pytest"

# ── GCP Setup ─────────────────────────────────────────────────────────────────

setup-gcp:
	gcloud services enable \
	  container.googleapis.com \
	  artifactregistry.googleapis.com \
	  storage.googleapis.com \
	  sqladmin.googleapis.com \
	  redis.googleapis.com \
	  --project $(PROJECT_ID)
	gsutil mb -l $(REGION) gs://$(GCS_BUCKET) 2>/dev/null || echo "Bucket already exists"
	gcloud artifacts repositories create $(AR_REPO) \
	  --repository-format=docker \
	  --location=$(REGION) \
	  --project=$(PROJECT_ID) 2>/dev/null || echo "Repo already exists"
	@echo "✓ GCP setup complete"

cluster-create:
	gcloud container clusters create-auto $(CLUSTER) \
	  --region $(REGION) \
	  --project $(PROJECT_ID)
	gcloud container clusters get-credentials $(CLUSTER) \
	  --region $(REGION) \
	  --project $(PROJECT_ID)
	@echo "✓ Cluster ready"

cluster-delete:
	@echo "⚠ This will delete the cluster and stop billing for compute"
	gcloud container clusters delete $(CLUSTER) \
	  --region $(REGION) \
	  --project $(PROJECT_ID) \
	  --quiet

# ── Local Dev ──────────────────────────────────────────────────────────────────

install:
	uv sync --all-extras  # creates .venv/ automatically

.venv/bin/dvc-init:
	.venv/bin/dvc init
	.venv/bin/dvc remote add -d gcs gs://$(GCS_BUCKET)/.venv/bin/dvc-cache
	.venv/bin/dvc remote modify gcs version_aware true
	@echo "✓ DVC initialized with GCS remote gs://$(GCS_BUCKET)/.venv/bin/dvc-cache"

data-download:
	@echo "Downloading Kaggle Credit Card Fraud dataset..."
	kaggle datasets download mlg-ulb/creditcardfraud -p data/raw/ --unzip
	.venv/bin/dvc add data/raw/creditcard.csv
	@echo "✓ Dataset added and tracked by DVC"

# ── Pipeline ───────────────────────────────────────────────────────────────────

pipeline-run:
	.venv/bin/dvc repro

train:
	.venv/bin/python src/training/train.py

# ── Quality ────────────────────────────────────────────────────────────────────

lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/ruff format --check src/ tests/

test:
	.venv/bin/pytest tests/ -v

test-ci:
	.venv/bin/pytest tests/unit/ -v  # Only unit tests in CI (no GCP creds needed)
