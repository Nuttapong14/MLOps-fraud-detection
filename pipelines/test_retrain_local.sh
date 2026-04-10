#!/bin/bash
# Local simulation of the Argo retrain pipeline (no GKE needed)
# Runs each DAG step sequentially to verify the full pipeline works end-to-end.
#
# Usage: bash pipelines/test_retrain_local.sh

set -e  # exit on first error
PYTHON=".venv/bin/python"
LOG_PREFIX="[retrain-test]"

echo "$LOG_PREFIX ══════════════════════════════════════════"
echo "$LOG_PREFIX  Fraud Detector — Local Retrain Simulation"
echo "$LOG_PREFIX ══════════════════════════════════════════"

# Step 1: Data validation (mirrors 'validate-data' DAG step)
echo ""
echo "$LOG_PREFIX [1/4] Validating data..."
$PYTHON src/features/validate.py
echo "$LOG_PREFIX ✓ Data validation passed"

# Step 2: Training (mirrors 'train-model' DAG step)
echo ""
echo "$LOG_PREFIX [2/4] Training model (n_trials from params.yaml)..."
$PYTHON src/training/train.py --experiment-name "retrain-local-$(date +%Y%m%d)"
echo "$LOG_PREFIX ✓ Training complete"

# Step 3: Evaluate (mirrors 'evaluate-model' DAG step)
echo ""
echo "$LOG_PREFIX [3/4] Evaluating on test set..."
$PYTHON src/training/evaluate.py
PASSED=$(python3 -c "import json; r=json.load(open('data/metrics/eval_report.json')); print(r['passed_threshold'])")
ROC=$(python3 -c "import json; r=json.load(open('data/metrics/eval_report.json')); print(f\"{r['test_roc_auc']:.4f}\")")
echo "$LOG_PREFIX ✓ Evaluation: passed=$PASSED  roc_auc=$ROC"

# Step 4: Register if passed (mirrors 'deploy-if-better' DAG condition)
echo ""
if [ "$PASSED" = "True" ]; then
    echo "$LOG_PREFIX [4/4] Model passed threshold — registering as Production..."
    $PYTHON src/training/register.py
    echo "$LOG_PREFIX ✓ Model promoted to Production"
else
    echo "$LOG_PREFIX [4/4] Model did NOT pass threshold (roc_auc=$ROC < 0.90)"
    echo "$LOG_PREFIX ✗ Skipping registration — retrain with more data or tune hyperparams"
    exit 1
fi

echo ""
echo "$LOG_PREFIX ══════════════════════════════════════════"
echo "$LOG_PREFIX  ✓ Full retrain pipeline simulation PASSED"
echo "$LOG_PREFIX    Check MLflow UI: http://localhost:5000"
echo "$LOG_PREFIX ══════════════════════════════════════════"
