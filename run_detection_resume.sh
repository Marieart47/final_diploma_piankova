#!/bin/zsh
# Resume detection experiments from where run_detection_all.sh failed.
# original/baseline and original/stagewise already completed.
# Restarts from original/static onwards.

set -e

STEPS=3000
MODEL=yolov8n
BATCH=8
DATA=VOC.yaml

echo "===== Detection experiments (resume): VOC / YOLOv8n ====="
echo "Steps per run: $STEPS"
echo ""

# ── Original: only remaining strategies ─────────────────────────────────────
echo ">> original (static + online only)"
python3 -u run_detection.py \
  --data $DATA --dataset_type original \
  --model $MODEL --strategies static,online \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

# ── Noisy labels ─────────────────────────────────────────────────────────────
echo ">> noisy noise=0.3"
python3 -u run_detection.py \
  --data $DATA --dataset_type noisy --noise_level 0.3 \
  --model $MODEL --strategies baseline,stagewise,static,online \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

echo ">> noisy noise=0.6"
python3 -u run_detection.py \
  --data $DATA --dataset_type noisy --noise_level 0.6 \
  --model $MODEL --strategies baseline,stagewise,static,online \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

# ── Class imbalance ──────────────────────────────────────────────────────────
echo ">> imbalance factor=0.3"
python3 -u run_detection.py \
  --data $DATA --dataset_type imbalance --imbalance_factor 0.3 \
  --model $MODEL --strategies baseline,stagewise,static,online \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

echo ">> imbalance factor=0.6"
python3 -u run_detection.py \
  --data $DATA --dataset_type imbalance --imbalance_factor 0.6 \
  --model $MODEL --strategies baseline,stagewise,static,online \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

# ── Visual artifacts ─────────────────────────────────────────────────────────
echo ">> artifacts blur level=0.4"
python3 -u run_detection.py \
  --data $DATA --dataset_type artifacts --artifact_type blur --artifact_level 0.4 \
  --model $MODEL --strategies baseline,stagewise,static,online \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

echo ">> artifacts noise level=0.4"
python3 -u run_detection.py \
  --data $DATA --dataset_type artifacts --artifact_type noise --artifact_level 0.4 \
  --model $MODEL --strategies baseline,stagewise,static,online \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

echo ""
echo "===== All detection experiments done ====="
