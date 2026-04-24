#!/bin/zsh
# Full detection experiment suite on PASCAL VOC with YOLOv8n.
# Mirrors run_all.sh structure: 4 difficulty types × 4 strategies.
# Budget: 3000 gradient steps per run (same as classification for fair comparison).

set -e  # stop on first error

STEPS=3000
MODEL=yolov8n
BATCH=8
STRATS="baseline,stagewise,static,online"
DATA=VOC.yaml

echo "===== Detection experiments: VOC / YOLOv8n ====="
echo "Steps per run: $STEPS | Strategies: $STRATS"
echo ""

# ── Original (clean dataset) ────────────────────────────────────────────────────
echo ">> original"
python3 -u run_detection.py \
  --data $DATA --dataset_type original \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

# ── Noisy labels ────────────────────────────────────────────────────────────────
echo ">> noisy noise=0.3"
python3 -u run_detection.py \
  --data $DATA --dataset_type noisy --noise_level 0.3 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

echo ">> noisy noise=0.6"
python3 -u run_detection.py \
  --data $DATA --dataset_type noisy --noise_level 0.6 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

# ── Class imbalance ─────────────────────────────────────────────────────────────
echo ">> imbalance factor=0.3"
python3 -u run_detection.py \
  --data $DATA --dataset_type imbalance --imbalance_factor 0.3 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

echo ">> imbalance factor=0.6"
python3 -u run_detection.py \
  --data $DATA --dataset_type imbalance --imbalance_factor 0.6 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

# ── Visual artifacts ────────────────────────────────────────────────────────────
echo ">> artifacts blur level=0.4"
python3 -u run_detection.py \
  --data $DATA --dataset_type artifacts --artifact_type blur --artifact_level 0.4 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

echo ">> artifacts noise level=0.4"
python3 -u run_detection.py \
  --data $DATA --dataset_type artifacts --artifact_type noise --artifact_level 0.4 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps

echo ""
echo "===== All detection experiments done ====="
