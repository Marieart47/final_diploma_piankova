#!/bin/zsh
# Detection experiment suite v2 — with bug fixes:
#   1. Static stage calculation excludes warm-up steps (was jumping to hardest group immediately)
#   2. Loss initialization is random (was ordered by index)
#   3. val_box_loss / val_cls_loss columns removed from CSV (were always 0)
#
# Results saved to results/voc_*_yolov8n_v2/ to keep old results intact.

set -e

STEPS=3000
MODEL=yolov8n
BATCH=8
STRATS="baseline,stagewise,static,online"
DATA=VOC.yaml
SUFFIX=v2

echo "===== Detection experiments v2: VOC / YOLOv8n ====="
echo "Steps per run: $STEPS | Strategies: $STRATS | Suffix: $SUFFIX"
echo ""

# ── Original (clean dataset) ─────────────────────────────────────────────────
echo ">> original"
python3 -u run_detection.py \
  --data $DATA --dataset_type original \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps \
  --exp_suffix $SUFFIX

# ── Noisy labels ─────────────────────────────────────────────────────────────
echo ">> noisy noise=0.3"
python3 -u run_detection.py \
  --data $DATA --dataset_type noisy --noise_level 0.3 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps \
  --exp_suffix $SUFFIX

echo ">> noisy noise=0.6"
python3 -u run_detection.py \
  --data $DATA --dataset_type noisy --noise_level 0.6 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps \
  --exp_suffix $SUFFIX

# ── Class imbalance ──────────────────────────────────────────────────────────
echo ">> imbalance factor=0.3"
python3 -u run_detection.py \
  --data $DATA --dataset_type imbalance --imbalance_factor 0.3 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps \
  --exp_suffix $SUFFIX

echo ">> imbalance factor=0.6"
python3 -u run_detection.py \
  --data $DATA --dataset_type imbalance --imbalance_factor 0.6 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps \
  --exp_suffix $SUFFIX

# ── Visual artifacts ─────────────────────────────────────────────────────────
echo ">> artifacts blur level=0.4"
python3 -u run_detection.py \
  --data $DATA --dataset_type artifacts --artifact_type blur --artifact_level 0.4 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps \
  --exp_suffix $SUFFIX

echo ">> artifacts noise level=0.4"
python3 -u run_detection.py \
  --data $DATA --dataset_type artifacts --artifact_type noise --artifact_level 0.4 \
  --model $MODEL --strategies $STRATS \
  --batch $BATCH --total_steps $STEPS --budget_mode steps \
  --exp_suffix $SUFFIX

echo ""
echo "===== All detection experiments v2 done ====="
