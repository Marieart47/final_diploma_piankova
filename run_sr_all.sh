#!/usr/bin/env bash
set -e
cd /Users/maria/Documents/diplom_v2
source venv/bin/activate
export PYTHONUNBUFFERED=1

TRAIN_DIR="data/sr_hr/stl10/train"
VAL_DIR="data/sr_hr/stl10/val"
DATASET="stl10"
SCALE=2
PATCH=64
ARCHS="srcnn,espcn,swinir"
STRATS="baseline,stagewise,static,online"
STEPS=3000
BS=32
PATIENCE=10

run() {
  local DEG=$1; local LVL=$2
  echo ""; echo "============================================================"
  echo " DEGRADATION: $DEG  level=$LVL"
  echo "============================================================"
  python -u run_sr.py \
    --hr_train_dir "$TRAIN_DIR" --hr_val_dir "$VAL_DIR" \
    --dataset_name "$DATASET" --scale $SCALE --patch_size $PATCH \
    --degradation "$DEG" --deg_level $LVL \
    --archs "$ARCHS" --strategies "$STRATS" \
    --total_steps $STEPS --batch_size $BS --patience $PATIENCE
}

run original 0.0
run noise    0.5
run blur     0.5
run jpeg     0.5

echo ""; echo "=== ALL EXPERIMENTS DONE ==="
