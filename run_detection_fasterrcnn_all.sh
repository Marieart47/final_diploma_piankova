#!/usr/bin/env bash
# Эксперименты Faster R-CNN / VOC — зеркалит run_detection_all.sh.
# Те же условия: 3000 шагов, 4 стратегии.

set -e

STEPS=3000
BATCH=4
STRATS="baseline,stagewise,static,online"

echo "===== Detection experiments: VOC / Faster R-CNN ====="
echo "Steps: $STEPS | Strategies: $STRATS"
echo ""

source venv/bin/activate
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

run() {
  python -u run_detection_fasterrcnn.py \
    --strategies $STRATS --total_steps $STEPS --batch $BATCH --workers 0 --device cpu "$@"
}

echo ">> original"
run --dataset_type original

echo ">> noisy noise=0.3"
run --dataset_type noisy --noise_level 0.3

echo ">> noisy noise=0.6"
run --dataset_type noisy --noise_level 0.6

echo ">> imbalance factor=0.3"
run --dataset_type imbalance --imbalance_factor 0.3

echo ">> imbalance factor=0.6"
run --dataset_type imbalance --imbalance_factor 0.6

echo ">> artifacts blur level=0.4"
run --dataset_type artifacts --artifact_type blur --artifact_level 0.4

echo ">> artifacts noise level=0.4"
run --dataset_type artifacts --artifact_type noise --artifact_level 0.4

echo ""
echo "===== Faster R-CNN experiments done ====="
