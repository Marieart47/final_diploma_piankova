# Curriculum Learning for Computer Vision

Research into the effect of Curriculum Learning (CL) strategies on computer vision model quality under degraded data conditions: noisy labels, class imbalance, and visual artifacts.

## Tasks, Models, and Datasets

| Task | Models | Datasets |
|------|--------|----------|
| Image Classification | ResNet-18, Swin-T | CIFAR-10, STL-10 |
| Object Detection | YOLOv8n, Faster R-CNN | PASCAL VOC |
| Super Resolution | SRCNN, ESPCN, SwinIR | DIV2K ×4, STL-10 ×2 |

## CL Strategies

| Strategy | Description |
|----------|-------------|
| `baseline` | Full dataset each epoch, no curriculum |
| `stagewise` | Easy → Medium → Hard in three sequential stages |
| `static` | Groups fixed after a warm-up pass; cycled each epoch |
| `online` | Each epoch selects the easiest 30% by current loss |

## Data Degradation Types

| Type | Description | Parameter |
|------|-------------|-----------|
| `original` | Clean dataset | — |
| `noisy` | Random label replacement with probability p | `--noise_level` |
| `imbalance` | Undersample rare classes by factor f | `--imbalance_factor` |
| `artifacts` | Gaussian blur / pixel noise | `--artifact_level`, `--artifact_type` |

## Project Structure

```
diplom_v2/
├── curriculum/
│   └── strategies.py              # CL strategy implementations
├── datasets/
│   ├── curriculum_dataset.py      # Per-sample loss tracking wrapper
│   ├── difficult_datasets.py      # Degraded datasets (noise, imbalance, artifacts)
│   ├── detection_dataset.py       # YOLO detection dataset with degradation
│   ├── voc_fasterrcnn_dataset.py  # VOC dataset for Faster R-CNN
│   └── sr_dataset.py              # Super resolution dataset
├── models/
│   ├── classification.py          # ResNet-18, Swin-T
│   └── sr_models.py               # SRCNN, ESPCN, SwinIR
├── training/
│   ├── classification_trainer.py  # Classification trainer
│   ├── detection_trainer.py       # YOLOv8 trainer
│   ├── fasterrcnn_trainer.py      # Faster R-CNN trainer
│   └── sr_trainer.py              # Super resolution trainer
├── plots/
│   ├── visualize_data_pipeline.py # Data pipeline diagrams
│   ├── visualize_cl_schemes.py    # CL strategy diagrams
│   ├── visualize_results.py       # Classification plots
│   ├── visualize_det_results.py   # Detection plots (YOLOv8)
│   ├── visualize_sr_results.py    # SR plots (per dataset/scale)
│   └── visualize_sr_extra.py      # SR extra analysis plots
├── utils/
│   └── metrics.py
├── run_classification.py          # Classification experiment runner
├── run_detection.py               # YOLOv8 experiment runner
├── run_detection_fasterrcnn.py    # Faster R-CNN experiment runner
├── run_sr.py                      # Super resolution experiment runner
├── run_detection_all.sh           # Run all YOLO detection experiments
├── run_detection_fasterrcnn_all.sh # Run all Faster R-CNN experiments
└── run_sr_all.sh                  # Run all SR experiments
```

## Quick Start

### Classification

```bash
# CIFAR-10, noisy labels, ResNet-18 + Swin-T, all strategies
python run_classification.py \
    --dataset cifar10 --dataset_type noisy --noise_level 0.3 \
    --models resnet18,swin_t \
    --strategies baseline,stagewise,static,online \
    --total_steps 3000

# STL-10, class imbalance
python run_classification.py \
    --dataset stl10 --dataset_type imbalance --imbalance_factor 0.6 \
    --models resnet18 --total_steps 3000

# CIFAR-10, blur artifacts
python run_classification.py \
    --dataset cifar10 --dataset_type artifacts \
    --artifact_type blur --artifact_level 0.4 \
    --models resnet18,swin_t --total_steps 3000
```

### Object Detection — YOLOv8

```bash
bash run_detection_all.sh
# or individually:
python run_detection.py --dataset_type noisy --noise_level 0.3 \
    --strategies baseline,stagewise,static,online --total_steps 3000 --batch 8
```

### Object Detection — Faster R-CNN

```bash
bash run_detection_fasterrcnn_all.sh
# or individually:
python run_detection_fasterrcnn.py --dataset_type original \
    --strategies baseline,stagewise,static,online --total_steps 3000 --batch 4 --workers 0
```

### Super Resolution

```bash
bash run_sr_all.sh
# or individually:
python run_sr.py \
    --dataset_name div2k --scale 4 --degradation noise --deg_level 0.5 \
    --archs srcnn,espcn,swinir \
    --strategies baseline,stagewise,static,online \
    --total_steps 5000 --batch_size 16 --patience 15
```

### Visualization

```bash
python plots/visualize_sr_results.py        # SR main plots
python plots/visualize_sr_extra.py          # SR extra analysis
python plots/visualize_det_results.py       # Detection plots
python plots/visualize_results.py           # Classification plots
python plots/visualize_data_pipeline.py     # Data pipeline diagrams
python plots/visualize_cl_schemes.py        # CL strategy diagrams
```

## Key Results

### Super Resolution

- **STL-10 ×2:** Stagewise CL improves PSNR by up to +0.16 dB (ESPCN/Original). SRCNN consistently gains +0.10–0.13 dB on clean and noisy data.
- **DIV2K ×4:** CL strategies generally underperform the baseline. Only SRCNN/Stagewise on Blur shows a marginal gain (+0.12 dB).
- **Blur degradation** is consistently harder than noise; CL benefits collapse under strong blur.
- **SwinIR** benefits least from CL (or is hurt) across all conditions — its capacity is sufficient to learn without explicit ordering.
- **Online CL** is the weakest strategy; Stagewise is the most reliable.

### Object Detection (YOLOv8)

- CL strategies show moderate gains under noise/imbalance conditions.
- All strategies collapse under strong visual artifacts (blur/noise α=0.4): mAP ≈ 0.
- Stagewise CL provides the most consistent improvement.

## Results Format

Results are saved to `results/<experiment_name>/`:

- `classification_results.csv` — per-epoch train/val loss and accuracy
- `test_results.csv` — final test accuracy per strategy
- `detection_results.csv` — mAP@50, mAP@50-95 per strategy
- `sr_test_results.csv` — PSNR, SSIM, LPIPS per strategy
- `sr_results.csv` — per-epoch training curves (SR)

Plots are saved to `plots/output/`.

## Dependencies

```bash
pip install torch torchvision ultralytics \
            pandas matplotlib numpy pillow opencv-python \
            torchmetrics
```
