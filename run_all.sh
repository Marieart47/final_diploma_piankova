#!/bin/zsh

echo "Starting all experiments..."

# # ── CIFAR-10 ────────────────────────────────────────────────────────────────────
# python3 -u run_classification.py --dataset cifar10 --dataset_type original --models resnet18 --resize 96 --budget_mode steps --total_steps 3000
# python3 -u run_classification.py --dataset cifar10 --dataset_type noisy --noise_level 0.3 --models resnet18 --resize 96 --budget_mode steps --total_steps 3000
# python3 -u run_classification.py --dataset cifar10 --dataset_type noisy --noise_level 0.6 --models resnet18 --resize 96 --budget_mode steps --total_steps 3000
# python3 -u run_classification.py --dataset cifar10 --dataset_type imbalance --imbalance_factor 0.3 --models resnet18 --resize 96 --budget_mode steps --total_steps 3000
# python3 -u run_classification.py --dataset cifar10 --dataset_type imbalance --imbalance_factor 0.6 --models resnet18 --resize 96 --budget_mode steps --total_steps 3000
# python3 -u run_classification.py --dataset cifar10 --dataset_type artifacts --artifact_type blur --models resnet18 --resize 96 --artifact_level 0.4 --budget_mode steps --total_steps 3000
# python3 -u run_classification.py --dataset cifar10 --dataset_type artifacts --artifact_type noise --models resnet18 --resize 96 --artifact_level 0.4 --budget_mode steps --total_steps 3000

# ── CIFAR-10 swin-T ────────────────────────────────────────────────────────────────────
python3 -u run_classification.py --dataset cifar10 --dataset_type original --models swin_t --resize 96 --budget_mode steps --total_steps 3000
python3 -u run_classification.py --dataset cifar10 --dataset_type noisy --noise_level 0.3 --models swin_t --resize 96 --budget_mode steps --total_steps 3000
python3 -u run_classification.py --dataset cifar10 --dataset_type noisy --noise_level 0.6 --models swin_t --resize 96 --budget_mode steps --total_steps 3000
python3 -u run_classification.py --dataset cifar10 --dataset_type imbalance --imbalance_factor 0.3 --models swin_t --resize 96 --budget_mode steps --total_steps 3000
python3 -u run_classification.py --dataset cifar10 --dataset_type imbalance --imbalance_factor 0.6 --models swin_t --resize 96 --budget_mode steps --total_steps 3000
python3 -u run_classification.py --dataset cifar10 --dataset_type artifacts --artifact_type blur --models swin_t --resize 96 --artifact_level 0.4 --budget_mode steps --total_steps 3000
python3 -u run_classification.py --dataset cifar10 --dataset_type artifacts --artifact_type noise --models swin_t --resize 96 --artifact_level 0.4 --budget_mode steps --total_steps 3000


# ── ResNet-18 on STL-10 (224×224) ───────────────────────────────────────────────

# python3 -u run_classification.py --dataset stl10 --dataset_type original \
#     --models resnet18 --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type noisy --noise_level 0.3 \
#     --models resnet18 --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type noisy --noise_level 0.6 \
#     --models resnet18 --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type imbalance --imbalance_factor 0.3 \
#     --models resnet18 --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type imbalance --imbalance_factor 0.6 \
#     --models resnet18 --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type artifacts --artifact_type blur \
#     --artifact_level 0.4 --models resnet18 --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type artifacts --artifact_type noise \
#     --artifact_level 0.4 --models resnet18 --resize 224 --budget_mode steps --total_steps 3000

# ── Swin-T on STL-10 (224×224) ──────────────────────────────────────────────────
# Swin Transformer Tiny — ICCV 2021 Best Paper, ~28M params.
# Requires 224×224 input. Run separately from ResNet-18 (different resolution).

# python3 -u run_classification.py --dataset stl10 --dataset_type original \
#     --models swin_t --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type noisy --noise_level 0.3 \
#     --models swin_t --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type noisy --noise_level 0.6 \
#     --models swin_t --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type imbalance --imbalance_factor 0.3 \
#     --models swin_t --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type imbalance --imbalance_factor 0.6 \
#     --models swin_t --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type artifacts --artifact_type blur \
#     --artifact_level 0.4 --models swin_t --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type artifacts --artifact_type noise \
#     --artifact_level 0.4 --models swin_t --resize 224 --budget_mode steps --total_steps 3000

# ── MLP-Mixer-S/16 on STL-10 (224×224) ──────────────────────────────────────────
# MLP-Mixer Small — NeurIPS 2021, ~18.5M params. Pure MLP, no convolutions or attention.
# Requires 224×224 input.

# python3 -u run_classification.py --dataset stl10 --dataset_type original \
#     --models mlp_mixer --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type noisy --noise_level 0.3 \
#     --models mlp_mixer --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type noisy --noise_level 0.6 \
#     --models mlp_mixer --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type imbalance --imbalance_factor 0.3 \
#     --models mlp_mixer --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type imbalance --imbalance_factor 0.6 \
#     --models mlp_mixer --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type artifacts --artifact_type blur \
#     --artifact_level 0.4 --models mlp_mixer --resize 224 --budget_mode steps --total_steps 3000

# python3 -u run_classification.py --dataset stl10 --dataset_type artifacts --artifact_type noise \
#     --artifact_level 0.4 --models mlp_mixer --resize 224 --budget_mode steps --total_steps 3000

echo "All done! Now visualizing..."
# python3 plots/visualize_all_results.py