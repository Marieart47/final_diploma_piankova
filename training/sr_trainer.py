"""Тренер для задачи Super Resolution с метрикой PSNR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Среднее PSNR по батчу (dB), значения в диапазоне [0, 1]."""
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])  # (B,)
    return (10.0 * torch.log10(1.0 / (mse + 1e-8))).mean().item()


class SRTrainer:
    """
    Тренер для SR-моделей.

    Лосс: L1 (MAE) по пикселям — стандарт для задачи SR.
    Дополнительно вычисляет PSNR на каждой эпохе.
    Интегрируется с CurriculumDataset через update_losses() по per-sample L1.
    """

    def __init__(self, model: nn.Module, optimizer, device: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader, dataset) -> Tuple[float, float]:
        """Возвращает (средний L1-лосс, средний PSNR в dB) за эпоху."""
        self.model.train()
        total_loss, total_psnr, count = 0.0, 0.0, 0

        for batch in loader:
            (lr, hr), idx = batch
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            self.optimizer.zero_grad()
            sr = self.model(lr)

            # Per-sample L1 (среднее по C, H, W) — для курикулума
            per_sample = F.l1_loss(sr, hr, reduction="none").mean(dim=[1, 2, 3])
            loss = per_sample.mean()
            loss.backward()
            self.optimizer.step()

            if hasattr(dataset, "update_losses"):
                try:
                    dataset.update_losses(idx.cpu(), per_sample.detach().cpu())
                except Exception as e:
                    print(f"Warning: update_losses failed: {e}")

            bs = lr.size(0)
            total_loss += loss.item() * bs
            total_psnr += psnr(sr.detach(), hr) * bs
            count += bs

        n = max(count, 1)
        return total_loss / n, total_psnr / n

    @torch.no_grad()
    def validate(self, loader) -> Tuple[float, float]:
        """Возвращает (средний L1-лосс, средний PSNR в dB) на валидации."""
        self.model.eval()
        total_loss, total_psnr, count = 0.0, 0.0, 0

        for batch in loader:
            # Поддержка как обычного DataLoader, так и с CurriculumDataset
            if isinstance(batch[0], (list, tuple)):
                (lr, hr), _ = batch
            else:
                lr, hr = batch

            lr = lr.to(self.device)
            hr = hr.to(self.device)

            sr = self.model(lr)

            bs = lr.size(0)
            total_loss += F.l1_loss(sr, hr).item() * bs
            total_psnr += psnr(sr, hr) * bs
            count += bs

        n = max(count, 1)
        return total_loss / n, total_psnr / n
