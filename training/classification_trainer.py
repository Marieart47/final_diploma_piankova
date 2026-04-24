import torch
import torch.nn as nn
from typing import Tuple


def _unpack_batch(batch):
    def is_tensor_4d(obj):
        return isinstance(obj, torch.Tensor) and obj.dim() == 4

    if is_tensor_4d(batch[0]):
        return batch[0].float(), batch[1].long(), None

    elif (isinstance(batch[0], (list, tuple))
          and isinstance(batch[0][0], (list, tuple))
          and is_tensor_4d(batch[0][0][0])):
        return batch[0][0][0].float(), batch[0][0][1].long(), batch[0][1].long()

    elif isinstance(batch[0], (list, tuple)) and is_tensor_4d(batch[0][0]):
        idx = batch[1].long() if (len(batch) > 1 and isinstance(batch[1], torch.Tensor)) else None
        return batch[0][0].float(), batch[0][1].long(), idx

    else:
        raise ValueError(f"Unrecognised batch structure: {[type(b) for b in batch]}")


class ClassificationTrainer:
    def __init__(self, model: nn.Module, optimizer, loss_fn: nn.Module, device: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        try:
            self._loss_per_sample = type(loss_fn)(reduction="none")
        except Exception:
            self._loss_per_sample = loss_fn

    def train_epoch(self, loader, dataset) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            x, y, idx = _unpack_batch(batch)
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)

            per_sample_losses = self._loss_per_sample(outputs, y)
            loss = per_sample_losses.mean()
            loss.backward()
            self.optimizer.step()

            if hasattr(dataset, "update_losses"):
                try:
                    dataset.update_losses(idx.cpu(), per_sample_losses.detach().cpu())
                except Exception as e:
                    print(f"Warning: Could not update losses: {e}")

            total_loss += loss.item() * x.size(0)
            correct += outputs.max(1)[1].eq(y).sum().item()
            total += x.size(0)

        return (total_loss / total if total > 0 else 0.0,
                correct / total if total > 0 else 0.0)

    def validate(self, loader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in loader:
                x, y, idx = _unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)

                total_loss += loss.item() * x.size(0)
                correct += outputs.max(1)[1].eq(y).sum().item()
                total += x.size(0)

        return (total_loss / total if total > 0 else 0.0,
                correct / total if total > 0 else 0.0)