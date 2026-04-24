import torch
from torch.utils.data import Dataset


class CurriculumDataset(Dataset):
    """Обёртка над датасетом с отслеживанием per-sample лосса для курикулума."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.losses = torch.zeros(len(base_dataset))
        self.seen_counts = torch.zeros(len(base_dataset), dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        return data, idx

    def update_losses(self, idxs: torch.Tensor, per_sample_losses: torch.Tensor):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.cpu()
        else:
            idxs = torch.tensor(idxs)

        per_sample_losses = per_sample_losses.detach().cpu()

        if idxs.shape != per_sample_losses.shape:
            raise ValueError(
                f"Shape mismatch: idxs {idxs.shape} vs losses {per_sample_losses.shape}. "
                "Make sure you pass per-sample losses, not a batch-averaged scalar."
            )

        self.losses[idxs] = per_sample_losses
        self.seen_counts[idxs] += 1

    def ranked_indices(self) -> torch.Tensor:
        return torch.argsort(self.losses)

    def difficulty_stats(self) -> dict:
        seen_mask = self.seen_counts > 0
        return {
            "num_seen": seen_mask.sum().item(),
            "num_unseen": (~seen_mask).sum().item(),
            "mean_loss": self.losses[seen_mask].mean().item() if seen_mask.any() else 0.0,
            "max_loss": self.losses[seen_mask].max().item() if seen_mask.any() else 0.0,
            "min_loss": self.losses[seen_mask].min().item() if seen_mask.any() else 0.0,
        }
