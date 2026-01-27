import torch
from torch.utils.data import Dataset

class CurriculumDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.losses = torch.zeros(len(base_dataset))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        return data, idx

    def update_losses(self, idxs, losses):
        self.losses[idxs] = losses.detach().cpu()

    def ranked_indices(self):
        return torch.argsort(self.losses)
