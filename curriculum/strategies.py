from torch.utils.data import Subset
import torch

class Baseline:
    def get_dataset(self, dataset):
        return dataset

class StageWise:
    def __init__(self, stages=(0.0, 0.3, 0.6, 1.0)):
        self.stages = stages
        self.stage = 0

    def step(self):
        self.stage = min(self.stage + 1, len(self.stages) - 2)

    def get_dataset(self, dataset):
        ranked = dataset.ranked_indices()
        n = len(ranked)
        s, e = self.stages[self.stage], self.stages[self.stage+1]
        # Проверяем, что индексы валидны
        start_idx = int(s * n)
        end_idx = int(e * n)
        
        if start_idx >= end_idx:
            end_idx = min(start_idx + 1, n)
        
        indices = ranked[start_idx:end_idx]
        # Убедимся, что индексы это тензор или список
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        
        return Subset(dataset, indices)

class Static:
    def __init__(self):
        self.groups = None

    def initialize(self, dataset):
        ranked = dataset.ranked_indices()
        n = len(ranked)
        
        # Преобразуем в список если это тензор
        if isinstance(ranked, torch.Tensor):
            ranked = ranked.tolist()
        
        self.groups = [
            ranked[:n//3],
            ranked[n//3:2*n//3],
            ranked[2*n//3:]
        ]

    def get_dataset(self, dataset, stage):
        stage = min(stage, len(self.groups) - 1)
        return Subset(dataset, self.groups[stage])

class Online:
    def __init__(self, pct=0.3):
        self.pct = pct

    def get_dataset(self, dataset):
        ranked = dataset.ranked_indices()
        
        # Преобразуем в список если это тензор
        if isinstance(ranked, torch.Tensor):
            ranked = ranked.tolist()
        
        subset_size = int(len(ranked) * self.pct)
        if subset_size == 0:
            subset_size = 1
        
        return Subset(dataset, ranked[:subset_size])
    
