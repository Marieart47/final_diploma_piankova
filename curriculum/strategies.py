from torch.utils.data import Subset
import torch
from typing import Optional


class Baseline:
    """Использует весь датасет на каждой эпохе."""

    def get_dataset(self, dataset):
        return dataset


class StageWise:
    """
    Поэтапный курикулум: обучение разбито на стадии по окнам сложности.
    Переход к следующей стадии выполняется вызовом step().
    """

    def __init__(self, stages=(0.0, 0.3, 0.6, 1.0)):
        self.stages = stages
        self.stage = 0

    def step(self):
        self.stage = min(self.stage + 1, len(self.stages) - 2)

    @property
    def current_fraction(self) -> float:
        s = self.stages[self.stage]
        e = self.stages[self.stage + 1]
        return e - s

    def get_dataset(self, dataset) -> Subset:
        ranked = dataset.ranked_indices()
        if isinstance(ranked, torch.Tensor):
            ranked = ranked.tolist()

        n = len(ranked)
        start_idx = int(self.stages[self.stage] * n)
        end_idx = int(self.stages[self.stage + 1] * n)
        end_idx = max(end_idx, start_idx + 1)
        end_idx = min(end_idx, n)

        return Subset(dataset, ranked[start_idx:end_idx])


class Static:
    """
    Статический курикулум: группы сложности фиксируются один раз после
    warm-up прохода и не обновляются в процессе обучения.
    """

    def __init__(self, num_groups: int = 3):
        self.num_groups = num_groups
        self.groups: Optional[list] = None

    def initialize(self, dataset):
        ranked = dataset.ranked_indices()
        if isinstance(ranked, torch.Tensor):
            ranked = ranked.tolist()

        n = len(ranked)
        self.groups = []
        for i in range(self.num_groups):
            start = (i * n) // self.num_groups
            end = ((i + 1) * n) // self.num_groups
            self.groups.append(ranked[start:end])

    def get_dataset(self, dataset, stage: int) -> Subset:
        if self.groups is None:
            raise RuntimeError("Call initialize(dataset) before get_dataset().")
        stage = min(stage, len(self.groups) - 1)
        return Subset(dataset, self.groups[stage])


class Online:
    """
    Динамический курикулум: на каждой эпохе выбирается pct-доля наилегчайших
    образцов по текущему лоссу.
    """

    def __init__(self, pct: float = 0.3):
        assert 0.0 < pct <= 1.0
        self.pct = pct

    def get_dataset(self, dataset) -> Subset:
        ranked = dataset.ranked_indices()
        if isinstance(ranked, torch.Tensor):
            ranked = ranked.tolist()

        subset_size = max(1, int(len(ranked) * self.pct))
        return Subset(dataset, ranked[:subset_size])


class AntiCurriculum:
    """Обратный курикулум: обучение начинается с наиболее сложных образцов."""

    def __init__(self, pct: float = 0.3):
        assert 0.0 < pct <= 1.0
        self.pct = pct

    def get_dataset(self, dataset) -> Subset:
        ranked = dataset.ranked_indices()
        if isinstance(ranked, torch.Tensor):
            ranked = ranked.tolist()

        subset_size = max(1, int(len(ranked) * self.pct))
        return Subset(dataset, ranked[-subset_size:])
