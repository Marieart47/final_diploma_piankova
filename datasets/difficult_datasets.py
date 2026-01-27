import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFilter
import random
from collections import Counter

class ClassImbalanceDataset(Dataset):
    """Датасет с дисбалансом классов"""
    def __init__(self, base_dataset, imbalance_factor=0.5, rare_classes=None):
        self.base = base_dataset
        self.losses = torch.zeros(len(base_dataset))
        self.imbalance_factor = imbalance_factor
        self.rare_classes = rare_classes if rare_classes is not None else [0, 1, 2]
        self.indices = self._create_imbalance()
        
        print(f"\n[Class Imbalance Dataset]")
        print(f"  Total original samples: {len(base_dataset)}")
        print(f"  Total after imbalance: {len(self.indices)}")
        print(f"  Imbalance factor: {imbalance_factor}")
        print(f"  Rare classes: {self.rare_classes}")
    
    def _create_imbalance(self):
        class_indices = {i: [] for i in range(10)}
        
        for idx in range(len(self.base)):
            data = self.base[idx]
            if isinstance(data, tuple):
                if len(data) >= 2:
                    _, label = data[:2]
                else:
                    label = -1
            else:
                label = -1
            
            class_indices[label].append(idx)
        
        kept_indices = []
        for class_idx in range(10):
            indices = class_indices[class_idx]
            if class_idx in self.rare_classes:
                keep_count = max(1, int(len(indices) * (1 - self.imbalance_factor)))
                kept_indices.extend(random.sample(indices, keep_count))
            else:
                kept_indices.extend(indices)
        
        # Сортируем для воспроизводимости
        kept_indices.sort()
        return kept_indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data = self.base[real_idx]
        
        if isinstance(data, tuple):
            if len(data) == 2:
                image, label = data
                return image, label, idx
            elif len(data) == 3:
                image, label, _ = data
                return image, label, idx
        
        return data, 0, idx
    
    def update_losses(self, idxs, losses):
        """
        idxs - индексы из датасета (0..len(self.indices)-1)
        """
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        
        for ds_idx, loss in zip(idxs, losses):
            if ds_idx < len(self.indices):
                real_idx = self.indices[ds_idx]
                self.losses[real_idx] = loss
    
    def ranked_indices(self):
        """
        Возвращает индексы ДАТАСЕТА (0..len(self.indices)-1), отсортированные по потерям
        """
        # Создаем список пар (индекс датасета, потеря реального индекса)
        ds_loss_pairs = []
        for ds_idx, real_idx in enumerate(self.indices):
            ds_loss_pairs.append((ds_idx, self.losses[real_idx].item()))
        
        # Сортируем по потерям (от меньших к большим)
        ds_loss_pairs.sort(key=lambda x: x[1])
        
        # Возвращаем только индексы датасета
        return torch.tensor([ds_idx for ds_idx, _ in ds_loss_pairs])


class NoisyLabelDataset(Dataset):
    """Датасет с шумными метками"""
    def __init__(self, base_dataset, noise_level=0.3, noise_type='random'):
        self.base = base_dataset
        self.losses = torch.zeros(len(base_dataset))
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.noisy_labels = self._create_noisy_labels()
        
        print(f"\n[Noisy Label Dataset]")
        print(f"  Noise level: {noise_level}")
        print(f"  Noise type: {noise_type}")
    
    def _create_noisy_labels(self):
        noisy_labels = {}
        for idx in range(len(self.base)):
            data = self.base[idx]
            if isinstance(data, tuple) and len(data) >= 2:
                _, original_label = data[:2]
            else:
                original_label = 0
            
            if random.random() < self.noise_level:
                if self.noise_type == 'random':
                    noisy_labels[idx] = random.choice([i for i in range(10) if i != original_label])
                elif self.noise_type == 'flip':
                    noisy_labels[idx] = 9 - original_label
                elif self.noise_type == 'pair_flip':
                    pair_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 
                               6: 7, 7: 6, 8: 9, 9: 8}
                    noisy_labels[idx] = pair_map.get(original_label, original_label)
            else:
                noisy_labels[idx] = original_label
        return noisy_labels
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        data = self.base[idx]
        if isinstance(data, tuple):
            if len(data) == 2:
                image, original_label = data
            elif len(data) == 3:
                image, original_label, _ = data
            else:
                image, original_label = data[0], 0
        else:
            image, original_label = data, 0
        
        # Используем оригинальный idx для доступа к noisy_labels
        noisy_label = self.noisy_labels.get(idx, original_label)
        return image, noisy_label, idx
    
    def update_losses(self, idxs, losses):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        
        for idx, loss in zip(idxs, losses):
            self.losses[idx] = loss
    
    def ranked_indices(self):
        """
        Возвращает индексы (0..len(self)-1), отсортированные по потерям
        """
        return torch.argsort(self.losses)


class VisualArtifactsDataset(Dataset):
    """Датасет с визуальными артефактами"""
    def __init__(self, base_dataset, artifact_level=0.5, artifact_type='mixed'):
        self.base = base_dataset
        self.losses = torch.zeros(len(base_dataset))
        self.artifact_level = artifact_level
        self.artifact_type = artifact_type
        self.artifact_cache = {}
        
        print(f"\n[Visual Artifacts Dataset]")
        print(f"  Artifact level: {artifact_level}")
        print(f"  Artifact type: {artifact_type}")
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        if idx in self.artifact_cache:
            return self.artifact_cache[idx]
        
        data = self.base[idx]
        if isinstance(data, tuple):
            if len(data) == 2:
                image, label = data
            elif len(data) == 3:
                image, label, _ = data
            else:
                image, label = data[0], 0
        else:
            image, label = data, 0
        
        if self.artifact_level > 0:
            image = self._apply_artifact(image)
        
        self.artifact_cache[idx] = (image, label, idx)
        return image, label, idx
    
    def _apply_artifact(self, image):
        if isinstance(image, torch.Tensor):
            pil_image = transforms.ToPILImage()(image)
        else:
            pil_image = image
        
        if self.artifact_type == 'mixed':
            artifact_type = random.choice(['blur', 'noise', 'low_res'])
        else:
            artifact_type = self.artifact_type
        
        if artifact_type == 'blur':
            blur_radius = 1 + self.artifact_level * 4
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        elif artifact_type == 'noise':
            img_array = np.array(pil_image).astype(np.float32)
            noise_std = self.artifact_level * 50
            noise = np.random.normal(0, noise_std, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            pil_image = Image.fromarray(img_array.astype(np.uint8))
        
        elif artifact_type == 'low_res':
            scale = max(0.1, 1 - self.artifact_level * 0.9)
            small_w = int(pil_image.width * scale)
            small_h = int(pil_image.height * scale)
            small_image = pil_image.resize((small_w, small_h), Image.BILINEAR)
            pil_image = small_image.resize((pil_image.width, pil_image.height), Image.NEAREST)
        
        return transforms.ToTensor()(pil_image)
    
    def update_losses(self, idxs, losses):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        
        for idx, loss in zip(idxs, losses):
            self.losses[idx] = loss
    
    def ranked_indices(self):
        return torch.argsort(self.losses)


# Фабрика для создания датасетов
def create_dataset(base_dataset, dataset_type='original', **kwargs):
    """
    Создает датасет с заданными сложностями
    """
    if dataset_type == 'original':
        from datasets.curriculum_dataset import CurriculumDataset
        return CurriculumDataset(base_dataset)
    
    elif dataset_type == 'imbalance':
        imbalance_factor = kwargs.get('imbalance_factor', 0.5)
        rare_classes = kwargs.get('rare_classes', [0, 1, 2])
        return ClassImbalanceDataset(base_dataset, imbalance_factor, rare_classes)
    
    elif dataset_type == 'noisy':
        noise_level = kwargs.get('noise_level', 0.3)
        noise_type = kwargs.get('noise_type', 'random')
        return NoisyLabelDataset(base_dataset, noise_level, noise_type)
    
    elif dataset_type == 'artifacts':
        artifact_level = kwargs.get('artifact_level', 0.5)
        artifact_type = kwargs.get('artifact_type', 'mixed')
        return VisualArtifactsDataset(base_dataset, artifact_level, artifact_type)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
