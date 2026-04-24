import random
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.curriculum_dataset import CurriculumDataset


class ClassImbalanceDataset(Dataset):
    """Датасет с искусственным дисбалансом классов: редкие классы прореживаются."""

    def __init__(self, base_dataset, imbalance_factor=0.5, rare_classes=None):
        self.base = base_dataset
        self.losses = torch.zeros(len(base_dataset))
        self.imbalance_factor = imbalance_factor
        self.rare_classes = rare_classes if rare_classes is not None else [0, 1, 2]

        print(f"\n[Class Imbalance Dataset]")
        print(f"  Original dataset size: {len(base_dataset)}")
        print(f"  Imbalance factor: {imbalance_factor}")
        print(f"  Rare classes: {self.rare_classes}")

        self._collect_class_stats()
        self.indices = self._create_imbalance()

        print(f"  Final dataset size: {len(self.indices)}")
        print(f"  Samples removed: {len(base_dataset) - len(self.indices)}")

    def _collect_class_stats(self):
        class_counts = Counter()
        sample_size = min(1000, len(self.base))

        for i in range(sample_size):
            try:
                item = self.base[i]
                label = item[1] if len(item) >= 2 else None
                if label is not None:
                    class_counts[label] += 1
            except Exception:
                continue

        print(f"  Original class distribution (first {sample_size} samples):")
        for class_id in range(10):
            count = class_counts.get(class_id, 0)
            percentage = (count / sample_size) * 100
            status = "RARE" if class_id in self.rare_classes else "COMMON"
            print(f"    Class {class_id}: {count} samples ({percentage:.1f}%) [{status}]")

    def _create_imbalance(self):
        all_samples = []
        for idx in range(len(self.base)):
            try:
                item = self.base[idx]
                label = item[1] if len(item) >= 2 else -1
            except Exception:
                label = -1
            all_samples.append((idx, label))

        # Group indices by class
        class_indices = {i: [] for i in range(10)}
        for idx, label in all_samples:
            if 0 <= label <= 9:
                class_indices[label].append(idx)

        kept_indices = []
        for class_idx in range(10):
            indices = class_indices[class_idx]
            if class_idx in self.rare_classes:
                # Keep only (1 - imbalance_factor) of rare class samples
                keep_count = max(1, int(len(indices) * (1 - self.imbalance_factor)))
                selected = random.sample(indices, keep_count) if indices else []
                kept_indices.extend(selected)
                print(f"  Class {class_idx} (rare): {len(selected)}/{len(indices)} "
                      f"samples kept ({len(selected)/len(indices)*100:.0f}%)")
            else:
                kept_indices.extend(indices)
                print(f"  Class {class_idx} (common): {len(indices)}/{len(indices)} "
                      f"samples kept (100%)")

        random.shuffle(kept_indices)
        return kept_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map dataset-local idx → original base index
        real_idx = self.indices[idx]
        try:
            data = self.base[real_idx]
            image = data[0]
            label = data[1]
            return image, label, idx
        except Exception:
            return torch.zeros((3, 32, 32)), 0, idx

    def update_losses(self, idxs, losses):
        """Store per-sample losses using base-dataset indices."""
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        for ds_idx, loss in zip(idxs, losses):
            if ds_idx < len(self.indices):
                real_idx = self.indices[ds_idx]
                self.losses[real_idx] = loss

    def ranked_indices(self):
        """Returns dataset-local indices sorted from easiest (lowest loss) to hardest."""
        ds_loss_pairs = [
            (ds_idx, self.losses[real_idx].item())
            for ds_idx, real_idx in enumerate(self.indices)
        ]
        ds_loss_pairs.sort(key=lambda x: x[1])
        return torch.tensor([ds_idx for ds_idx, _ in ds_loss_pairs])


class NoisyLabelDataset(Dataset):
    """Датасет с зашумлёнными метками: каждая метка случайно заменяется с вероятностью noise_level."""

    def __init__(self, base_dataset, noise_level=0.3):
        self.base = base_dataset
        self.losses = torch.zeros(len(base_dataset))
        self.noise_level = noise_level

        print(f"\n[Noisy Label Dataset]")
        print(f"  Noise level: {noise_level}")

        self.noisy_labels = self._create_noisy_labels()
        self._print_noise_stats()

    def _create_noisy_labels(self):
        noisy_labels = {}
        for idx in range(len(self.base)):
            try:
                item = self.base[idx]
                original_label = item[1] if len(item) >= 2 else 0
            except Exception:
                original_label = 0

            if random.random() < self.noise_level:
                # Replace with a uniformly random label from the other 9 classes
                possible = [i for i in range(10) if i != original_label]
                noisy_labels[idx] = random.choice(possible) if possible else original_label
            else:
                noisy_labels[idx] = original_label

        return noisy_labels

    def _print_noise_stats(self):
        correct = 0
        sample_size = min(1000, len(self.base))

        for idx in range(sample_size):
            try:
                item = self.base[idx]
                original_label = item[1] if len(item) >= 2 else None
                if original_label is not None and self.noisy_labels.get(idx, original_label) == original_label:
                    correct += 1
            except Exception:
                continue

        accuracy = (correct / sample_size) * 100
        print(f"  Label accuracy in sample: {accuracy:.1f}%")
        print(f"  Estimated noisy samples: {sample_size - correct}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        try:
            item = self.base[idx]
            image = item[0]
            original_label = item[1] if len(item) >= 2 else 0
        except Exception:
            image, original_label = torch.zeros((3, 32, 32)), 0

        noisy_label = self.noisy_labels.get(idx, original_label)
        return image, noisy_label, idx

    def update_losses(self, idxs, losses):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        for idx, loss in zip(idxs, losses):
            self.losses[idx] = loss

    def ranked_indices(self):
        """Returns indices sorted from easiest (lowest loss) to hardest."""
        return torch.argsort(self.losses)


class VisualArtifactsDataset(Dataset):
    """Датасет с визуальными артефактами: blur / noise / low_res / mixed.

    Артефакт применяется один раз и кэшируется для воспроизводимости.
    """

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
        # Return cached result to ensure reproducible artifacts across epochs
        if idx in self.artifact_cache:
            return self.artifact_cache[idx]

        try:
            item = self.base[idx]
            image = item[0]
            label = item[1] if len(item) >= 2 else 0
        except Exception:
            image, label = torch.zeros((3, 32, 32)), 0

        if self.artifact_level > 0:
            image = self._apply_artifact(image)

        result = (image, label, idx)
        self.artifact_cache[idx] = result
        return result

    def _apply_artifact(self, image):
        """Applies a single artifact to the image and returns a tensor."""
        if isinstance(image, torch.Tensor):
            pil_image = transforms.ToPILImage()(image)
        else:
            pil_image = image

        artifact_type = (
            random.choice(['blur', 'noise', 'low_res'])
            if self.artifact_type == 'mixed'
            else self.artifact_type
        )

        if artifact_type == 'blur':
            # Gaussian blur radius: 1 (artifact_level=0) → 5 (artifact_level=1)
            blur_radius = 1 + self.artifact_level * 4
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        elif artifact_type == 'noise':
            # Additive Gaussian noise, std: 0 → 50
            img_array = np.array(pil_image).astype(np.float32)
            noise_std = self.artifact_level * 50
            noise = np.random.normal(0, noise_std, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            pil_image = Image.fromarray(img_array.astype(np.uint8))

        elif artifact_type == 'low_res':
            # Downscale to (scale × original size) then upscale back with nearest-neighbor
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
        """Returns indices sorted from easiest (lowest loss) to hardest."""
        return torch.argsort(self.losses)


def create_dataset(base_dataset, dataset_type='original', **kwargs):
    print(f"  Creating {dataset_type} dataset with parameters: {kwargs}")

    if dataset_type == 'original':
        return CurriculumDataset(base_dataset)
    elif dataset_type == 'imbalance':
        return ClassImbalanceDataset(
            base_dataset,
            imbalance_factor=kwargs.get('imbalance_factor', 0.5),
            rare_classes=kwargs.get('rare_classes', [0, 1, 2]),
        )
    elif dataset_type == 'noisy':
        return NoisyLabelDataset(
            base_dataset,
            noise_level=kwargs.get('noise_level', 0.3),
        )
    elif dataset_type == 'artifacts':
        return VisualArtifactsDataset(
            base_dataset,
            artifact_level=kwargs.get('artifact_level', 0.5),
            artifact_type=kwargs.get('artifact_type', 'mixed'),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
