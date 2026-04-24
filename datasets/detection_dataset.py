"""Датасет детекции с поддержкой курикулума и инжекцией сложности (YOLODataset)."""

import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from ultralytics.data import YOLODataset


# ── Rare VOC classes used for imbalance (indices 0-19 = aeroplane..tvmonitor) ──
# First 5 classes designated as "rare" (aeroplane, bicycle, bird, boat, bottle)
VOC_RARE_CLASSES = [0, 1, 2, 3, 4]
VOC_NUM_CLASSES  = 20


class CurriculumDetectionDataset(YOLODataset):
    """YOLODataset с отслеживанием per-sample лосса и инжекцией сложности."""

    def __init__(self, *args,
                 difficulty_type: str = 'original',
                 noise_level: float = 0.3,
                 imbalance_factor: float = 0.3,
                 artifact_level: float = 0.4,
                 artifact_type: str = 'blur',
                 rare_classes: list = None,
                 num_classes: int = VOC_NUM_CLASSES,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.difficulty_type   = difficulty_type
        self.noise_level       = noise_level
        self.imbalance_factor  = imbalance_factor
        self.artifact_level    = artifact_level
        self.artifact_type     = artifact_type
        self.rare_classes      = set(rare_classes or VOC_RARE_CLASSES)
        self.num_classes       = num_classes

        n = len(self.im_files)
        self.sample_losses = torch.rand(n) * 1e-6
        self.seen_counts   = torch.zeros(n, dtype=torch.long)

        # Imbalance: filter out rare-class images at construction time
        if difficulty_type == 'imbalance':
            self._active_indices = self._apply_imbalance()
        else:
            self._active_indices = list(range(n))

        print(f'\n[DetectionDataset] type={difficulty_type} | '
              f'{len(self._active_indices)}/{n} images active')

    # ── Imbalance: filter images at construction ────────────────────────────────

    def _apply_imbalance(self) -> list:
        keep = []
        for idx, label in enumerate(self.labels):
            cls_arr = label.get('cls', None)
            if cls_arr is None or len(cls_arr) == 0:
                keep.append(idx)
                continue

            # Majority class of this image
            classes = cls_arr.flatten().tolist()
            majority = max(set(classes), key=classes.count)

            if int(majority) in self.rare_classes:
                # Keep with probability (1 - imbalance_factor)
                if random.random() > self.imbalance_factor:
                    keep.append(idx)
            else:
                keep.append(idx)

        print(f'  Imbalance: kept {len(keep)}/{len(self.labels)} images '
              f'(removed {len(self.labels) - len(keep)} rare-class images)')
        return keep

    # ── __len__ / __getitem__ ───────────────────────────────────────────────────

    def __len__(self):
        return len(self._active_indices)

    def __getitem__(self, index):
        # Map dataset-local index → full-list index via active_indices
        real_idx = self._active_indices[index]
        sample = super().__getitem__(real_idx)
        sample['orig_idx'] = torch.tensor(real_idx, dtype=torch.long)

        if self.difficulty_type == 'noisy':
            sample = self._corrupt_labels(sample)
        elif self.difficulty_type == 'artifacts':
            sample['img'] = self._apply_artifact(sample['img'])

        return sample

    # ── Noisy: corrupt class labels ────────────────────────────────────────────

    def _corrupt_labels(self, sample: dict) -> dict:
        """Randomly replace each object's class label with a wrong class."""
        cls = sample.get('cls')
        if cls is None or len(cls) == 0:
            return sample

        new_cls = cls.clone()
        for i in range(len(new_cls)):
            if random.random() < self.noise_level:
                original = int(new_cls[i].item())
                choices = [c for c in range(self.num_classes) if c != original]
                new_cls[i] = random.choice(choices)
        sample['cls'] = new_cls
        return sample

    # ── Artifacts: degrade image ────────────────────────────────────────────────

    def _apply_artifact(self, img_tensor: torch.Tensor) -> torch.Tensor:
        to_pil   = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        pil = to_pil(img_tensor.clamp(0, 1))

        art = (random.choice(['blur', 'noise', 'low_res'])
               if self.artifact_type == 'mixed' else self.artifact_type)

        if art == 'blur':
            # Radius: 1 (level=0) → 5 (level=1)
            radius = 1 + self.artifact_level * 4
            pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))

        elif art == 'noise':
            # Additive Gaussian noise, std: 0 → 50 pixel values
            arr = np.array(pil).astype(np.float32)
            noise = np.random.normal(0, self.artifact_level * 50, arr.shape)
            arr = np.clip(arr + noise, 0, 255)
            pil = Image.fromarray(arr.astype(np.uint8))

        elif art == 'low_res':
            # Downscale then upscale (pixelation), scale: 1.0 → 0.1
            scale = max(0.1, 1 - self.artifact_level * 0.9)
            small = pil.resize(
                (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                Image.BILINEAR
            )
            pil = small.resize((pil.width, pil.height), Image.NEAREST)

        return to_tensor(pil)

    # ── Collate ────────────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch):
        """Extends YOLODataset.collate_fn to stack orig_idx into a 1-D tensor."""
        new_batch = YOLODataset.collate_fn(batch)
        if 'orig_idx' in new_batch:
            val = new_batch['orig_idx']
            if isinstance(val, (list, tuple)):
                new_batch['orig_idx'] = torch.stack(list(val))
        return new_batch

    # ── Curriculum interface ───────────────────────────────────────────────────

    def update_losses(self, idxs, losses):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.cpu().tolist()
        if isinstance(losses, (int, float)):
            losses = [losses] * len(idxs)
        elif isinstance(losses, torch.Tensor):
            losses = losses.cpu().tolist()

        for i, loss in zip(idxs, losses):
            if 0 <= i < len(self.sample_losses):
                self.sample_losses[i] = float(loss)
                self.seen_counts[i] += 1

    def ranked_indices(self):
        active = torch.tensor(self._active_indices, dtype=torch.long)
        losses = self.sample_losses[active]
        order  = torch.argsort(losses)
        # Return positions in [0, len(active_indices)) that strategies expect
        return order
