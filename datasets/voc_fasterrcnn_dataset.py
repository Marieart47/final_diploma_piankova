"""VOC dataset для Faster R-CNN: читает YOLO-формат labels, отдаёт torchvision-совместимые targets."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


VOC_NUM_CLASSES = 20
VOC_RARE_CLASSES = [0, 1, 2, 3, 4]   # aeroplane, bicycle, bird, boat, bottle
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Пути по умолчанию (зеркалят VOC.yaml)
VOC_ROOT = Path('/Users/maria/Documents/datasets/VOC')
TRAIN_SPLITS = ['train2007', 'train2012', 'val2007', 'val2012']
VAL_SPLITS   = ['test2007']


def _make_pairs(splits):
    """Возвращает список (img_path, lbl_path) для списка split-имён."""
    pairs = []
    for split in splits:
        img_dir = VOC_ROOT / 'images' / split
        lbl_dir = VOC_ROOT / 'labels' / split
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in _IMG_EXTS:
                lbl = (lbl_dir / p.name).with_suffix('.txt')
                if lbl.exists():
                    pairs.append((p, lbl))
    return pairs


class VOCFasterRCNNDataset(Dataset):
    """
    Датасет VOC для Faster R-CNN.

    Читает YOLO-формат: class cx cy w h (нормированные).
    Возвращает (img_tensor, target), где target — dict с:
      'boxes'    : (N,4) float32, [x1,y1,x2,y2] в пикселях
      'labels'   : (N,)  int64,   1-индексированные (0=фон)
      'orig_idx' : scalar tensor — глобальный индекс в датасете (для curriculum)
    """

    def __init__(self, splits, imgsz=640,
                 difficulty_type='original', noise_level=0.3,
                 imbalance_factor=0.3, artifact_level=0.4, artifact_type='blur'):
        self.imgsz           = imgsz
        self.difficulty_type = difficulty_type
        self.noise_level     = noise_level
        self.imbalance_factor = imbalance_factor
        self.artifact_level  = artifact_level
        self.artifact_type   = artifact_type

        self.samples = _make_pairs(splits)
        n = len(self.samples)

        self.sample_losses = torch.zeros(n)
        self.seen_counts   = torch.zeros(n, dtype=torch.long)

        if difficulty_type == 'imbalance':
            self._active = self._apply_imbalance()
        else:
            self._active = list(range(n))

        print(f'[VOCFasterRCNN] {difficulty_type} | {len(self._active)}/{n} images')

    # ── imbalance ──────────────────────────────────────────────────────────────

    def _apply_imbalance(self):
        keep = []
        for idx, (_, lbl) in enumerate(self.samples):
            classes = self._read_classes(lbl)
            if not classes:
                keep.append(idx)
                continue
            majority = max(set(classes), key=classes.count)
            if majority in VOC_RARE_CLASSES:
                if random.random() > self.imbalance_factor:
                    keep.append(idx)
            else:
                keep.append(idx)
        return keep

    @staticmethod
    def _read_classes(lbl_path):
        try:
            return [int(l.split()[0]) for l in lbl_path.read_text().splitlines() if l.strip()]
        except Exception:
            return []

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self._active)

    def __getitem__(self, index):
        real_idx = self._active[index]
        img_path, lbl_path = self.samples[real_idx]

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)

        if self.difficulty_type == 'artifacts':
            img = self._apply_artifact(img)

        img_t = TF.to_tensor(img)

        # Конвертация YOLO-меток → pixel xyxy
        boxes, labels = [], []
        sx, sy = self.imgsz / orig_w, self.imgsz / orig_h
        try:
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = max(0., (cx - w / 2) * orig_w * sx)
                y1 = max(0., (cy - h / 2) * orig_h * sy)
                x2 = min(float(self.imgsz), (cx + w / 2) * orig_w * sx)
                y2 = min(float(self.imgsz), (cy + h / 2) * orig_h * sy)
                if x2 > x1 + 1 and y2 > y1 + 1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)   # 0=background в torchvision
        except Exception:
            pass

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        if self.difficulty_type == 'noisy' and len(labels_t) > 0:
            labels_t = self._corrupt_labels(labels_t)

        target = {
            'boxes':    boxes_t,
            'labels':   labels_t,
            'orig_idx': torch.tensor(real_idx, dtype=torch.long),
        }
        return img_t, target

    # ── difficulty helpers ─────────────────────────────────────────────────────

    def _corrupt_labels(self, labels):
        labels = labels.clone()
        for i in range(len(labels)):
            if random.random() < self.noise_level:
                orig = int(labels[i].item())
                choices = [c for c in range(1, VOC_NUM_CLASSES + 1) if c != orig]
                labels[i] = random.choice(choices)
        return labels

    def _apply_artifact(self, img):
        art = (random.choice(['blur', 'noise', 'low_res'])
               if self.artifact_type == 'mixed' else self.artifact_type)
        if art == 'blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=1 + self.artifact_level * 4))
        elif art == 'noise':
            arr = np.array(img).astype(np.float32)
            arr = np.clip(arr + np.random.normal(0, self.artifact_level * 50, arr.shape), 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))
        elif art == 'low_res':
            scale = max(0.1, 1 - self.artifact_level * 0.9)
            small = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
            img = small.resize((img.width, img.height), Image.NEAREST)
        return img

    # ── Curriculum interface ───────────────────────────────────────────────────

    def update_losses(self, idxs, losses):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        if isinstance(losses, torch.Tensor):
            losses = losses.tolist()
        elif isinstance(losses, (int, float)):
            losses = [losses] * len(idxs)
        for i, l in zip(idxs, losses):
            if 0 <= i < len(self.sample_losses):
                self.sample_losses[i] = float(l)
                self.seen_counts[i] += 1

    def ranked_indices(self):
        active = torch.tensor(self._active, dtype=torch.long)
        return torch.argsort(self.sample_losses[active])

    @staticmethod
    def collate_fn(batch):
        imgs, targets = zip(*batch)
        return list(imgs), list(targets)
