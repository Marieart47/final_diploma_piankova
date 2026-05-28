"""Датасет для Super Resolution: LR/HR пары из папки с HR-изображениями."""

import io
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Типы деградации LR-изображений
DEGRADATION_TYPES = ("original", "noise", "blur", "jpeg", "mixed")


def _apply_degradation(lr: torch.Tensor, deg_type: str, deg_level: float) -> torch.Tensor:
    """
    Применяет деградацию к LR-тензору [C, H, W] в диапазоне [0, 1].

    Типы:
      original — без изменений
      noise    — гауссовский шум (std = deg_level * 0.10)
      blur     — размытие по Гауссу (sigma = deg_level * 2.5)
      jpeg     — JPEG-компрессия (quality = 100 - int(deg_level * 80))
      mixed    — случайная комбинация двух из трёх выше
    """
    if deg_type == "original" or deg_level == 0.0:
        return lr

    if deg_type == "mixed":
        # случайно выбираем 1–2 типа
        choices = random.sample(["noise", "blur", "jpeg"], k=random.randint(1, 2))
        for t in choices:
            lr = _apply_degradation(lr, t, deg_level)
        return lr

    if deg_type == "noise":
        std = deg_level * 0.10
        return (lr + torch.randn_like(lr) * std).clamp(0.0, 1.0)

    if deg_type == "blur":
        # kernel_size должен быть нечётным >= 3
        ks = 2 * max(1, round(deg_level * 3)) + 1   # [3, 7] при level in [0,1]
        sigma = deg_level * 2.5 + 0.5
        return TF.gaussian_blur(lr, kernel_size=ks, sigma=sigma)

    if deg_type == "jpeg":
        quality = max(10, 100 - int(deg_level * 80))  # [20, 100]
        pil_img = TF.to_pil_image(lr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return TF.to_tensor(Image.open(buf).convert("RGB"))

    raise ValueError(f"Unknown degradation type: {deg_type!r}. Use one of {DEGRADATION_TYPES}.")


class SRDataset(Dataset):
    """
    Загружает HR-изображения из папки, создаёт LR-версии билинейным даунскейлом
    и опционально применяет деградацию к LR (noise / blur / jpeg / mixed).

    При обучении нарезает случайные патчи patch_size × patch_size + аугментация.
    При валидации/тесте возвращает изображение целиком.
    """

    def __init__(
        self,
        hr_dir: str,
        scale: int = 4,
        patch_size: int = 128,
        training: bool = True,
        augment: bool = True,
        degradation: str = "original",
        deg_level: float = 0.0,
    ):
        assert degradation in DEGRADATION_TYPES, \
            f"degradation must be one of {DEGRADATION_TYPES}, got {degradation!r}"
        assert 0.0 <= deg_level <= 1.0, "deg_level must be in [0.0, 1.0]"

        self.hr_dir     = Path(hr_dir)
        self.scale      = scale
        self.patch_size = patch_size
        self.training   = training
        self.augment    = augment
        self.deg_type   = degradation
        self.deg_level  = deg_level

        self.hr_paths = sorted(
            p for p in self.hr_dir.rglob("*")
            if p.suffix.lower() in _EXTS
        )
        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir!r}")

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int):
        hr = Image.open(self.hr_paths[idx]).convert("RGB")

        if self.training:
            hr = self._random_crop(hr, self.patch_size)
            if self.augment:
                hr = self._augment(hr)
        else:
            w, h = hr.size
            w = (w // self.scale) * self.scale
            h = (h // self.scale) * self.scale
            hr = hr.crop((0, 0, w, h))

        lr_pil = hr.resize(
            (hr.width // self.scale, hr.height // self.scale),
            Image.BILINEAR,
        )

        lr_t = TF.to_tensor(lr_pil)
        hr_t = TF.to_tensor(hr)

        # Деградация применяется только к LR
        if self.deg_type != "original" and self.deg_level > 0.0:
            lr_t = _apply_degradation(lr_t, self.deg_type, self.deg_level)

        return lr_t, hr_t

    @staticmethod
    def _random_crop(img: Image.Image, size: int) -> Image.Image:
        w, h = img.size
        min_side = min(w, h)
        if min_side < size:
            s = size / min_side + 0.01
            img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
            w, h = img.size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return img.crop((x, y, x + size, y + size))

    @staticmethod
    def _augment(img: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)
        angle = random.choice([0, 90, 180, 270])
        if angle:
            img = img.rotate(angle, expand=True)
        return img
