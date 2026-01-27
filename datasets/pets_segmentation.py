import os
import cv2
import torch
from torch.utils.data import Dataset

class PetsSegmentationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(self.mask_dir, img_name.replace(".jpg", ".png")),
            cv2.IMREAD_GRAYSCALE
        )

        # original labels: 1,2,3 → convert to 0,1
        mask = (mask > 1).astype("int64")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, torch.tensor(mask)


import albumentations as A
from albumentations.pytorch import ToTensorV2

seg_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])
