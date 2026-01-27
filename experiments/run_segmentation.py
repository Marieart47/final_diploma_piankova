import torch
import pandas as pd
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from datasets.pets_segmentation import PetsSegmentationDataset
from datasets.curriculum_dataset import CurriculumDataset
from curriculum.strategies import Baseline, StageWise, Static, Online
from models.segmentation import unet, deeplab
from training.segmentation_trainer import SegmentationTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# CONFIG
# ======================
EPOCHS = 40
BATCH_SIZE = 4
LR = 1e-4
NUM_CLASSES = 2

models = {
    "unet": unet,
    "deeplab": deeplab
}

strategies = {
    "baseline": Baseline(),
    "stagewise": StageWise(),
    "static": Static(),
    "online": Online()
}

results = []

# ======================
# DATASET
# ======================
base_ds = PetsSegmentationDataset(
    root="data/pets",
    transform=seg_transform
)

for model_name, model_fn in models.items():
    curr_ds = CurriculumDataset(base_ds)

    for strat_name, strat in strategies.items():
        print(f"\nMODEL={model_name}, STRATEGY={strat_name}")

        model = model_fn(NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), LR)
        criterion = smp.losses.DiceLoss(mode="multiclass")

        trainer = SegmentationTrainer(
            model, optimizer, criterion, device, NUM_CLASSES
        )

        # ===== static init =====
        if strat_name == "static":
            loader = DataLoader(curr_ds, BATCH_SIZE, shuffle=True)
            trainer.train_epoch(loader, curr_ds)
            strat.initialize(curr_ds)

        for epoch in range(EPOCHS):
            if strat_name == "static":
                subset = strat.get_dataset(curr_ds, min(epoch // 15, 2))
            elif strat_name == "stagewise":
                if epoch in [15, 30]:
                    strat.step()
                subset = strat.get_dataset(curr_ds)
            else:
                subset = strat.get_dataset(curr_ds)

            loader = DataLoader(subset, BATCH_SIZE, shuffle=True)

            loss, miou = trainer.train_epoch(loader, curr_ds)

            results.append([
                model_name, strat_name, epoch, loss, miou
            ])

            print(
                f"Epoch {epoch:02d} | "
                f"loss={loss:.4f}, mIoU={miou:.4f}"
            )

# ======================
# SAVE
# ======================
df = pd.DataFrame(
    results,
    columns=["model", "strategy", "epoch", "loss", "miou"]
)
df.to_csv("segmentation_results.csv", index=False)
