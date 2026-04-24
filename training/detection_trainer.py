"""DetectionTrainer с поддержкой курикулум-обучения на основе ultralytics."""

import torch
from torch.utils.data import DataLoader, Subset

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.data.build import build_yolo_dataset

from datasets.detection_dataset import CurriculumDetectionDataset
from curriculum.strategies import Baseline, Static, StageWise


class CurriculumDetectionTrainer(DetectionTrainer):

    def __init__(self, overrides=None, curriculum_strategy=None,
                 total_steps=1000, budget_mode='steps',
                 difficulty_type='original', noise_level=0.3,
                 imbalance_factor=0.3, artifact_level=0.4, artifact_type='blur'):
        super().__init__(overrides=overrides)
        self.curriculum_strategy = curriculum_strategy or Baseline()
        self.total_steps = total_steps
        self.budget_mode = budget_mode
        # Difficulty injection parameters forwarded to CurriculumDetectionDataset
        self.difficulty_type   = difficulty_type
        self.noise_level       = noise_level
        self.imbalance_factor  = imbalance_factor
        self.artifact_level    = artifact_level
        self.artifact_type     = artifact_type
        self._steps_done = 0
        self._warmup_steps = 0     # Static: steps used by warm-up epoch (excluded from stage calc)
        self._curr_ds = None       # CurriculumDetectionDataset, set in build_dataset
        self._batch_idxs = None    # orig_idx from current batch
        self._static_initialized = False  # Static: warm-up done flag

        self.add_callback('on_train_batch_end',   lambda t: self._on_batch_end())
        self.add_callback('on_train_epoch_start', lambda t: self._apply_curriculum())
        self.add_callback('on_train_epoch_end',   lambda t: self._check_budget())

    def build_dataset(self, img_path, mode='train', batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)

        if mode == 'train':
            ds = CurriculumDetectionDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=True,
                hyp=self.args,
                rect=self.args.rect or False,
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=gs,
                pad=0.0,
                prefix=colorstr('train: '),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction,
                # Difficulty injection
                difficulty_type=self.difficulty_type,
                noise_level=self.noise_level,
                imbalance_factor=self.imbalance_factor,
                artifact_level=self.artifact_level,
                artifact_type=self.artifact_type,
            )
            self._curr_ds = ds
            LOGGER.info(f'  [Curriculum] Training dataset: {len(ds)} images')
            return ds

        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=True, stride=gs
        )

    def preprocess_batch(self, batch):
        self._batch_idxs = batch.get('orig_idx')
        return super().preprocess_batch(batch)

    def _on_batch_end(self):
        if self._curr_ds is None or self._batch_idxs is None:
            return
        batch_loss = float(self.loss.detach().cpu().item()) if hasattr(self, 'loss') else 0.0
        self._curr_ds.update_losses(self._batch_idxs.cpu(), batch_loss)
        self._steps_done += 1

    def _apply_curriculum(self):
        if self._curr_ds is None:
            return

        strat = self.curriculum_strategy

        if isinstance(strat, Static):
            if not self._static_initialized:
                LOGGER.info('  [Static] Warm-up epoch — using full dataset to initialize groups.')
                selected = self._curr_ds
            else:
                post_warmup = self._steps_done - self._warmup_steps
                remaining   = max(self.total_steps - self._warmup_steps, 1)
                stage = min(post_warmup * 3 // remaining, 2)
                selected = strat.get_dataset(self._curr_ds, stage)
        elif isinstance(strat, StageWise):
            # Advance stage at 1/3 and 2/3 of total step budget
            new_stage = min(self._steps_done * 3 // max(self.total_steps, 1), 2)
            while strat.stage < new_stage:
                strat.step()
                LOGGER.info(f'  [StageWise] Advanced to stage {strat.stage} at step {self._steps_done}')
            selected = strat.get_dataset(self._curr_ds)
        else:
            selected = strat.get_dataset(self._curr_ds)

        if isinstance(selected, Subset):
            indices = list(selected.indices)
        else:
            indices = list(range(len(self._curr_ds)))

        subset = Subset(self._curr_ds, indices)
        self.train_loader = DataLoader(
            subset,
            batch_size=self.args.batch,
            shuffle=True,
            num_workers=self.args.workers,
            collate_fn=CurriculumDetectionDataset.collate_fn,
            pin_memory=True,
        )
        LOGGER.info(
            f'  [Curriculum] {self.curriculum_strategy.__class__.__name__}: '
            f'{len(indices)}/{len(self._curr_ds)} samples selected'
        )

    def _check_budget(self):
        if (isinstance(self.curriculum_strategy, Static)
                and not self._static_initialized
                and self._curr_ds is not None):
            self.curriculum_strategy.initialize(self._curr_ds)
            self._static_initialized = True
            self._warmup_steps = self._steps_done  # record where warm-up ended
            LOGGER.info(f'  [Static] Difficulty groups initialized after warm-up epoch '
                        f'({self._warmup_steps} warm-up steps).')

        if self.budget_mode == 'steps' and self._steps_done >= self.total_steps:
            LOGGER.info(
                f'  [Budget] Exhausted ({self._steps_done}/{self.total_steps} steps). Stopping.'
            )
            self.stop = True
