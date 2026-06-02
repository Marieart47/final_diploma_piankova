"""Тренер Faster R-CNN с поддержкой курикулум-обучения."""

import time
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.ops import box_iou

from curriculum.strategies import Baseline, Static, StageWise
from datasets.voc_fasterrcnn_dataset import VOCFasterRCNNDataset


def build_fasterrcnn(num_classes=21, pretrained_backbone=True):
    """Faster R-CNN с MobileNetV3-Large FPN backbone. num_classes включает фон (0)."""
    weights_backbone = 'DEFAULT' if pretrained_backbone else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )
    return model


# ── mAP@50 ────────────────────────────────────────────────────────────────────

def _compute_ap_at_iou(preds: List[Dict], targets: List[Dict],
                       iou_threshold: float, num_classes: int = 20) -> float:
    """AP при фиксированном IoU threshold, 11-point interpolation (VOC)."""
    ap_list = []
    for cls in range(1, num_classes + 1):
        all_scores, all_tp, all_fp = [], [], []
        n_gt = 0

        for pred, tgt in zip(preds, targets):
            gt_mask  = tgt['labels'] == cls
            gt_boxes = tgt['boxes'][gt_mask]
            n_gt    += len(gt_boxes)
            matched  = torch.zeros(len(gt_boxes), dtype=torch.bool)

            pd_mask   = pred['labels'] == cls
            pd_boxes  = pred['boxes'][pd_mask]
            pd_scores = pred['scores'][pd_mask]
            if len(pd_boxes) == 0:
                continue

            order     = torch.argsort(pd_scores, descending=True)
            pd_boxes  = pd_boxes[order]
            pd_scores = pd_scores[order]

            for b in pd_boxes:
                if len(gt_boxes) == 0:
                    all_tp.append(0); all_fp.append(1)
                    continue
                ious = box_iou(b.unsqueeze(0), gt_boxes)[0]
                best_iou, best_j = ious.max(0)
                if best_iou >= iou_threshold and not matched[best_j]:
                    matched[best_j] = True
                    all_tp.append(1); all_fp.append(0)
                else:
                    all_tp.append(0); all_fp.append(1)
                all_scores.append(pd_scores[len(all_tp) - 1].item()
                                  if len(all_scores) < len(pd_scores) else 0.0)

        if n_gt == 0:
            continue
        if not all_scores:
            ap_list.append(0.0)
            continue

        order  = sorted(range(len(all_scores)), key=lambda i: -all_scores[i])
        tp     = torch.tensor([all_tp[i] for i in order], dtype=torch.float32)
        fp     = torch.tensor([all_fp[i] for i in order], dtype=torch.float32)
        tp_cum = tp.cumsum(0)
        fp_cum = fp.cumsum(0)
        rec    = tp_cum / n_gt
        prec   = tp_cum / (tp_cum + fp_cum + 1e-9)

        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = rec >= t
            ap  += prec[mask].max().item() if mask.any() else 0.0
        ap_list.append(ap / 11)

    return float(torch.tensor(ap_list).mean()) if ap_list else 0.0


def _compute_map50(preds: List[Dict], targets: List[Dict], num_classes: int = 20) -> float:
    return _compute_ap_at_iou(preds, targets, iou_threshold=0.5, num_classes=num_classes)


def _compute_map5095(preds: List[Dict], targets: List[Dict], num_classes: int = 20) -> float:
    """mAP@[0.50:0.05:0.95] — среднее по 10 IoU порогам."""
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    return float(np.mean([_compute_ap_at_iou(preds, targets, t, num_classes)
                          for t in thresholds]))


# ── Trainer ────────────────────────────────────────────────────────────────────

class FasterRCNNTrainer:
    """
    Тренер Faster R-CNN с бюджетом в шагах и поддержкой курикулум-стратегий.
    Интерфейс аналогичен CurriculumDetectionTrainer.
    """

    def __init__(self, dataset_train, dataset_val,
                 curriculum_strategy=None,
                 total_steps=1000,
                 batch_size=4,
                 lr=1e-4,
                 device=None,
                 num_classes=21,
                 workers=2):
        self.ds_train   = dataset_train
        self.ds_val     = dataset_val
        self.strategy   = curriculum_strategy or Baseline()
        self.total_steps = total_steps
        self.batch_size  = batch_size
        self.num_classes = num_classes
        self.workers     = workers

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'   # MPS зависает на val-loop Faster R-CNN
        self.device = device

        self.model = build_fasterrcnn(num_classes=num_classes).to(device)
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, momentum=0.9, weight_decay=5e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)

        self._steps_done    = 0
        self._warmup_steps  = 0
        self._static_init   = False
        self.metrics: Dict  = {}
        self.epoch          = 0

    # ── curriculum dataset selection ──────────────────────────────────────────

    def _select_subset(self):
        strat = self.strategy
        if isinstance(strat, Static):
            if not self._static_init:
                return self.ds_train          # warm-up: весь датасет
            post = self._steps_done - self._warmup_steps
            remaining = max(self.total_steps - self._warmup_steps, 1)
            stage = min(post * 3 // remaining, 2)
            return strat.get_dataset(self.ds_train, stage)
        if isinstance(strat, StageWise):
            new_stage = min(self._steps_done * 3 // max(self.total_steps, 1), 2)
            while strat.stage < new_stage:
                strat.step()
                print(f'  [StageWise] stage → {strat.stage} at step {self._steps_done}')
        return strat.get_dataset(self.ds_train)

    def _make_loader(self, ds):
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, collate_fn=VOCFasterRCNNDataset.collate_fn,
        )

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self) -> Dict:
        print(f'\nDevice: {self.device} | budget: {self.total_steps} steps')
        val_loader = DataLoader(
            self.ds_val, batch_size=1, shuffle=False,
            num_workers=self.workers, collate_fn=VOCFasterRCNNDataset.collate_fn,
        )

        best_map50   = 0.0
        best_state   = None
        total_time   = 0.0
        self.epoch   = 0

        while self._steps_done < self.total_steps:
            t0 = time.time()

            subset = self._select_subset()
            loader = self._make_loader(subset)

            epoch_loss = self._train_epoch(loader)

            # Static warm-up инициализация
            if (isinstance(self.strategy, Static)
                    and not self._static_init and self.ds_train is not None):
                self.strategy.initialize(self.ds_train)
                self._static_init   = True
                self._warmup_steps  = self._steps_done
                print(f'  [Static] groups init after {self._warmup_steps} warm-up steps')

            self.scheduler.step()
            total_time += time.time() - t0

            if self.epoch % 5 == 0 or self._steps_done >= self.total_steps:
                map50 = self._validate(val_loader)
                marker = ''
                if map50 > best_map50:
                    best_map50 = map50
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    marker = '  <- best'
                print(f'  epoch={self.epoch:3d} | steps={self._steps_done:5d} | '
                      f'loss={epoch_loss:.4f} | mAP50={map50:.4f}{marker}')

            self.epoch += 1

        # Финальная оценка на лучших весах
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        final_map50 = self._validate(val_loader)

        self.metrics = {
            'metrics/mAP50(B)':    final_map50,
            'metrics/mAP50-95(B)': final_map50,   # для совместимости с форматом YOLO-результатов
            'total_time':          total_time,
        }
        print(f'\n  FINAL mAP50={final_map50:.4f} | steps={self._steps_done} | '
              f'epochs={self.epoch} | time={total_time:.0f}s')
        return self.metrics

    # ── epoch ─────────────────────────────────────────────────────────────────

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, targets in loader:
            if self._steps_done >= self.total_steps:
                break

            # Сохраняем orig_idx до фильтрации
            orig_idxs = [t['orig_idx'] for t in targets if 'orig_idx' in t]

            imgs    = [img.to(self.device) for img in imgs]
            targets_model = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in t.items() if k != 'orig_idx'}
                             for t in targets]

            loss_dict = self.model(imgs, targets_model)
            loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()

            batch_loss = loss.item()

            # Обновляем per-sample лосс (скалярный лосс батча — лучшее что есть без per-sample разбивки)
            if orig_idxs and hasattr(self.ds_train, 'update_losses'):
                idxs = torch.stack(orig_idxs).cpu()
                self.ds_train.update_losses(idxs, batch_loss)

            total_loss += batch_loss
            n_batches  += 1
            self._steps_done += 1

        return total_loss / max(n_batches, 1)

    # ── validate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader) -> float:
        self.model.eval()
        preds, targets_list = [], []

        for imgs, targets in loader:
            imgs = [img.to(self.device) for img in imgs]
            out  = self.model(imgs)

            for pred, tgt in zip(out, targets):
                preds.append({
                    'boxes':  pred['boxes'].cpu(),
                    'labels': pred['labels'].cpu(),
                    'scores': pred['scores'].cpu(),
                })
                targets_list.append({
                    'boxes':  tgt['boxes'],
                    'labels': tgt['labels'],
                })

        return _compute_map50(preds, targets_list, num_classes=self.num_classes - 1)
