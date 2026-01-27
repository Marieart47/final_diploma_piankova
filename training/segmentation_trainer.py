from utils.metrics import iou

class SegmentationTrainer:
    def __init__(self, model, optimizer, criterion, device, num_classes):
        self.model = model.to(device)
        self.opt = optimizer
        self.crit = criterion
        self.device = device
        self.num_classes = num_classes

    def train_epoch(self, loader, dataset):
        self.model.train()
        total_loss, total_iou = 0, 0

        for (img,mask), idx in loader:
            img,mask = img.to(self.device), mask.to(self.device)
            out = self.model(img)
            loss = self.crit(out, mask)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            dataset.update_losses(idx, loss.detach())
            total_loss += loss.item()
            total_iou += iou(out, mask, self.num_classes)

        return total_loss/len(loader), total_iou/len(loader)
