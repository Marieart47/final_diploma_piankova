import torch
from utils.metrics import accuracy
import torch.nn as nn
import numpy as np

# class ClassificationTrainer:
    # def __init__(self, model, optimizer, criterion, device):
    #     self.model = model.to(device)
    #     self.opt = optimizer
    #     self.crit = criterion
    #     self.device = device
    

#     def train_epoch(self, loader, dataset):
#         self.model.train()
#         total_loss, total_acc = 0, 0

#         for (x,y), idx in loader:
#             x,y = x.to(self.device), y.to(self.device)
#             out = self.model(x)
#             loss = self.crit(out, y)

#             self.opt.zero_grad()
#             loss.backward()
#             self.opt.step()

#             dataset.update_losses(idx, loss.detach())
#             total_loss += loss.item()
#             total_acc += accuracy(out, y)

#         return total_loss/len(loader), total_acc/len(loader)


class ClassificationTrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def train_epoch(self, loader, dataset):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        losses_batch = []
        idxs_batch = []
        
        for batch in loader:
            if len(batch) == 2:
                (x, y), idx = batch
            elif len(batch) == 3:
                x, y, idx = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            x, y = x.to(self.device), y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += x.size(0)
            
            # Store for dataset update
            batch_size = x.size(0)
            losses_batch.append(loss.detach().cpu().expand(batch_size))

            if isinstance(idx, torch.Tensor):
                if idx.dim() == 0:  
                    idx = idx.unsqueeze(0)
                idxs_batch.append(idx.cpu())
            elif isinstance(idx, (int, np.integer)):
                idxs_batch.append(torch.tensor([idx]))
            else:
                idxs_batch.append(torch.tensor(idx))
        
        # Update dataset with losses
        if losses_batch and hasattr(dataset, 'update_losses'):
            try:
                losses_all = torch.cat(losses_batch)
                idxs_all = torch.cat(idxs_batch)
                
                # Убедимся, что размеры совпадают
                if len(losses_all) == len(idxs_all):
                    dataset.update_losses(idxs_all, losses_all)
                else:
                    print(f"Warning: Mismatched sizes - losses: {len(losses_all)}, idxs: {len(idxs_all)}")
            except Exception as e:
                print(f"Warning: Could not update dataset losses: {e}")
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy