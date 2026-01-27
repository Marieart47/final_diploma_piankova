import torch
import pandas as pd
import os
import argparse
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import json
from datetime import datetime

# Импортируем нашу фабрику датасетов
from datasets.difficult_datasets import create_dataset

from curriculum.strategies import Baseline, StageWise, Static, Online
from models.classification import resnet18, efficientnet_b0
from training.classification_trainer import ClassificationTrainer

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Run classification experiments with different dataset difficulties')
parser.add_argument('--dataset_type', type=str, default='original',
                   choices=['original', 'imbalance', 'noisy', 'artifacts'],
                   help='Type of dataset to use')
parser.add_argument('--imbalance_factor', type=float, default=0.5,
                   help='Imbalance factor for class imbalance dataset')
parser.add_argument('--noise_level', type=float, default=0.3,
                   help='Noise level for noisy label dataset')
parser.add_argument('--artifact_level', type=float, default=0.5,
                   help='Artifact level for visual artifacts dataset')
parser.add_argument('--epochs', type=int, default=10,
                   help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128,
                   help='Batch size for training')
parser.add_argument('--models', type=str, default='resnet18',
                   help='Comma-separated list of models to train')
parser.add_argument('--strategies', type=str, default='baseline,stagewise,static,online',
                   help='Comma-separated list of strategies to use')

args = parser.parse_args()

# Создаем папки для сохранения результатов
exp_name = f"{args.dataset_type}_im{args.imbalance_factor}_noise{args.noise_level}_art{args.artifact_level}"
results_dir = f"results/{exp_name}"
checkpoints_dir = f"checkpoints/{exp_name}"
logs_dir = f"training_logs/{exp_name}"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Парсим списки моделей и стратегий
model_names = args.models.split(',')
strategy_names = args.strategies.split(',')

models = {}
if 'resnet18' in model_names:
    models["resnet18"] = resnet18
if 'efficientnet' in model_names:
    models["efficientnet"] = efficientnet_b0

strategies = {}
if 'baseline' in strategy_names:
    strategies["baseline"] = Baseline()
if 'stagewise' in strategy_names:
    strategies["stagewise"] = StageWise()
if 'static' in strategy_names:
    strategies["static"] = Static()
if 'online' in strategy_names:
    strategies["online"] = Online()

results = []
all_metrics = []

# Функция для сохранения чекпоинта
def save_checkpoint(model, optimizer, epoch, loss, acc, model_name, strat_name, 
                   best=False, final=False):
    
    checkpoint_dir = f"{checkpoints_dir}/{model_name}_{strat_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'model_name': model_name,
        'strategy_name': strat_name,
        'timestamp': datetime.now().isoformat()
    }
    
    if best:
        filename = f"{checkpoint_dir}/best_model.pth"
    elif final:
        filename = f"{checkpoint_dir}/final_model_epoch_{epoch}.pth"
    else:
        filename = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
    
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint: {filename}")

# Функция для загрузки чекпоинта
def load_checkpoint(model, optimizer, model_name, strat_name, epoch=None, best=False):
    
    checkpoint_dir = f"{checkpoints_dir}/{model_name}_{strat_name}"
    
    if best and os.path.exists(f"{checkpoint_dir}/best_model.pth"):
        checkpoint_path = f"{checkpoint_dir}/best_model.pth"
    elif epoch is not None and os.path.exists(f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"):
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
    else:
        print(f"No checkpoint found for {model_name}_{strat_name}")
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"loss: {checkpoint['loss']:.4f}, acc: {checkpoint['accuracy']:.4f}")
    
    return checkpoint['epoch'] + 1

# Функция для сохранения метрик обучения
def save_training_metrics(model_name, strat_name, epoch, loss, acc, phase='train'):
    
    metrics_dir = f"{logs_dir}/{model_name}_{strat_name}"
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = f"{metrics_dir}/metrics.csv"
    
    metrics_entry = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'strategy': strat_name,
        'epoch': epoch,
        'loss': float(loss),
        'accuracy': float(acc),
        'phase': phase,
        'dataset_type': args.dataset_type,
        'imbalance_factor': args.imbalance_factor if args.dataset_type == 'imbalance' else 0,
        'noise_level': args.noise_level if args.dataset_type == 'noisy' else 0,
        'artifact_level': args.artifact_level if args.dataset_type == 'artifacts' else 0
    }
    
    all_metrics.append(metrics_entry)
    
    df_metrics = pd.DataFrame([metrics_entry])
    if os.path.exists(metrics_file):
        df_metrics.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        df_metrics.to_csv(metrics_file, index=False)

# Сохраняем конфигурацию эксперимента
experiment_config = {
    'device': device,
    'dataset_type': args.dataset_type,
    'imbalance_factor': args.imbalance_factor,
    'noise_level': args.noise_level,
    'artifact_level': args.artifact_level,
    'models': list(models.keys()),
    'strategies': list(strategies.keys()),
    'total_epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': 1e-4,
    'start_time': datetime.now().isoformat()
}

config_file = f"{results_dir}/experiment_config.json"
with open(config_file, 'w') as f:
    json.dump(experiment_config, f, indent=2)

print(f"Starting experiment: {exp_name}")
print(f"Device: {device}")
print(f"Configuration: {experiment_config}")

n_epochs = args.epochs
best_results = {}

# Основной цикл обучения
# Основной цикл обучения (ИСПРАВЛЕННАЯ ВЕРСИЯ)
for model_name, model_fn in models.items():
    base_ds = CIFAR10("./data", train=True, download=True, transform=ToTensor())
    
    # Создаем датасет с заданными сложностями
    ds = create_dataset(
        base_ds,
        dataset_type=args.dataset_type,
        imbalance_factor=args.imbalance_factor,
        noise_level=args.noise_level,
        artifact_level=args.artifact_level
    )
    
    for strat_name, strat in strategies.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name} with {strat_name} strategy")
        print(f"Dataset type: {args.dataset_type}")
        if args.dataset_type == 'imbalance':
            print(f"Imbalance factor: {args.imbalance_factor}")
        elif args.dataset_type == 'noisy':
            print(f"Noise level: {args.noise_level}")
        elif args.dataset_type == 'artifacts':
            print(f"Artifact level: {args.artifact_level}")
        print(f"{'='*60}")
        
        model = model_fn(10)
        opt = torch.optim.Adam(model.parameters(), 1e-4)
        trainer = ClassificationTrainer(model, opt, torch.nn.CrossEntropyLoss(), device)
        
        start_epoch = load_checkpoint(model, opt, model_name, strat_name, best=False)
        
        best_acc = 0.0
        best_loss = float('inf')
        
        # Инициализация для разных стратегий
        if strat_name == "static":
            # Для Static: один раз обучаем на всем датасете для инициализации
            loader = DataLoader(ds, args.batch_size, shuffle=True)
            trainer.train_epoch(loader, ds)  # Собираем начальные потери
            strat.initialize(ds)  # Делим на группы
        
        elif strat_name == "stagewise":
            # Для StageWise: сбрасываем на начальный этап
            strat.stage = 0
            # Собираем начальные потери на всем датасете
            loader = DataLoader(ds, args.batch_size, shuffle=True)
            trainer.train_epoch(loader, ds)  # Важно для инициализации ranked_indices()
            print(f"StageWise: starting at stage {strat.stage}")
        
        # Определяем моменты смены этапов для StageWise
        stage_change_epochs = []
        if strat_name == "stagewise":
            # Меняем этапы каждые n_epochs/3 эпох (для 10 эпох: после 3, 6, 9)
            for i in range(1, 4):  # 3 смены этапов
                change_epoch = (n_epochs * i) // 3
                if change_epoch < n_epochs:
                    stage_change_epochs.append(change_epoch)
            print(f"StageWise will change stages at epochs: {stage_change_epochs}")
        
        # Цикл обучения
        for epoch in range(start_epoch, n_epochs):
            if strat_name == "stagewise" and epoch in stage_change_epochs:
                # 1. Сначала обновляем потери на всех данных
                full_loader = DataLoader(ds, args.batch_size, shuffle=True)
                full_loss, _ = trainer.train_epoch(full_loader, ds)
                print(f"StageWise: updated losses on full dataset before stage change")
                
                # 2. Переходим на следующий этап
                strat.step()
                print(f"StageWise: moved to stage {strat.stage} at epoch {epoch}")
            
            # Получаем подмножество данных согласно стратегии
            if strat_name == "static":
                # Для Static: меняем группу каждые 10 эпох
                stage = min(epoch // max(1, n_epochs // 3), 2)
                sub = strat.get_dataset(ds, stage)
                if epoch == start_epoch or epoch % max(1, n_epochs // 3) == 0:
                    print(f"Static: using group {stage} (epoch {epoch})")
            
            elif strat_name == "stagewise":
                # Для StageWise: берем данные текущего этапа
                sub = strat.get_dataset(ds)
                # Выводим информацию о размере подмножества
                if epoch == start_epoch or epoch in stage_change_epochs:
                    n_total = len(ds)
                    n_subset = len(sub)
                    s, e = strat.stages[strat.stage], strat.stages[strat.stage+1]
                    print(f"StageWise: stage {strat.stage}, data range: [{s:.0%}-{e:.0%}], "
                          f"samples: {n_subset}/{n_total}")
            
            else:
                # Для Baseline и Online
                sub = strat.get_dataset(ds)
            
            # Создаем DataLoader и обучаем
            loader = DataLoader(sub, args.batch_size, shuffle=True)
            loss, acc = trainer.train_epoch(loader, ds)
            
            # Сохраняем результаты
            results.append([
                exp_name,
                model_name, 
                strat_name, 
                epoch, 
                loss, 
                acc,
                args.dataset_type,
                args.imbalance_factor if args.dataset_type == 'imbalance' else 0,
                args.noise_level if args.dataset_type == 'noisy' else 0,
                args.artifact_level if args.dataset_type == 'artifacts' else 0
            ])
            save_training_metrics(model_name, strat_name, epoch, loss, acc)
            
            # Сохраняем чекпоинт
            if epoch % 2 == 0 or epoch == n_epochs - 1:
                save_checkpoint(model, opt, epoch, loss, acc, model_name, strat_name)
            
            # Сохраняем лучшую модель
            if acc > best_acc:
                best_acc = acc
                best_loss = loss
                save_checkpoint(model, opt, epoch, loss, acc, model_name, strat_name, best=True)
                
                best_results[f"{model_name}_{strat_name}"] = {
                    'epoch': epoch,
                    'accuracy': float(acc),
                    'loss': float(loss),
                    'timestamp': datetime.now().isoformat()
                }
            
            print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f} | Best acc: {best_acc:.4f}")
           
        # Сохраняем финальную модель
        save_checkpoint(model, opt, n_epochs - 1, loss, acc, model_name, strat_name, final=True)
        
        # Сохраняем сводку
        training_summary = {
            'model': model_name,
            'strategy': strat_name,
            'dataset_type': args.dataset_type,
            'best_accuracy': float(best_acc),
            'best_loss': float(best_loss),
            'final_accuracy': float(acc),
            'final_loss': float(loss),
            'training_time': datetime.now().isoformat()
        }
        
        summary_file = f"{logs_dir}/{model_name}_{strat_name}/summary.json"
        with open(summary_file, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"Completed: {model_name}_{strat_name}")
        print(f"Best accuracy: {best_acc:.4f}, Final accuracy: {acc:.4f}")

# Сохраняем все результаты
df_results = pd.DataFrame(results, columns=[
    "experiment", "model", "strategy", "epoch", "loss", "acc",
    "dataset_type", "imbalance_factor", "noise_level", "artifact_level"
])
df_results.to_csv(f"{results_dir}/classification_results.csv", index=False)

df_all_metrics = pd.DataFrame(all_metrics)
df_all_metrics.to_csv(f"{results_dir}/all_training_metrics.csv", index=False)

if best_results:
    df_best = pd.DataFrame.from_dict(best_results, orient='index')
    df_best.to_csv(f"{results_dir}/best_results_summary.csv")

# Обновляем итоговую конфигурацию
experiment_config['end_time'] = datetime.now().isoformat()
experiment_config['best_results'] = best_results

with open(f"{results_dir}/experiment_final_config.json", 'w') as f:
    json.dump(experiment_config, f, indent=2)

print(f"\n{'='*60}")
print(f"Experiment {exp_name} completed!")
print(f"Results saved in: {results_dir}")
print(f"Checkpoints saved in: {checkpoints_dir}")
print(f"Training logs saved in: {logs_dir}")
print(f"{'='*60}")

print("\nBest Results Summary:")
for key, value in best_results.items():
    print(f"{key}: Epoch {value['epoch']}, Accuracy: {value['accuracy']:.4f}")