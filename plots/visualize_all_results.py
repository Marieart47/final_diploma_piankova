import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# Конфигурация
base_results_dir = "results"  # Основная папка с результатами
graphs_base_dir = "all_experiments_plots"  # Папка для всех графиков
os.makedirs(graphs_base_dir, exist_ok=True)

print(f"Поиск результатов в папке: {base_results_dir}")
print(f"Графики будут сохранены в папку: {graphs_base_dir}")

# Находим все CSV файлы с результатами
result_files = glob.glob(os.path.join(base_results_dir, "**", "classification_results.csv"), recursive=True)

if not result_files:
    print(f"Не найдено файлов results в папке {base_results_dir}")
    exit()

print(f"Найдено {len(result_files)} файлов с результатами:")

# Функция для парсинга имени эксперимента
def parse_experiment_name(exp_name):
    """
    Парсит имя эксперимента и извлекает параметры.
    Примеры:
      - "artifacts_im0.5_noise0.3_art0.3" → dataset_type='artifacts', art=0.3, im=0.5, noise=0.3
      - "imbalance_im0.3_noise0.3_art0.5" → dataset_type='imbalance', im=0.3, noise=0.3, art=0.5
      - "original_im0.5_noise0.3_art0.5" → dataset_type='original', все параметры=0
    """
    parts = exp_name.split('_')
    
    # Определяем основной тип датасета
    dataset_type = parts[0] if len(parts) > 0 else 'unknown'
    
    # Значения по умолчанию
    params = {
        'im': 0.0,
        'noise': 0.0,
        'art': 0.0
    }
    
    # Парсим параметры из всех частей
    for part in parts:
        if part.startswith('im') and len(part) > 2:
            try:
                params['im'] = float(part[2:])
            except:
                params['im'] = 0.0
        elif part.startswith('noise') and len(part) > 5:
            try:
                params['noise'] = float(part[5:])
            except:
                params['noise'] = 0.0
        elif part.startswith('art') and len(part) > 3:
            try:
                params['art'] = float(part[3:])
            except:
                params['art'] = 0.0
    
    # Для original датасета все параметры должны быть 0
    if dataset_type == 'original':
        params = {'im': 0.0, 'noise': 0.0, 'art': 0.0}
    
    return dataset_type, params

# Собираем все данные в один DataFrame
all_data = []
experiment_info = {}  # Сохраняем информацию об экспериментах

for file_path in result_files:
    # Извлекаем имя эксперимента из пути
    experiment_dir = os.path.dirname(file_path)
    experiment_name = os.path.basename(experiment_dir)
    
    # Парсим имя эксперимента
    dataset_type, params = parse_experiment_name(experiment_name)
    
    print(f"  - {experiment_name}")
    print(f"    Тип: {dataset_type}, Параметры: im={params['im']}, noise={params['noise']}, art={params['art']}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Добавляем колонку с именем эксперимента
        if 'experiment' not in df.columns:
            df['experiment'] = experiment_name
        
        # Добавляем информацию о типе датасета и параметрах
        df['dataset_type'] = dataset_type
        df['imbalance_factor'] = params['im']
        df['noise_level'] = params['noise']
        df['artifact_level'] = params['art']
        
        # Вычисляем основной параметр для этого типа датасета
        if dataset_type == 'imbalance':
            df['main_param'] = params['im']
            df['main_param_name'] = 'imbalance'
        elif dataset_type == 'noisy':
            df['main_param'] = params['noise']
            df['main_param_name'] = 'noise'
        elif dataset_type == 'artifacts':
            df['main_param'] = params['art']
            df['main_param_name'] = 'artifacts'
        else:  # original
            df['main_param'] = 0.0
            df['main_param_name'] = 'none'
        
        all_data.append(df)
        
        # Сохраняем информацию об эксперименте
        experiment_info[experiment_name] = {
            'dataset_type': dataset_type,
            'params': params,
            'file_path': file_path,
            'num_rows': len(df)
        }
        
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")

if not all_data:
    print("Нет данных для построения графиков")
    exit()

# Объединяем все данные
combined_df = pd.concat(all_data, ignore_index=True)

print(f"\nОбъединенный DataFrame содержит {len(combined_df)} строк")
print(f"Уникальные эксперименты: {len(experiment_info)}")
print(f"Уникальные модели: {combined_df.model.unique()}")
print(f"Уникальные стратегии: {combined_df.strategy.unique()}")
print(f"Типы датасетов: {combined_df.dataset_type.unique()}")

# ========== 1. ГРАФИКИ ПО ЭКСПЕРИМЕНТАМ ==========
print("\n" + "="*60)
print("Создание графиков по отдельным экспериментам")
print("="*60)

for experiment_name, info in experiment_info.items():
    exp_data = combined_df[combined_df.experiment == experiment_name]
    
    if exp_data.empty:
        continue
    
    # Создаем папку для этого эксперимента
    exp_graphs_dir = os.path.join(graphs_base_dir, experiment_name)
    os.makedirs(exp_graphs_dir, exist_ok=True)
    
    dataset_type = info['dataset_type']
    params = info['params']
    
    # Создаем подзаголовок с параметрами
    title_suffix = f" ({dataset_type}"
    if dataset_type == 'imbalance':
        title_suffix += f", imbalance={params['im']}"
    if params['noise'] > 0:
        title_suffix += f", noise={params['noise']}"
    if params['art'] > 0:
        title_suffix += f", artifacts={params['art']}"
    title_suffix += ")"
    
    # 1.1 Графики по моделям (точность)
    for model in exp_data.model.unique():
        plt.figure(figsize=(12, 8))
        
        for strat in exp_data.strategy.unique():
            d = exp_data[(exp_data.model == model) & (exp_data.strategy == strat)]
            
            if not d.empty:
                # Сортируем по эпохам
                d = d.sort_values('epoch')
                plt.plot(d.epoch, d.acc, marker='o', markersize=4, linewidth=2, label=strat)
        
        plt.title(f"Model: {model}{title_suffix}", fontsize=14, fontweight='bold')
        plt.legend(title="Strategy", fontsize=10, title_fontsize=11)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        filename = os.path.join(exp_graphs_dir, f"{model}_accuracy.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 1.2 Графики по стратегиям
    strategies_dir = os.path.join(exp_graphs_dir, "by_strategy")
    os.makedirs(strategies_dir, exist_ok=True)
    
    for strat in exp_data.strategy.unique():
        plt.figure(figsize=(12, 8))
        
        for model in exp_data.model.unique():
            d = exp_data[(exp_data.model == model) & (exp_data.strategy == strat)]
            
            if not d.empty:
                d = d.sort_values('epoch')
                plt.plot(d.epoch, d.acc, marker='s', markersize=4, linewidth=2, label=model)
        
        plt.title(f"Strategy: {strat}{title_suffix}", fontsize=14, fontweight='bold')
        plt.legend(title="Model", fontsize=10, title_fontsize=11)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        filename = os.path.join(strategies_dir, f"strategy_{strat}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 1.3 Графики потерь
    loss_dir = os.path.join(exp_graphs_dir, "loss_plots")
    os.makedirs(loss_dir, exist_ok=True)
    
    for model in exp_data.model.unique():
        plt.figure(figsize=(12, 8))
        
        for strat in exp_data.strategy.unique():
            d = exp_data[(exp_data.model == model) & (exp_data.strategy == strat)]
            
            if not d.empty:
                d = d.sort_values('epoch')
                plt.plot(d.epoch, d.loss, marker='o', markersize=4, linewidth=2, label=strat)
        
        plt.title(f"Model: {model}{title_suffix} - Loss", fontsize=14, fontweight='bold')
        plt.legend(title="Strategy", fontsize=10, title_fontsize=11)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        filename = os.path.join(loss_dir, f"{model}_loss.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Графики для эксперимента '{experiment_name}' сохранены в {exp_graphs_dir}")

# ========== 2. СРАВНИТЕЛЬНЫЕ ГРАФИКИ ПО ТИПАМ ДАННЫХ ==========
print("\n" + "="*60)
print("Создание сравнительных графиков по типам данных")
print("="*60)

comparison_dir = os.path.join(graphs_base_dir, "comparisons")
os.makedirs(comparison_dir, exist_ok=True)

# 2.1 Сравнение стратегий для каждого типа данных и параметра
for dataset_type in combined_df.dataset_type.unique():
    if dataset_type == 'original':
        continue  # Пропускаем original, там нет параметров
    
    type_data = combined_df[combined_df.dataset_type == dataset_type]
    
    if type_data.empty:
        continue
    
    # Получаем уникальные значения основного параметра
    param_values = sorted(type_data['main_param'].unique())
    
    for param_value in param_values:
        param_data = type_data[type_data.main_param == param_value]
        
        if param_data.empty:
            continue
        
        plt.figure(figsize=(16, 10))
        
        # Создаем subplot для каждой модели
        models = param_data.model.unique()
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6), sharey=True)
        if n_models == 1:
            axes = [axes]
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            model_data = param_data[param_data.model == model]
            
            for strat in model_data.strategy.unique():
                strat_data = model_data[model_data.strategy == strat]
                
                # Группируем по эксперименту (может быть несколько)
                for exp in strat_data.experiment.unique():
                    exp_data = strat_data[strat_data.experiment == exp].sort_values('epoch')
                    
                    if not exp_data.empty:
                        # Получаем все параметры для подписи
                        exp_info = experiment_info.get(exp, {})
                        exp_params = exp_info.get('params', {})
                        
                        label = strat
                        # Добавляем дополнительные параметры если они есть
                        extra_params = []
                        if exp_params.get('im', 0) > 0 and dataset_type != 'imbalance':
                            extra_params.append(f"im={exp_params['im']}")
                        if exp_params.get('noise', 0) > 0 and dataset_type != 'noisy':
                            extra_params.append(f"noise={exp_params['noise']}")
                        if exp_params.get('art', 0) > 0 and dataset_type != 'artifacts':
                            extra_params.append(f"art={exp_params['art']}")
                        
                        if extra_params:
                            label += f" ({', '.join(extra_params)})"
                        
                        ax.plot(exp_data.epoch, exp_data.acc, linewidth=2, label=label)
            
            ax.set_title(f"Model: {model}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Accuracy", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1)
        
        param_name = param_data['main_param_name'].iloc[0]
        plt.suptitle(f"Dataset: {dataset_type} ({param_name}={param_value}) - Strategy Comparison", 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = os.path.join(comparison_dir, f"{dataset_type}_{param_name}{param_value}_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Сравнительный график для {dataset_type} ({param_name}={param_value}) сохранен")

# 2.2 Сравнение влияния параметров для каждой стратегии
for dataset_type in ['imbalance', 'noisy', 'artifacts']:
    type_data = combined_df[combined_df.dataset_type == dataset_type]
    
    if type_data.empty:
        continue
    
    param_name = type_data['main_param_name'].iloc[0]
    param_values = sorted(type_data['main_param'].unique())
    
    if len(param_values) < 2:
        continue  # Нужно хотя бы 2 значения параметра для сравнения
    
    plt.figure(figsize=(16, 12))
    
    # Создаем subplot для каждой стратегии
    strategies = type_data.strategy.unique()
    n_strategies = len(strategies)
    
    n_cols = 2
    n_rows = (n_strategies + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), sharey=True)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, strat in enumerate(strategies):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        strat_data = type_data[type_data.strategy == strat]
        
        for model in strat_data.model.unique():
            model_data = strat_data[strat_data.model == model]
            
            # Для каждого значения параметра
            for param_value in param_values:
                param_model_data = model_data[model_data.main_param == param_value]
                
                if not param_model_data.empty:
                    # Берем последнюю эпоху (финальная точность)
                    final_acc = param_model_data.sort_values('epoch')['acc'].iloc[-1]
                    
                    # Находим все эксперименты с этим параметром
                    exps = param_model_data.experiment.unique()
                    
                    # Собираем дополнительные параметры
                    extra_info = []
                    for exp in exps:
                        exp_info = experiment_info.get(exp, {})
                        exp_params = exp_info.get('params', {})
                        
                        # Добавляем дополнительные параметры
                        for p_name, p_value in exp_params.items():
                            if p_name != param_name and p_value > 0:
                                extra_info.append(f"{p_name}={p_value}")
                    
                    label = model
                    if extra_info:
                        label += f" ({', '.join(set(extra_info))})"
                    
                    ax.scatter(param_value, final_acc, label=label if param_value == param_values[0] else "", 
                              s=100, alpha=0.7)
        
        ax.set_title(f"Strategy: {strat}", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"{param_name} value", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("Final Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    
    # Удаляем лишние subplots
    for idx in range(len(strategies), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f"Dataset: {dataset_type} - Effect of {param_name} on Final Accuracy", 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = os.path.join(comparison_dir, f"{dataset_type}_parameter_effect.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  График влияния параметра для {dataset_type} сохранен")

# ========== 3. СВОДНЫЕ ГРАФИКИ И АНАЛИЗ ==========
print("\n" + "="*60)
print("Создание сводных графиков и анализа")
print("="*60)

summary_dir = os.path.join(graphs_base_dir, "summary")
os.makedirs(summary_dir, exist_ok=True)

# 3.1 Создаем сводную таблицу с лучшими результатами
summary_data = []

for (exp, model, strat), group in combined_df.groupby(['experiment', 'model', 'strategy']):
    if group.empty:
        continue
    
    # Сортируем по эпохам
    group = group.sort_values('epoch')
    
    # Вычисляем метрики
    max_acc = group['acc'].max()
    final_acc = group['acc'].iloc[-1]
    min_loss = group['loss'].min()
    final_loss = group['loss'].iloc[-1]
    
    # Получаем информацию об эксперименте
    exp_info = experiment_info.get(exp, {})
    dataset_type = exp_info.get('dataset_type', 'unknown')
    params = exp_info.get('params', {})
    
    summary_data.append({
        'experiment': exp,
        'model': model,
        'strategy': strat,
        'dataset_type': dataset_type,
        'imbalance': params.get('im', 0),
        'noise': params.get('noise', 0),
        'artifacts': params.get('art', 0),
        'max_accuracy': max_acc,
        'final_accuracy': final_acc,
        'min_loss': min_loss,
        'final_loss': final_loss,
        'num_epochs': len(group['epoch'].unique())
    })

summary_df = pd.DataFrame(summary_data)

# Сохраняем сводную таблицу
summary_df.to_csv(os.path.join(summary_dir, "experiments_summary.csv"), index=False)
print(f"  Сводная таблица сохранена: {os.path.join(summary_dir, 'experiments_summary.csv')}")

# 3.2 Heatmap: лучшие стратегии для каждой комбинации
plt.figure(figsize=(14, 10))

# Создаем pivot таблицу
pivot_data = summary_df.pivot_table(
    index=['dataset_type', 'model'],
    columns='strategy',
    values='final_accuracy',
    aggfunc='mean'
)

# Визуализируем heatmap
if not pivot_data.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Final Accuracy'})
    plt.title("Average Final Accuracy by Dataset Type and Strategy", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(summary_dir, "accuracy_heatmap.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap точности сохранен")

# 3.3 График лучших стратегий
plt.figure(figsize=(16, 10))

# Группируем по типу датасета и модели
for dataset_type in summary_df.dataset_type.unique():
    type_data = summary_df[summary_df.dataset_type == dataset_type]
    
    for model in type_data.model.unique():
        model_data = type_data[type_data.model == model]
        
        # Находим лучшую стратегию
        best_idx = model_data['final_accuracy'].idxmax()
        best_row = model_data.loc[best_idx]
        
        # Создаем метку с параметрами
        label_parts = [model]
        if dataset_type == 'imbalance' and best_row['imbalance'] > 0:
            label_parts.append(f"im={best_row['imbalance']}")
        if best_row['noise'] > 0:
            label_parts.append(f"noise={best_row['noise']}")
        if best_row['artifacts'] > 0:
            label_parts.append(f"art={best_row['artifacts']}")
        
        label = f"{best_row['strategy']} ({', '.join(label_parts)})"
        
        plt.scatter(dataset_type, best_row['final_accuracy'], 
                   label=label, s=200, alpha=0.7, edgecolors='black')

plt.title("Best Strategies for Each Experiment", fontsize=16, fontweight='bold')
plt.xlabel("Dataset Type", fontsize=12)
plt.ylabel("Best Final Accuracy", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.xticks(rotation=45)
plt.tight_layout()

filename = os.path.join(summary_dir, "best_strategies_overview.png")
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"  График лучших стратегий сохранен")

# 3.4 Сводный график всех экспериментов
plt.figure(figsize=(20, 12))

# Цвета для стратегий
strategy_colors = {
    'baseline': 'blue',
    'stagewise': 'green',
    'static': 'red',
    'online': 'purple'
}

# Маркеры для моделей
model_markers = {
    'resnet18': 'o',
    'efficientnet': 's'
}

# Создаем subplot для каждого типа датасета
dataset_types = summary_df.dataset_type.unique()
n_types = len(dataset_types)

fig, axes = plt.subplots(1, n_types, figsize=(6*n_types, 8), sharey=True)
if n_types == 1:
    axes = [axes]

for idx, dataset_type in enumerate(dataset_types):
    ax = axes[idx]
    type_data = summary_df[summary_df.dataset_type == dataset_type]
    
    for _, row in type_data.iterrows():
        color = strategy_colors.get(row['strategy'], 'gray')
        marker = model_markers.get(row['model'], 'o')
        
        # Создаем метку с параметрами
        params = []
        if row['imbalance'] > 0:
            params.append(f"im={row['imbalance']}")
        if row['noise'] > 0:
            params.append(f"noise={row['noise']}")
        if row['artifacts'] > 0:
            params.append(f"art={row['artifacts']}")
        
        param_str = f" ({', '.join(params)})" if params else ""
        
        ax.scatter(f"{row['strategy']}{param_str}", row['final_accuracy'], 
                  color=color, marker=marker, s=150, alpha=0.7, 
                  label=f"{row['model']}" if idx == 0 else "")
    
    ax.set_title(f"Dataset: {dataset_type}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Strategy and Parameters", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Final Accuracy", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    
    # Добавляем легенду для моделей
    if idx == 0:
        legend_elements = [Line2D([0], [0], marker=marker, color='w', 
                                 markerfacecolor='gray', markersize=10, 
                                 label=model_name)
                          for model_name, marker in model_markers.items()]
        ax.legend(handles=legend_elements, title="Models", loc='upper left')

plt.suptitle("Final Accuracy Across All Experiments", fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

filename = os.path.join(summary_dir, "all_experiments_overview.png")
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Сводный график всех экспериментов сохранен")

# ========== 4. СОЗДАНИЕ ОТЧЕТА ==========
print("\n" + "="*60)
print("Генерация сводного отчета")
print("="*60)

report_file = os.path.join(graphs_base_dir, "experiment_summary_report.txt")
with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("ПОЛНЫЙ ОТЧЕТ ПО ВСЕМ ЭКСПЕРИМЕНТАМ\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Всего экспериментов: {len(experiment_info)}\n")
    f.write(f"Всего строк данных: {len(combined_df)}\n")
    f.write(f"Модели: {', '.join(sorted(combined_df.model.unique()))}\n")
    f.write(f"Стратегии: {', '.join(sorted(combined_df.strategy.unique()))}\n\n")
    
    f.write("СПИСОК ЭКСПЕРИМЕНТОВ:\n")
    f.write("-" * 80 + "\n")
    
    for exp_name, info in sorted(experiment_info.items()):
        dataset_type = info['dataset_type']
        params = info['params']
        num_rows = info['num_rows']
        
        f.write(f"\n{exp_name}:\n")
        f.write(f"  Тип: {dataset_type}\n")
        f.write(f"  Параметры: imbalance={params['im']}, noise={params['noise']}, artifacts={params['art']}\n")
        f.write(f"  Данных: {num_rows} строк\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("ЛУЧШИЕ РЕЗУЛЬТАТЫ ПО КАТЕГОРИЯМ\n")
    f.write("="*80 + "\n\n")
    
    # Лучшие результаты для каждого типа датасета
    for dataset_type in sorted(summary_df.dataset_type.unique()):
        f.write(f"\n{dataset_type.upper()} ДАННЫЕ:\n")
        f.write("-" * 40 + "\n")
        
        type_data = summary_df[summary_df.dataset_type == dataset_type]
        
        # Группируем по параметрам
        if dataset_type == 'imbalance':
            param_col = 'imbalance'
        elif dataset_type == 'noisy':
            param_col = 'noise'
        elif dataset_type == 'artifacts':
            param_col = 'artifacts'
        else:
            param_col = None
        
        if param_col:
            param_values = sorted(type_data[param_col].unique())
            
            for param_value in param_values:
                if param_value == 0:
                    continue
                    
                param_data = type_data[type_data[param_col] == param_value]
                
                f.write(f"\n  {param_col}={param_value}:\n")
                
                for model in sorted(param_data.model.unique()):
                    model_data = param_data[param_data.model == model]
                    best_idx = model_data['final_accuracy'].idxmax()
                    best_row = model_data.loc[best_idx]
                    
                    # Собираем дополнительные параметры
                    extra_params = []
                    if best_row['imbalance'] > 0 and param_col != 'imbalance':
                        extra_params.append(f"im={best_row['imbalance']}")
                    if best_row['noise'] > 0 and param_col != 'noise':
                        extra_params.append(f"noise={best_row['noise']}")
                    if best_row['artifacts'] > 0 and param_col != 'artifacts':
                        extra_params.append(f"art={best_row['artifacts']}")
                    
                    extra_str = f" (+{', '.join(extra_params)})" if extra_params else ""
                    
                    f.write(f"    {model}: {best_row['strategy']} = {best_row['final_accuracy']:.4f}{extra_str}\n")
        else:
            # Для original данных
            for model in sorted(type_data.model.unique()):
                model_data = type_data[type_data.model == model]
                best_idx = model_data['final_accuracy'].idxmax()
                best_row = model_data.loc[best_idx]
                
                f.write(f"  {model}: {best_row['strategy']} = {best_row['final_accuracy']:.4f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("ВЫВОДЫ И РЕКОМЕНДАЦИИ\n")
    f.write("="*80 + "\n\n")
    
    # Простые выводы
    f.write("1. Общая статистика:\n")
    f.write(f"   - Средняя финальная точность: {summary_df['final_accuracy'].mean():.4f}\n")
    f.write(f"   - Максимальная точность: {summary_df['final_accuracy'].max():.4f}\n")
    f.write(f"   - Минимальная точность: {summary_df['final_accuracy'].min():.4f}\n\n")
    
    f.write("2. Лучшие стратегии в среднем:\n")
    strategy_avg = summary_df.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
    for strat, avg_acc in strategy_avg.items():
        f.write(f"   - {strat}: {avg_acc:.4f}\n")
    
    f.write("\n3. Устойчивость к разным типам сложностей:\n")
    for dataset_type in summary_df.dataset_type.unique():
        if dataset_type == 'original':
            continue
            
        type_data = summary_df[summary_df.dataset_type == dataset_type]
        avg_acc = type_data['final_accuracy'].mean()
        f.write(f"   - {dataset_type}: {avg_acc:.4f}\n")

print(f"\nВсе графики успешно созданы!")
print(f"Основная папка с графиками: {graphs_base_dir}")
print(f"Сводная таблица: {os.path.join(summary_dir, 'experiments_summary.csv')}")
print(f"Текстовый отчет: {report_file}")
print(f"\nСтруктура созданных папок:")
print(f"  {graphs_base_dir}/")
print(f"    ├── [experiment_name]/        # Графики по каждому эксперименту")
print(f"    │   ├── model_accuracy.png")
print(f"    │   ├── by_strategy/")
print(f"    │   └── loss_plots/")
print(f"    ├── comparisons/              # Сравнительные графики")
print(f"    ├── summary/                  # Сводные графики и анализ")
print(f"    │   ├── experiments_summary.csv")
print(f"    │   ├── accuracy_heatmap.png")
print(f"    │   ├── best_strategies_overview.png")
print(f"    │   └── all_experiments_overview.png")
print(f"    └── experiment_summary_report.txt")