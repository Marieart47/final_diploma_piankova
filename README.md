# Курикулум-обучение для задач компьютерного зрения

Исследование влияния стратегий курикулум-обучения (Curriculum Learning, CL) на качество моделей компьютерного зрения при работе с некачественными данными: зашумлёнными метками, дисбалансом классов и визуальными артефактами.

## Задачи и модели

| Задача | Модели | Датасеты |
|--------|--------|----------|
| Классификация | ResNet-18, Swin-T | CIFAR-10, STL-10 |
| Детекция объектов | YOLOv8n | PASCAL VOC |

## Типы деградации данных

| Тип | Описание | Параметр |
|-----|----------|----------|
| `original` | Исходный датасет без изменений | — |
| `noisy` | Случайная замена меток с вероятностью p | `--noise_level` |
| `imbalance` | Прореживание редких классов на долю f | `--imbalance_factor` |
| `artifacts` | Гауссово размытие / пиксельный шум / пикселизация | `--artifact_level`, `--artifact_type` |

## Стратегии обучения

| Стратегия | Описание |
|-----------|----------|
| `baseline` | Полный датасет на каждой эпохе |
| `stagewise` | Последовательное обучение на окнах сложности: лёгкие → средние → сложные |
| `static` | Группы фиксируются после warm-up прохода, затем циклически используются |
| `online` | На каждой эпохе выбирается 30% наилегчайших образцов по текущему лоссу |

## Структура проекта

```
diplom_v2/
├── curriculum/
│   └── strategies.py              # Стратегии курикулум-обучения
├── datasets/
│   ├── curriculum_dataset.py      # Обёртка с отслеживанием per-sample лосса
│   ├── difficult_datasets.py      # Датасеты с деградацией (шум, дисбаланс, артефакты)
│   └── detection_dataset.py       # Аналог для детекции (YOLODataset)
├── models/
│   └── classification.py          # ResNet-18, Swin-T
├── plots/
│   ├── visualize_results.py       # Графики для классификации
│   └── visualize_detection_results.py  # Графики для детекции
├── training/
│   ├── classification_trainer.py  # Тренер классификации
│   └── detection_trainer.py       # Тренер детекции (YOLOv8)
├── utils/
│   └── metrics.py                 # accuracy(), iou()
├── run_classification.py          # Запуск экспериментов по классификации
├── run_detection.py               # Запуск экспериментов по детекции
└── run_all.sh                     # Пакетный запуск всех экспериментов
```

## Быстрый старт

### Классификация

```bash
# CIFAR-10, зашумлённые метки, ResNet-18 и Swin-T, все стратегии
python run_classification.py \
    --dataset cifar10 \
    --dataset_type noisy \
    --noise_level 0.3 \
    --models resnet18,swin_t \
    --strategies baseline,stagewise,static,online \
    --total_steps 3000

# STL-10, дисбаланс классов
python run_classification.py \
    --dataset stl10 \
    --dataset_type imbalance \
    --imbalance_factor 0.6 \
    --models resnet18 \
    --total_steps 3000

# CIFAR-10, blur-артефакты
python run_classification.py \
    --dataset cifar10 \
    --dataset_type artifacts \
    --artifact_type blur \
    --artifact_level 0.4 \
    --models resnet18,swin_t \
    --total_steps 3000
```

### Детекция (PASCAL VOC)

```bash
# Исходный датасет, все стратегии
python run_detection.py \
    --data VOC.yaml \
    --dataset_type original \
    --strategies baseline,stagewise,static,online \
    --total_steps 3000 \
    --batch 8

# Зашумлённые датасеты
python run_detection.py \
    --data VOC.yaml \
    --dataset_type noisy \
    --noise_level 0.3 \
    --strategies baseline,stagewise,static,online \
    --total_steps 3000
```

### Визуализация результатов

```bash
# Графики классификации
python plots/visualize_results.py

# Графики детекции
python plots/visualize_detection_results.py
```

## Аргументы run_classification.py

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--dataset` | `cifar10` | `cifar10`, `stl10`, `cifar100` - датасет|
| `--dataset_type` | `original` | `original`, `noisy`, `imbalance`, `artifacts` - тип качества датасета |
| `--models` | `resnet18` | Модель |
| `--strategies` | `baseline,stagewise,static,online` | Стратегия обучения |
| `--budget_mode` | `steps` | `steps` — равное число шагов, `epochs` — равное число эпох |
| `--total_steps` | `5000` | Число градиентных шагов (при `budget_mode=steps`) |
| `--noise_level` | `0.3` | Вероятность замены метки |
| `--imbalance_factor` | `0.5` | Доля удаляемых образцов редких классов |
| `--artifact_level` | `0.5` | Интенсивность артефакта [0, 1] |
| `--artifact_type` | `mixed` | `blur`, `noise`, `low_res`, `mixed` - тип артефакта |
| `--patience` | `15` | Ранняя остановка (число эпох без улучшения) |

## Аргументы run_detection.py

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--data` | `VOC.yaml` | YAML-конфиг датасета для ultralytics |
| `--dataset_type` | `original` | `original`, `noisy`, `imbalance`, `artifacts` |
| `--model` | `yolov8n` | `yolov8n`, `yolov8s`, `yolov8m` |
| `--strategies` | `baseline` | Через запятую |
| `--total_steps` | `1000` | Число градиентных шагов |
| `--batch` | `16` | Размер батча |
| `--noise_level` | `0.3` | Вероятность замены метки класса объекта |
| `--imbalance_factor` | `0.3` | Доля удаляемых изображений редких классов |
| `--artifact_level` | `0.4` | Интенсивность артефакта |
| `--artifact_type` | `blur` | `blur`, `noise`, `low_res`, `mixed` |

## Результаты

Результаты сохраняются в `results/<название_эксперимента>/`:

- `classification_results.csv` — метрики по эпохам (train/val loss, accuracy)
- `test_results.csv` — итоговая точность на тесте по каждой стратегии
- `detection_results.csv` — mAP50, mAP50-95 по стратегиям (детекция)
- `experiment_config.json` — конфигурация запуска

Графики сохраняются в `plots/output/`.

## Зависимости

```bash
pip install torch torchvision ultralytics segmentation-models-pytorch \
            albumentations pandas matplotlib numpy pillow opencv-python
```
