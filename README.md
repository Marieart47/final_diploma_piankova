## Project Architecture

```
project/
├── __pycache__/                # Python bytecode cache (auto-generated)
├── all_experiments_plots/      # Consolidated experiment visualizations
├── checkpoints/                # Model checkpoint
├── curriculum/                 # Curriculum learning configurations
├── data/                       # Raw and processed data
├── datasets/                   # Dataset loaders and utilities
├── experiment_logs/            # Individual experiment logs
├── experiments/                # Experiment configurations and scripts
├── models/                     # Model architectures
├── plots/                      # General plotting utilities 
├── results/                    # Experimental results and metrics
├── training/                   # Training loops and procedures
├── training_logs/              # Training process logs
├── utils/                      # Utility functions and helpers
└── venv/                       # Python virtual environment
```

## Quick Start

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate
```

### 2. Data Preparation
```bash
# Place your data in the data/ directory
# Use datasets/ for custom dataset implementations
```

### 3. Training
```bash
# Run an experiment
python experiments/your_experiment.py

# Or use training scripts
python training/train.py --config config.yaml
```

### 4. Evaluation & Results
- Checkpoints are saved in `checkpoints/`
- Results are stored in `results/`
- Visualizations in `all_experiments_plots/`

## Directory Details

### **Core Components**
- **`models/`**: Neural network architectures and model definitions
- **`training/`**: Training loops, loss functions, optimizers
- **`datasets/`**: Data loaders, preprocessing, augmentation

### **Experiment Management**
- **`experiments/`**: Individual experiment scripts and configurations
- **`experiment_logs/`**: Per-experiment logging outputs
- **`all_experiments_plots/`**: Comparative analysis across experiments and visualizations

### **Results & Analysis**
- **`results/`**: Quantitative results (metrics, scores)
- **`plots/`**: Various visualization functions

### **Utilities & Configuration**
- **`utils/`**: Helper functions, common utilities
- **`curriculum/`**: Curriculum learning schedules and strategies
- **`data/`**: Raw and intermediate data files

### **Automatic Directories**
- **`__pycache__/`**: Python cache 
- **`venv/`**: Virtual environment 

## 📊 Monitoring & Logging

- **Training Progress**: Check `training_logs/` for epoch-wise logs
- **Experiment Tracking**: `experiment_logs/` contains experiment-specific logs
- **Visualization**: All plots are organized in respective plot directories

## Configuration

1. **Model Configuration**: Modify files in `models/`
2. **Training Parameters**: Adjust settings in `training/` scripts
3. **Experiment Setup**: Create new files in `experiments/`
4. **Curriculum Learning**: Configure schedules in `curriculum/`

## Checkpoint System

- Models are automatically saved to `checkpoints/` during training
- Use checkpoints for:
  - Resuming interrupted training
  - Model evaluation
  - Fine-tuning


### Cleaning Cache
```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
```

## Performance Tracking

- Compare experiments using `all_experiments_plots/`
- Review detailed metrics in `results/`

---

*This project follows a modular structure for machine learning experiments, enabling easy reproducibility and comparison across different configurations.*