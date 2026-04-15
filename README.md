# SolarCityEV

> **Code and data for "Coupling rooftop photovoltaics with electric vehicle charging accelerates decarbonisation"**

## Overview

This repository contains the implementation and data for our research.

### What This Repository Contains

- **🔋 EV Charging Demand Prediction Models**: LSTM, ARIMA, and baseline models (FCNN, FGN) for forecasting electric vehicle charging patterns
- **🌍 Multi-City Dataset**: Charging behavior data from 40+ cities worldwide
- **📊 Meta-Learning Framework**: Implementation of meta-learning approaches for cross-city charging demand prediction
- **⚡ Baseline Comparisons**: Multiple baseline models for comprehensive performance evaluation

### Key Features

- **Multi-Model Approach**: Implements LSTM, ARIMA, FCNN, FGN, and meta-learning variants
- **Global Dataset**: Covers diverse charging patterns across different cities and climates
- **Meta-Learning**: Cross-city knowledge transfer for improved prediction accuracy
- **Comprehensive Evaluation**: Multiple baseline models for robust performance comparison

## System Requirements

### Software Dependencies
- Python ≥ 3.8
- See `requirements.txt` for the complete list of Python package dependencies

### Operating Systems
- Windows 10/11
- macOS 10.15 or higher
- Linux (Ubuntu 18.04 or higher)

### Tested Versions
- Python 3.8.10
- PyTorch 1.9.0
- CUDA 11.1 (for GPU acceleration)

### Hardware Requirements
- Minimum: 8GB RAM, 4GB free disk space
- Recommended: 16GB RAM, 8GB free disk space
- GPU: NVIDIA GPU with CUDA support (optional, for faster training)

## Installation Guide

### Prerequisites
1. Install Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. Install Git from [git-scm.com](https://git-scm.com/downloads)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/IntelligentSystemsLab/SolarCityEV.git
   cd SolarCityEV
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   # Option 1: Install from requirements.txt (recommended)
   pip install -r requirements.txt
   
   # Option 2: Install PyTorch separately (if needed)
   pip install torch torchvision torchaudio
   ```

4. Verify installation:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

### Typical Install Time
- On a normal desktop computer (8GB RAM, SSD): 5-10 minutes
- On a slower machine (4GB RAM, HDD): 10-15 minutes

## Demo

### Quick Start

**Option 1: Using the shell script (Recommended)**
```bash
# Make the script executable (first time only, on macOS/Linux)
chmod +x run.sh

# Run with default parameters (London, 300 epochs)
./run.sh

# Run with custom parameters
./run.sh --city London --epochs 100 --lr 0.001

# Run with multiple cities and divide modes
./run.sh --city London Paris NewYork --divide_mode by_month by_day

# See all available options
./run.sh --help
```

**Note**: On Windows, you can use Git Bash or WSL to run the shell script, or use Option 2 (Python directly).

**Option 2: Using Python directly**
```bash
# From project root directory
python code/run.py

# With custom parameters
python code/run.py --city London --epochs 100 --support_epochs 5 --lr 0.005

# With multiple cities and divide modes
python code/run.py --city London Paris --divide_mode by_month by_day --epochs 100

# See all available options
python code/run.py --help
```

### Command Line Arguments

- `--city`: City name(s) in English (default: London). Can specify multiple cities. Available cities include: London, Warsaw, Washington, Copenhagen, SaoPaulo, Melbourne, Toronto, Oslo, Paris, Sydney, Munich, Stockholm, SanFrancisco, Berlin, LosAngeles, Shenzhen, Ottawa, Honolulu, TelAviv, Milan, Johannesburg, NewYork, Vienna, Rome, Zurich, Montreal, Seattle, Helsinki, Miami, Dubai, Dublin, Amsterdam, Athens, Reykjavik, Madrid, Boston, etc.
  - Example: `--city London` or `--city London Paris NewYork`
- `--epochs`: Number of training epochs (default: 300)
- `--support_epochs`: Number of support epochs (default: 5)
- `--custom_epochs`: Number of custom epochs (default: 5)
- `--lr`: Learning rate (default: 0.005)
- `--divide_mode`: Data division mode(s) (default: by_month). Can specify multiple modes: `by_month` and/or `by_day`
  - Example: `--divide_mode by_month` or `--divide_mode by_month by_day`
- `--folder_path`: Data folder path (default: charging_data/by_station)
- `--seed`: Random seed for reproducibility (default: 2023)
- `--batch_size`: Batch size (default: None, uses default batch size)
- `--print_details`: Print detailed training information

**Note**: When multiple cities and/or divide_modes are specified, the script will iterate over all combinations. For example, `--city London Paris --divide_mode by_month by_day` will run 4 experiments (2 cities × 2 modes).

### Expected Output
The demo will:
- Display a formatted header with configuration summary
- Show progress for each experiment (e.g., "Experiment 1/4")
- Load charging data from specific cities (e.g., London)
- Train meta-learning models with LSTM architecture
- Display training progress with tqdm progress bars
- Show evaluation metrics after each experiment
- Display a summary table at the end with all experiment results
- Save results to `results/` directory
- Generate log files with detailed results in `results/log_desktop.txt`

**Example output format:**
```
================================================================================
  Meta-Learning Training for EV Charging Demand Prediction
================================================================================
📅 Start Time: 2025-01-12 18:30:00
📋 Configuration:
   Cities: London (1 city/cities)
   ...
================================================================================
🔬 Experiment 1/1
   City: London (伦敦)
   Divide Mode: by_month
================================================================================
...
✅ Experiment 1/1 completed in 15m 30s
   Metrics (RMSE, MAE, MAPE, MedAE, R2, EVS): [0.123, ...]
================================================================================
  Experiment Summary
================================================================================
...
```

### Expected Run Time
- On a normal desktop computer (8GB RAM, CPU): 15-30 minutes
- With GPU acceleration: 5-10 minutes
- On a slower machine (4GB RAM): 30-60 minutes

## Instructions for Use

### Running on Your Data

1. **Prepare your data**:
   - Place your data files in the `data/` directory
   - Follow the same format as the existing data files
   - Ensure data is properly formatted (see data format section below)

2. **Configure parameters**:
   - Use command-line arguments (recommended): `--city`, `--epochs`, `--lr`, etc.
   - Or modify hyperparameters in `code/model/train.py` for advanced customization

3. **Run training**:
   ```bash
   # Using shell script (from project root)
   ./run.sh --city London --epochs 300
   
   # Or using Python directly
   python code/run.py --city London --epochs 300
   ```

4. **Run specific models**:
   ```bash
   # Run ARIMA baseline
   cd code
   python run_arima.py
   
   # Run all baseline models (FCNN, FGN, etc.)
   python run_baselines.py
   
   # Run main meta-learning training with custom parameters
   python run.py --city London --epochs 300 --lr 0.005
   ```

### Data Format
Your data should be in NumPy format (.npy files) with the following structure:
- Training data: `train_data.npy`
- Test data: `test_data.npy`
- Data shape: (samples, features, time_steps)

### Model Configuration
- Edit `code/model/train.py` for LSTM model parameters
- Modify `code/run_arima.py` for ARIMA model settings
- Update `code/run_baselines.py` for baseline model configurations

### Output Files
All output files are saved in the `results/` directory:
- **Model checkpoints**: `results/model/pt_files/` (if model saving is enabled)
- **Prediction results**: `results/data/pre_result_<divide_mode>/<city>_<timestamp>.csv`
- **Baseline results**: `results/baselines/<baseline_name>/<city>_<timestamp>.csv`
- **Training logs**: `results/log_desktop.txt` (main training log)
- **Baseline logs**: `results/log_<baseline>_test.txt` or `results/log_<baseline>.txt`

## Reproduction Instructions

### Reproducing Paper Results

1. **Download the complete dataset**:
   - Ensure all city data is present in `data/charging_data/by_station/`
   - Verify data integrity and format

2. **Run the complete experiment**:
   ```bash
   # From project root directory
   ./run.sh --city London --epochs 300
   
   # Or using Python directly
   python code/run.py --city London --epochs 300
   ```

3. **Generate all baseline comparisons**:
   ```bash
   # From project root directory
   cd code
   python run_baselines.py
   ```

4. **Run ARIMA experiments**:
   ```bash
   # From project root directory
   cd code
   python run_arima.py
   ```

### Expected Results
- All quantitative results from the manuscript should be reproducible
- Model performance metrics will be displayed in the console
- Figures and visualizations will be saved automatically
- Results will match the values reported in the paper (within small numerical precision differences)

### Troubleshooting
- If you encounter memory issues, reduce batch size in model configuration
- For GPU issues, ensure CUDA is properly installed and compatible
- Check data format if loading errors occur

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub.