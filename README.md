# SolarCityEV

> **Code and data for "Dramatic carbon reduction from rooftop photovoltaic sharing of electric vehicles by 2050"**

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
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.19.0
- Pandas ≥ 1.3.0
- Scikit-learn ≥ 0.24.0
- Matplotlib ≥ 3.3.0
- Seaborn ≥ 0.11.0

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
1. Navigate to the code directory:
   ```bash
   cd code
   ```

2. Run the demo script:
   ```bash
   python run.py
   ```

### Expected Output
The demo will:
- Load charging data from specific cities (e.g., London)
- Train meta-learning models with LSTM architecture
- Run baseline comparisons (FCNN, FGN, ARIMA)
- Display training progress and evaluation metrics
- Save model checkpoints to `model/pt_files/`
- Generate log files with detailed results

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
   - Edit `code/run.py` to adjust model parameters
   - Modify hyperparameters in `code/model/train.py`
   - Update data paths if necessary

3. **Run training**:
   ```bash
   cd code
   python run.py
   ```

4. **Run specific models**:
   ```bash
   # Run ARIMA baseline
   python run_arima.py
   
   # Run all baseline models (FCNN, FGN, etc.)
   python run_baselines.py
   
   # Run main meta-learning training
   python run.py
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
- Model checkpoints: `code/model/pt_files/`
- Training logs: Console output and log files
- Predictions: Saved as NumPy arrays in the output directory

## Reproduction Instructions

### Reproducing Paper Results

1. **Download the complete dataset**:
   - Ensure all city data is present in `data/by_station/`
   - Verify data integrity and format

2. **Run the complete experiment**:
   ```bash
   cd code
   python run.py
   ```

3. **Generate all baseline comparisons**:
   ```bash
   python run_baselines.py
   ```

4. **Run ARIMA experiments**:
   ```bash
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