"""
Configuration for volatility forecasting project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data settings
DEFAULT_TICKER = "SPY"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-10-28"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"

# Important lags based on feature analysis
IMPORTANT_LAGS = [1, 2, 6, 11, 16]

# Trading days
TRADING_DAYS = 252

# Feature engineering
VOLATILITY_WINDOWS = [5, 10, 20, 60]
EWMA_HALFLIVES = [5, 10, 20]
GK_WINDOW = 5  # Garman-Klass rolling window

# GARCH settings
GARCH_TRAIN_WINDOW = 1260  # 5 years
GARCH_REFIT_FREQ = 20  # Monthly (in trading days)
GARCH_MODELS = ['garch', 'egarch']  # Both GARCH and EGARCH
GARCH_DISTRIBUTION = 'studentst'  # Use Student-t by default

# LSTM settings
LSTM_SEQ_LEN = 30  # 30-day lookback
LSTM_TRAIN_WINDOW = 756  # 3 years
LSTM_REFIT_FREQ = 20  # Monthly
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_WEIGHT_DECAY = 1e-5
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 50
LSTM_PATIENCE = 15
LSTM_VAL_SPLIT = 0.2

# Transformer settings
TRANSFORMER_SEQ_LEN = 30
TRANSFORMER_HIDDEN = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1

# Backtesting
VOL_TARGET = 0.10  # 10% annualized
MAX_LEVERAGE = 2.0
TRANSACTION_COST_BPS = 5
SLIPPAGE_BPS = 1

# Evaluation
QLIKE_EPSILON = 1e-8
CORRELATION_THRESHOLD = 0.95  # For feature selection

# Plotting
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_FIGSIZE = (15, 6)

# Random seeds for reproducibility
RANDOM_SEED = 42
import numpy as np
import torch

def set_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
