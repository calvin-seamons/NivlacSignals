# settings.py structure

"""
Configuration settings for ML trading strategy
This file contains all constants, parameters, and configurations
"""

from datetime import datetime
import os
from typing import Dict, List

#############################################
# PATH CONFIGURATIONS
#############################################
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

#############################################
# DATA SETTINGS
#############################################
# Time periods
TRAINING_START_DATE = datetime(2010, 1, 1)
TRAINING_END_DATE = datetime(2020, 12, 31)
VALIDATION_START_DATE = datetime(2021, 1, 1)
VALIDATION_END_DATE = datetime(2021, 12, 31)
BACKTEST_START_DATE = datetime(2022, 1, 1)
BACKTEST_END_DATE = datetime(2023, 12, 31)

# Data parameters
UNIVERSE_SIZE = 500  # Number of stocks to consider
MIN_PRICE = 5.0  # Minimum stock price filter
MIN_MARKET_CAP = 1e9  # Minimum market cap filter
MIN_VOLUME = 100000  # Minimum daily volume filter
LOOKBACK_PERIOD = 252  # Days of historical data for features

#############################################
# FEATURE ENGINEERING SETTINGS
#############################################
# Technical indicators to generate
TECHNICAL_INDICATORS = {
    'RSI': [14, 30],  # [period, oversold_threshold]
    'MACD': [12, 26, 9],  # [fast, slow, signal]
    'BB': [20, 2],  # [period, std_dev]
    'MA': [50, 200]  # Moving average periods
}

# Feature generation parameters
FEATURE_PARAMETERS = {
    'price_features': ['returns', 'volatility', 'momentum'],
    'volume_features': ['volume_ma', 'volume_std'],
    'market_features': ['spy_correlation', 'sector_momentum'],
    'fundamental_features': ['pe_ratio', 'pb_ratio', 'debt_to_equity']
}

#############################################
# ML MODEL SETTINGS
#############################################
# Model hyperparameters
MODEL_PARAMS = {
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'reg:squarederror'
    },
    'lightgbm': {
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 100
    },
    'neural_network': {
        'layers': [64, 32, 16],
        'dropout': 0.2,
        'learning_rate': 0.001
    }
}

# Training parameters
TRAIN_PARAMS = {
    'train_test_split': 0.8,
    'validation_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

#############################################
# STRATEGY SETTINGS
#############################################
# Portfolio constraints
MAX_POSITIONS = 50
MAX_POSITION_SIZE = 0.05  # 5% max in single position
MAX_SECTOR_EXPOSURE = 0.25  # 25% max in single sector
MIN_HOLDING_PERIOD = 5  # trading days
MAX_HOLDING_PERIOD = 60  # trading days

# Risk management
STOP_LOSS = 0.05  # 5% stop loss
TAKE_PROFIT = 0.15  # 15% take profit
MAX_DRAWDOWN_LIMIT = 0.20  # 20% max drawdown

# Trading parameters
INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.001  # 0.1% commission per trade
SLIPPAGE = 0.0005  # 5 bps slippage
REBALANCE_FREQUENCY = 5  # trading days

#############################################
# BACKTESTING SETTINGS
#############################################
# Backtest parameters
BENCHMARK_INDEX = 'SPY'
RISK_FREE_RATE = 0.02  # Annual risk-free rate
POSITION_SIZING_METHOD = 'equal_weight'  # or 'kelly_criterion' or 'ml_confidence'

# Performance metrics to track
PERFORMANCE_METRICS = [
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'win_rate',
    'profit_factor',
    'calmar_ratio'
]

#############################################
# LOGGING AND DEBUG SETTINGS
#############################################
# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'trading.log')

# Debug settings
DEBUG_MODE = True
SAVE_PREDICTIONS = True
SAVE_POSITIONS = True
PLOT_RESULTS = True

#############################################
# API AND DATABASE SETTINGS
#############################################
# Data source configurations
DATA_SOURCE = {
    'primary': 'yahoo_finance',
    'backup': 'alpha_vantage',
    'api_keys': {
        'alpha_vantage': 'your_key_here',
        'polygon': 'your_key_here'
    }
}

# Database settings
DATABASE = {
    'host': 'localhost',
    'port': 5432,
    'name': 'ml_trading',
    'user': 'your_user',
    'password': 'your_password'
}

#############################################
# OPTIMIZATION SETTINGS
#############################################
# Hyperparameter optimization
OPTIMIZATION_PARAMS = {
    'n_trials': 100,
    'timeout': 3600,  # seconds
    'optimization_metric': 'sharpe_ratio'
}

# Walk-forward optimization
WALK_FORWARD_PARAMS = {
    'window_size': 252,  # trading days
    'step_size': 63,    # trading days
    'min_training_size': 756  # trading days
}