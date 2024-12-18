# settings.py

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
# MODEL PIPELINE SETTINGS
#############################################
# Model Selection
DEFAULT_MODEL_TYPE = 'lightgbm'  # Options: 'lightgbm', 'xgboost', 'catboost'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'saved_models')

# Model Parameters by Type
MODEL_PARAMS = {
    'lightgbm': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'colsample_bytree': 0.9,  # Changed from feature_fraction
        'n_estimators': 100,
        'early_stopping_rounds': 50,
        'force_col_wise': True,
        'verbose': -1
    },
    'xgboost': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'early_stopping_rounds': 50
    },
    'catboost': {
        'loss_function': 'RMSE',
        'iterations': 100,
        'learning_rate': 0.05,
        'depth': 6,
        'early_stopping_rounds': 50
    }
}

# Training Parameters
TRAIN_PARAMS = {
    'train_test_split': 0.8,
    'validation_size': 0.2,
    'cv_folds': 5,
    'random_state': 42,
    'shuffle': False  # Important for time series data
}

# Feature Engineering Parameters for Model
FEATURE_PARAMS = {
    'lookback_windows': [5, 10, 20, 60],  # Days for rolling features
    'target_horizon': 5,  # Prediction horizon in days
    'min_training_size': 252  # Minimum days required for training
}

# Model Evaluation Metrics
EVALUATION_METRICS = [
    'mse',
    'rmse',
    'mae',
    'r2',
    'sharpe_ratio',
    'hit_ratio'
]

# Model Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_r2': 0.1,
    'min_sharpe': 0.5,
    'max_drawdown': -0.2,
    'min_hit_ratio': 0.52
}

# Feature Importance Analysis
FEATURE_IMPORTANCE = {
    'use_shap': True,
    'top_n_features': 20,
    'importance_threshold': 0.01
}

#############################################
# MODEL OPTIMIZATION SETTINGS
#############################################
HYPEROPT_PARAMS = {
    'lightgbm': {
        'n_trials': 100,
        'timeout': 3600,
        'param_space': {
            'num_leaves': (15, 50),
            'learning_rate': (0.01, 0.1),
            'feature_fraction': (0.7, 1.0),
            'n_estimators': (50, 200)
        }
    },
    'xgboost': {
        'n_trials': 100,
        'timeout': 3600,
        'param_space': {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.1),
            'n_estimators': (50, 200)
        }
    }
}

#############################################
# MODEL VALIDATION SETTINGS
#############################################
VALIDATION_PARAMS = {
    'rolling_window_size': 252,  # Days for rolling validation
    'min_training_samples': 1000,
    'validation_metric': 'sharpe_ratio',
    'refit_frequency': 20  # Trading days between model refits
}

#############################################
# PREDICTION SETTINGS
#############################################
PREDICTION_PARAMS = {
    'confidence_threshold': 0.6,  # Minimum prediction confidence
    'prediction_frequency': 1,  # Days between predictions
    'ensemble_weighting': 'equal'  # or 'dynamic'
}

#############################################
# LOGGING SETTINGS
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
# PORTFOLIO STRATEGY SETTINGS
#############################################
PORTFOLIO_STRATEGY_PARAMS = {
    # Rebalancing parameters
    'rebalance_frequency': 20,  # Trading days between rebalances
    'min_rebalance_size': 0.02,  # Minimum position change to trigger trade
    
    # Position sizing
    'max_position_size': 0.10,  # Maximum single position size (10%)
    'min_positions': 3,  # Minimum positions per side
    'target_leverage': 1.0,  # Target portfolio leverage
    
    # Risk parameters
    'risk_free_rate': 0.02,  # Annual risk-free rate
    'max_sector_exposure': 0.30,  # Maximum exposure per sector
    'position_limits': {
        'market_cap': {  # Position limits by market cap
            'large': 0.15,
            'mid': 0.10,
            'small': 0.05
        }
    },
    
    # Transaction cost assumptions
    'transaction_costs': {
        'commission': 0.001,  # 10 bps per trade
        'slippage': 0.001,  # 10 bps slippage assumption
    },
    
    # Risk management
    'stop_loss': -0.05,  # 5% stop loss per position
    'profit_target': 0.10,  # 10% profit target
    'portfolio_stop_loss': -0.10,  # 10% portfolio stop loss
    
    # Performance tracking
    'benchmark': 'SPY',  # Benchmark for performance comparison
    'tracking_error_limit': 0.10,  # Maximum tracking error
}

# Portfolio optimization parameters
OPTIMIZATION_PARAMS = {
    'method': 'SLSQP',  # Optimization method
    'risk_aversion': 1.0,  # Risk aversion parameter
    'constraints': {
        'long_weight_min': 0.0,
        'short_weight_max': 0.0,
        'net_exposure_range': (-0.2, 0.2),  # Net exposure limits
        'gross_exposure_max': 2.0,  # Maximum gross exposure
    },
    'optimization_frequency': 'monthly',  # Frequency of full portfolio optimization
    'reoptimization_threshold': 0.05,  # Threshold for triggering reoptimization
}
