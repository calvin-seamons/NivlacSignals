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
