# Data parameters
DATA_START_DATE = '2010-01-01'
DATA_END_DATE = '2023-12-31'
UNIVERSE_SIZE = 500  # Number of stocks to consider

# Feature engineering parameters
TECHNICAL_INDICATORS = ['RSI', 'MACD', 'BB']
LOOKBACK_PERIODS = [5, 10, 20, 60]
TARGET_HORIZON = 5  # Prediction horizon in days

# Model parameters
MODEL_TYPE = 'lightgbm'  # or 'xgboost', 'catboost', etc.
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SIZE = 0.2

# Trading parameters
INITIAL_CAPITAL = 1_000_000
MAX_POSITION_SIZE = 0.1
TRANSACTION_COSTS = 0.001