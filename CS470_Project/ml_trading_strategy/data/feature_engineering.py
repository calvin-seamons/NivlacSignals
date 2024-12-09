import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

from config.logging_config import get_logger

class FeatureEngineering:
    """
    Feature engineering class for ML trading strategy.
    Handles all feature creation and engineering tasks.
    """
    def __init__(self, price_col: str = 'Close'):
        self.logger = get_logger(self.__class__.__name__)
        self.price_col = price_col

    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic financial features from price data
        """
        try:
            # Create copy to avoid modifying original
            data = df.copy()
            
            # Price-based features
            data['returns'] = data[self.price_col].pct_change()
            data['log_returns'] = np.log(data[self.price_col] / data[self.price_col].shift(1))
            
            # Volatility
            data['volatility'] = data['returns'].rolling(window=20).std()
            
            # Volume features
            data['volume_ma'] = data['Volume'].rolling(window=20).mean()
            data['volume_std'] = data['Volume'].rolling(window=20).std()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating basic features: {e}")
            raise

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        """
        try:
            data = df.copy()
            
            # Moving averages
            data['sma_10'] = data[self.price_col].rolling(window=10).mean()
            data['sma_30'] = data[self.price_col].rolling(window=30).mean()
            data['sma_60'] = data[self.price_col].rolling(window=60).mean()
            
            # RSI
            delta = data[self.price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['rsi'] = 100 - (100 / (1 + (gain / loss)))
            
            # MACD
            exp1 = data[self.price_col].ewm(span=12, adjust=False).mean()
            exp2 = data[self.price_col].ewm(span=26, adjust=False).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            raise

    def create_ml_features(self, 
                          df: pd.DataFrame, 
                          lookback_window: int = 20) -> pd.DataFrame:
        """
        Create features specifically for ML model training
        """
        try:
            data = df.copy()
            
            # Calculate lagged features
            for lag in range(1, lookback_window + 1):
                data[f'return_lag_{lag}'] = data['returns'].shift(lag)
                data[f'volume_lag_{lag}'] = data['Volume'].pct_change().shift(lag)
            
            # Rolling statistics
            data['return_mean_5d'] = data['returns'].rolling(window=5).mean()
            data['return_std_5d'] = data['returns'].rolling(window=5).std()
            data['volume_mean_5d'] = data['Volume'].rolling(window=5).mean()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating ML features: {e}")
            raise

    def prepare_features_for_training(self, 
                                    data: Dict[str, pd.DataFrame],
                                    lookback_window: int = 20,
                                    forecast_horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare final feature matrix and target vector for ML training
        """
        features = []
        targets = []
        
        for symbol, df in data.items():
            try:
                processed_data = self.calculate_basic_features(df)
                processed_data = self.calculate_technical_features(processed_data)
                processed_data = self.create_ml_features(processed_data, lookback_window)
                
                # Create feature vectors
                feature_cols = [col for col in processed_data.columns 
                              if col not in [self.price_col, 'Volume']]
                
                for i in range(lookback_window, len(processed_data) - forecast_horizon):
                    feature_window = processed_data.iloc[i-lookback_window:i][feature_cols]
                    
                    # Future return as target
                    future_return = (
                        processed_data.iloc[i + forecast_horizon][self.price_col] /
                        processed_data.iloc[i][self.price_col] - 1
                    )
                    
                    if not feature_window.isnull().any().any():
                        features.append(feature_window.values.flatten())
                        targets.append(future_return)
                        
            except Exception as e:
                self.logger.error(f"Error preparing features for {symbol}: {e}")
                continue
                
        return np.array(features), np.array(targets)