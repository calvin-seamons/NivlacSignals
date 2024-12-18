import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

from config.logging_config import get_logger
from exceptions import DataPipelineError

class FeatureEngineering:
    """
    Feature engineering class for ML trading strategy.
    Handles all feature creation and engineering tasks.
    """
    def __init__(self, price_col: str = 'Close'):
        self.logger = get_logger(self.__class__.__name__)
        self.price_col = price_col
        self.lookback_window = 20

    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic financial features from price data"""
        try:
            # Create copy to avoid modifying original
            data = df.copy()
            
            # First, check if we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataPipelineError(f"Missing required columns: {missing_columns}")
            
            # Price-based features
            try:
                data['returns'] = data['Close'].pct_change()
            except Exception as e:
                self.logger.error(f"Error calculating returns: {e}")
                data['returns'] = np.nan
                
            try:
                data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            except Exception as e:
                self.logger.error(f"Error calculating log returns: {e}")
                data['log_returns'] = np.nan
            
            # Volatility
            try:
                data['volatility'] = data['returns'].rolling(window=20).std()
            except Exception as e:
                self.logger.error(f"Error calculating volatility: {e}")
                data['volatility'] = np.nan
            
            # Volume features
            try:
                data['volume_ma'] = data['Volume'].rolling(window=20).mean()
                data['volume_std'] = data['Volume'].rolling(window=20).std()
            except Exception as e:
                self.logger.error(f"Error calculating volume features: {e}")
                data['volume_ma'] = np.nan
                data['volume_std'] = np.nan
            
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
                                    forecast_horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    
        self.logger.info(f"Starting feature preparation with {len(data)} symbols")
        features_list = []
        targets_list = []
        dates_list = []
        symbols_list = []
        
        for symbol_idx, (symbol, df) in enumerate(data.items(), 1):
            self.logger.info(f"Processing symbol {symbol} ({symbol_idx}/{len(data)})")
            try:
                # Process features
                processed_data = self.calculate_basic_features(df)
                processed_data = self.calculate_technical_features(processed_data)
                processed_data = self.create_ml_features(processed_data, lookback_window)
                
                # Get feature columns
                feature_cols = [col for col in processed_data.columns 
                            if col not in [self.price_col, 'Volume']]
                
                self.logger.info(f"{symbol}: Processing {len(processed_data) - lookback_window - forecast_horizon} potential samples")
                samples_for_symbol = 0
                
                for i in range(lookback_window, len(processed_data) - forecast_horizon):
                    feature_window = processed_data.iloc[i-lookback_window:i][feature_cols]
                    
                    if feature_window.isnull().any().any():
                        continue
                        
                    future_return = self._calculate_future_return(
                        processed_data[self.price_col],
                        i,
                        forecast_horizon
                    )
                    
                    if future_return is None:
                        continue
                    
                    features_list.append(feature_window.values)
                    targets_list.append(future_return)
                    dates_list.append(processed_data.index[i])
                    symbols_list.append(symbol)
                    samples_for_symbol += 1
                    
                    # Log progress every 1000 samples
                    if samples_for_symbol % 1000 == 0:
                        self.logger.info(f"{symbol}: Processed {samples_for_symbol} samples")
                
                self.logger.info(f"Completed {symbol} with {samples_for_symbol} valid samples")
                
            except Exception as e:
                self.logger.error(f"Error preparing features for {symbol}: {e}")
                continue
        
        self.logger.info(f"Feature generation complete. Total samples: {len(features_list)}")
        
        if not features_list:
            raise ValueError("No valid samples generated")
        
        self.logger.info("Creating features array...")
        features_array = np.stack(features_list)
        
        self.logger.info("Creating features DataFrame...")
        features_df = pd.DataFrame(
            features_array.reshape(len(features_list), -1),
            index=pd.MultiIndex.from_arrays([dates_list, symbols_list], 
                                        names=['date', 'symbol']),
            columns=[f"{col}_{t}" for col in feature_cols 
                    for t in range(lookback_window)]
        )
        
        self.logger.info("Creating targets series...")
        targets_series = pd.Series(
            targets_list,
            index=features_df.index,
            name='future_return'
        )
        
        self.logger.info("Applying normalization...")
        features_df = self._normalize_features(features_df)
        
        self.logger.info("Feature preparation complete")
        return features_df, targets_series

    def _calculate_future_return(self, 
                            price_series: pd.Series,
                            current_idx: int,
                            horizon: int) -> Optional[float]:
        """
        Safely calculate future return with validation.
        """
        if current_idx + horizon >= len(price_series):
            return None
            
        current_price = price_series.iloc[current_idx]
        future_price = price_series.iloc[current_idx + horizon]
        
        if pd.isna(current_price) or pd.isna(future_price) or current_price <= 0:
            return None
            
        return (future_price / current_price) - 1

    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional normalization using efficient NumPy operations.
        """
        self.logger.info(f"Starting normalization for DataFrame of shape {features_df.shape}")
        
        try:
            # Convert to numpy array for faster operations
            values = features_df.values
            dates = features_df.index.get_level_values('date')
            unique_dates = dates.unique()
            
            self.logger.info(f"Processing {len(unique_dates)} unique dates")
            
            # Pre-allocate normalized array
            normalized_values = np.empty_like(values)
            
            # Process each date
            for i, date in enumerate(unique_dates):
                if i % 100 == 0:  # Log progress
                    self.logger.info(f"Processed {i}/{len(unique_dates)} dates")
                    
                # Get mask for current date
                mask = dates == date
                date_data = values[mask]
                
                if len(date_data) > 1:
                    # Calculate statistics
                    means = np.nanmean(date_data, axis=0, keepdims=True)
                    stds = np.nanstd(date_data, axis=0, keepdims=True)
                    
                    # Replace zero stds with 1
                    stds[stds == 0] = 1
                    
                    # Normalize
                    normalized_values[mask] = (date_data - means) / (stds + 1e-8)
                else:
                    normalized_values[mask] = date_data
            
            # Convert back to DataFrame
            normalized_df = pd.DataFrame(
                normalized_values,
                index=features_df.index,
                columns=features_df.columns
            )
            
            self.logger.info("Normalization complete")
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"Error during normalization: {e}")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.warning("Returning unnormalized data due to error")
            return features_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order, including lookback window expansions."""
        # Price data columns (sorted alphabetically to match DataFrame columns)
        base_features = [
            'Dividends', 'High', 'Low', 'Open', 'Stock_Splits',  # Note the underscore
            # Basic features
            'log_returns', 'macd', 'macd_signal',
            'returns', 'rsi',
            'sma_10', 'sma_30', 'sma_60',
            'volatility', 'volume_ma', 'volume_std'
        ]
        
        # Add lag features in order
        for i in range(1, self.lookback_window + 1):
            base_features.extend([
                f'return_lag_{i}',
                f'volume_lag_{i}'
            ])
        
        # Add rolling statistics
        base_features.extend([
            'return_mean_5d', 'return_std_5d', 'volume_mean_5d'
        ])
        
        # Generate final feature names with time steps
        feature_names = []
        for feature in base_features:
            # Replace any spaces with underscores
            feature = feature.replace(' ', '_')
            for t in range(self.lookback_window):
                feature_names.append(f"{feature}_{t}")
        
        return feature_names