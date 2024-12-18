import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.logging_config import get_logger
from data.feature_engineering import FeatureEngineering

from exceptions import DataPipelineError

class DataPipeline:
    """
    Main data pipeline class for ML trading strategy.
    Handles data fetching, preprocessing, and management.
    """
    def __init__(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        universe_size: int = 500,
        cache_dir: Optional[str] = None,
        min_volume: float = 1e6,
        price_col: str = 'Close'
    ):
        """
        Initialize DataPipeline.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            universe_size: Number of stocks to include in universe
            cache_dir: Directory for caching data
            min_volume: Minimum average daily volume for stock inclusion
            price_col: Column to use for price data
        """
        # Setup logger
        self.logger = get_logger(self.__class__.__name__)

        self.feature_engineer = FeatureEngineering(price_col=price_col)

        # Convert dates if they're strings
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        self.universe_size = universe_size
        self.min_volume = min_volume
        self.price_col = price_col
        
        # Initialize data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.universe: List[str] = []
        self.market_data: Optional[pd.DataFrame] = None
        
        # Setup cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol"""
        return self.cache_dir / f"{symbol}_data.pkl"

    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        if not self.cache_dir:
            return None
        
        cache_path = self._get_cache_path(symbol)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {symbol}: {e}")
        return None

    def _save_to_cache(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data to cache"""
        if not self.cache_dir:
            return
        
        try:
            with open(self._get_cache_path(symbol), 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache data for {symbol}: {e}")

    def fetch_data(self, symbols: List[str], use_cache: bool = True) -> None:
        """
        Fetch data for given symbols using parallel processing.
        
        Args:
            symbols: List of stock symbols to fetch
            use_cache: Whether to use cached data
        """
        if not symbols:
            raise DataPipelineError("No symbols provided for fetching data")
            
        self.logger.info(f"Fetching data for {len(symbols)} symbols...")
        def fetch_single_symbol(symbol: str) -> tuple[str, Optional[pd.DataFrame]]:
            """Helper function to fetch data for a single symbol"""
            try:
                # Try loading from cache first
                if use_cache:
                    cached_data = self._load_from_cache(symbol)
                    if cached_data is not None:
                        return symbol, cached_data

                # Fetch from yfinance if not in cache
                stock = yf.Ticker(symbol)
                data = stock.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True
                )
                
                if data.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    return symbol, None
                
                # Cache the data
                if use_cache:
                    self._save_to_cache(symbol, data)
                
                return symbol, data
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, None

        # Fetch data in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(fetch_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    self.raw_data[symbol] = data

        self.logger.info(f"Successfully fetched data for {len(self.raw_data)} symbols")

    def process_data(self) -> None:
        """
        Process raw data into format suitable for ML.
        """
        self.logger.info(f"Starting data processing with {len(self.raw_data)} symbols")
        if not self.raw_data:
            raise DataPipelineError("No raw data available for processing")

        # Process each symbol
        for symbol, data in self.raw_data.items():
            try:
                # Handle missing data
                df = self._handle_missing_data(data.copy())
                
                # Use feature engineering class for processing
                df = self.feature_engineer.calculate_basic_features(df)
                
                # Store processed data
                self.processed_data[symbol] = df
                
            except Exception as e:
                self.logger.error(f"Error processing data for {symbol}: {e}")
                continue

        self.logger.info(f"Processed data for {len(self.processed_data)} symbols")

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in DataFrame
        
        Args:
            df: Input DataFrame
        
        Returns:
            Processed DataFrame with handled missing data
        """
        # Forward fill missing values
        df = df.ffill()
        
        # If still has missing values after forward fill, backward fill
        df = df.bfill()
        
        return df

    def create_universe(self) -> List[str]:
        """
        Create universe of tradeable stocks based on liquidity and data availability.
        
        Returns:
            List of symbols in the universe
        """
        if not self.processed_data:
            raise DataPipelineError("No processed data available for universe creation")

        valid_symbols = []
        
        for symbol, data in self.processed_data.items():
            # Check for sufficient data
            if len(data) < 252:  # At least one year of data
                continue
                
            # Check average daily volume
            avg_volume = data['Volume'].mean()
            if avg_volume < self.min_volume:
                continue
                
            valid_symbols.append(symbol)

        # Sort by average volume and take top N
        volume_rank = {
            symbol: self.processed_data[symbol]['Volume'].mean()
            for symbol in valid_symbols
        }
        
        sorted_symbols = sorted(
            volume_rank.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        self.universe = [
            symbol for symbol, _ in sorted_symbols[:self.universe_size]
        ]
        
        self.logger.info(f"Created universe with {len(self.universe)} symbols")
        return self.universe

    def get_training_data(self, lookback_window: int = 20, 
                        forecast_horizon: int = 5,
                        train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        self.logger.info("Starting get_training_data...")
        
        if not self.universe:
            raise DataPipelineError("Universe not created. Run create_universe first.")

        self.logger.info("Preparing features and targets...")
        features_df, targets_series = self.feature_engineer.prepare_features_for_training(
            {symbol: self.processed_data[symbol] for symbol in self.universe},
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon
        )
        
        self.logger.info(f"Features shape: {features_df.shape}")
        self.logger.info(f"Targets shape: {len(targets_series)}")

        # Ensure features and targets have same index
        self.logger.info("Checking index alignment...")
        if not features_df.index.equals(targets_series.index):
            raise DataPipelineError("Features and targets indices do not match")

        # Sort the MultiIndex - this is crucial for slicing
        self.logger.info("Sorting indices...")
        features_df = features_df.sort_index()
        targets_series = targets_series.sort_index()

        # Get unique dates and find split point
        self.logger.info("Calculating split point...")
        dates = features_df.index.get_level_values('date').unique()
        self.logger.info(f"Total unique dates: {len(dates)}")
        split_idx = int(len(dates) * train_ratio)
        split_date = dates[split_idx]
        
        self.logger.info(f"Split date calculated: {split_date}")
        
        # Split the data using the date level
        self.logger.info("Splitting data...")
        X_train = features_df.loc[features_df.index.get_level_values('date') <= split_date].values
        self.logger.info("X_train split complete")
        X_test = features_df.loc[features_df.index.get_level_values('date') > split_date].values
        self.logger.info("X_test split complete")
        y_train = targets_series.loc[targets_series.index.get_level_values('date') <= split_date].values
        self.logger.info("y_train split complete")
        y_test = targets_series.loc[targets_series.index.get_level_values('date') > split_date].values
        self.logger.info("y_test split complete")

        # Log split information
        self.logger.info(f"Training data from {dates[0]} to {split_date}")
        self.logger.info(f"Testing data from {split_date} to {dates[-1]}")
        self.logger.info(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def get_latest_data(self, lookback_window: int = 20) -> pd.DataFrame:
        """
        Get latest data for making predictions.
        
        Args:
            lookback_window: Number of past days to include
            
        Returns:
            DataFrame with latest data for each symbol
        """
        if not self.universe:
            raise DataPipelineError("Universe not created. Run create_universe first.")

        latest_data = {}
        
        for symbol in self.universe:
            data = self.processed_data[symbol]
            latest_data[symbol] = data.iloc[-lookback_window:]

        return pd.concat(latest_data, axis=1)

    def check_data_status(self) -> dict:
        """
        Check the status of data at each stage of the pipeline.
        
        Returns:
            Dictionary with data status information
        """
        status = {
            'raw_data': {
                'count': len(self.raw_data),
                'symbols': list(self.raw_data.keys()),
                'sample_columns': list(self.raw_data[next(iter(self.raw_data))].columns) if self.raw_data else None
            },
            'processed_data': {
                'count': len(self.processed_data),
                'symbols': list(self.processed_data.keys()),
                'sample_columns': list(self.processed_data[next(iter(self.processed_data))].columns) if self.processed_data else None
            },
            'universe': {
                'count': len(self.universe),
                'symbols': self.universe
            }
        }
        return status

    def validate_data_quality(self) -> Dict[str, dict]:
        """
        Run data quality checks on processed data.
        
        Returns:
            Dictionary with quality metrics for each symbol
        """
        quality_metrics = {}
        
        for symbol, data in self.processed_data.items():
            metrics = {
                'missing_values': data.isnull().sum().sum(),
                'data_points': len(data),
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'avg_volume': data['Volume'].mean(),
                'zero_volume_days': (data['Volume'] == 0).sum()
            }
            quality_metrics[symbol] = metrics
            
        return quality_metrics