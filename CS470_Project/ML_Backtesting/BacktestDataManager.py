import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pyarrow.parquet as pq

class BacktestDataManager:
    """
    Manages data retrieval and caching for backtesting operations.
    Implements an efficient caching system and provides validated market data.
    """
    
    def __init__(self, db_path: str, cache_dir: str):
        """
        Initialize the BacktestDataManager with database and cache paths.
        
        Args:
            db_path (str): Path to SQLite database
            cache_dir (str): Path to cache directory
        """
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, pd.DataFrame] = {}
        self.invalid_symbols: Dict[str, datetime] = {}
        
        # Ensure directories exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(cache_dir, 'metadata')).mkdir(exist_ok=True)
        Path(os.path.join(cache_dir, 'daily_data')).mkdir(exist_ok=True)
        Path(os.path.join(cache_dir, 'temp', 'downloads')).mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize database
        self._initialize_database()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> dict:
        """Load configuration settings from YAML file."""
        default_config = {
            'cache': {
                'max_memory_cache_size': 1000,
                'cache_expiry_days': 1,
                'update_frequency': '1d',
                'compression_type': 'parquet'
            },
            'download': {
                'max_retries': 3,
                'retry_delay': 5,
                'batch_size': 100,
                'timeout': 30
            },
            'validation': {
                'min_data_points': 50,
                'max_missing_pct': 0.1,
                'price_threshold': 0.01
            }
        }
        
        config_path = os.path.join(self.cache_dir, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return default_config

    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols_meta (
                    symbol TEXT PRIMARY KEY,
                    first_available_date DATE,
                    last_available_date DATE,
                    last_updated TIMESTAMP,
                    is_valid BOOLEAN
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)

    def get_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Primary method to retrieve data for backtesting.
        
        Args:
            symbols (List[str]): List of stock symbols
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        # Validate symbols first
        valid_symbols = self._validate_symbols(symbols, start_date, end_date)
        
        # Initialize result dictionary
        result = {}
        
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=self.config['download']['batch_size']) as executor:
            future_to_symbol = {
                executor.submit(self._get_symbol_data, symbol, start_date, end_date): symbol
                for symbol in valid_symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None:
                        result[symbol] = data
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
                    
        return result

    def _get_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get data for a single symbol with caching."""
        # Check memory cache first
        if symbol in self.memory_cache:
            df = self.memory_cache[symbol]
            if self._is_data_valid_for_range(df, start_date, end_date):
                return df
        
        # Check disk cache
        cache_path = self._get_cache_path(symbol)
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            if self._is_data_valid_for_range(df, start_date, end_date):
                # Update memory cache
                self.memory_cache[symbol] = df
                return df
        
        # Download if not in cache
        try:
            df = self._download_data(symbol, start_date, end_date)
            if df is not None:
                # Save to cache
                self._save_to_cache(symbol, df)
                return df
        except Exception as e:
            self.logger.error(f"Failed to download {symbol}: {str(e)}")
            return None

    def _validate_symbols(self, symbols: List[str], start_date: datetime, end_date: datetime) -> List[str]:
        """
        Validate symbols and filter out invalid ones.
        
        Args:
            symbols (List[str]): List of symbols to validate
            start_date (datetime): Start date for validation
            end_date (datetime): End date for validation
            
        Returns:
            List[str]: List of valid symbols
        """
        valid_symbols = []
        
        for symbol in symbols:
            # Check invalid symbols cache
            if symbol in self.invalid_symbols:
                if datetime.now() - self.invalid_symbols[symbol] < timedelta(days=self.config['cache']['cache_expiry_days']):
                    continue
            
            # Check database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT is_valid, first_available_date, last_available_date
                    FROM symbols_meta
                    WHERE symbol = ?
                """, (symbol,))
                
                result = cursor.fetchone()
                
                if result and result[0]:  # Symbol is valid
                    first_date = datetime.strptime(result[1], '%Y-%m-%d').date()
                    last_date = datetime.strptime(result[2], '%Y-%m-%d').date()
                    
                    if (first_date <= start_date.date() and 
                        last_date >= end_date.date()):
                        valid_symbols.append(symbol)
                else:
                    # New symbol or needs revalidation
                    valid_symbols.append(symbol)
        
        return valid_symbols

    def _download_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Download data from yfinance with retry logic.
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            Optional[pd.DataFrame]: Downloaded data or None if failed
        """
        for attempt in range(self.config['download']['max_retries']):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1d'
                )
                
                if len(df) >= self.config['validation']['min_data_points']:
                    df = self._validate_and_clean_data(df)
                    if df is not None:
                        self._update_symbol_metadata(symbol, df)
                        return df
                
                return None
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < self.config['download']['max_retries'] - 1:
                    time.sleep(self.config['download']['retry_delay'])
        
        self.invalid_symbols[symbol] = datetime.now()
        return None

    def _validate_and_clean_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Validate and clean downloaded data.
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            Optional[pd.DataFrame]: Cleaned data or None if invalid
        """
        if df.empty:
            return None
            
        # Check for missing values
        missing_pct = df.isnull().mean().max()
        if missing_pct > self.config['validation']['max_missing_pct']:
            return None
            
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Check for price validity
        if (df['Close'] < self.config['validation']['price_threshold']).any():
            return None
            
        return df

    def _update_symbol_metadata(self, symbol: str, df: pd.DataFrame) -> None:
        """Update symbol metadata in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO symbols_meta
                (symbol, first_available_date, last_available_date, last_updated, is_valid)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol,
                df.index[0].date().isoformat(),
                df.index[-1].date().isoformat(),
                datetime.now().isoformat(),
                True
            ))

    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol)
        df.to_parquet(cache_path)
        
        # Update memory cache
        if len(self.memory_cache) >= self.config['cache']['max_memory_cache_size']:
            # Remove oldest item
            oldest_symbol = next(iter(self.memory_cache))
            del self.memory_cache[oldest_symbol]
        
        self.memory_cache[symbol] = df

    def _get_cache_path(self, symbol: str) -> str:
        """Get cache file path for symbol."""
        return os.path.join(
            self.cache_dir,
            'daily_data',
            f"{symbol}.parquet"
        )

    def _is_data_valid_for_range(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> bool:
        """Check if cached data covers the requested date range."""
        if df.empty:
            return False
            
        return (df.index[0].date() <= start_date.date() and 
                df.index[-1].date() >= end_date.date())

    def update_cache(self, symbols: List[str]) -> None:
        """
        Update cache for specified symbols.
        
        Args:
            symbols (List[str]): List of symbols to update
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['cache']['cache_expiry_days'])
        
        with ThreadPoolExecutor(max_workers=self.config['download']['batch_size']) as executor:
            future_to_symbol = {
                executor.submit(self._download_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Failed to update cache for {symbol}: {str(e)}")