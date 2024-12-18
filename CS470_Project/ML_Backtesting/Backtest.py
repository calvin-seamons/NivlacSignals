import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Union
import yfinance as yf
from pathlib import Path
import pickle

# Import our custom classes (to be implemented later)
from factor_pipeline import FactorPipeline
from ml_model import MLModel
from mean_reversion import MeanReversionAnalyzer
from trading_strategy import TradingStrategy

from config.logging_config import get_logger
from config.settings import (
    PORTFOLIO_STRATEGY_PARAMS,
    OPTIMIZATION_PARAMS,
    DATA_DIR
)

class CustomYahooFeed(bt.feeds.PandasData):
    """
    Custom feed for Backtrader that handles Yahoo Finance data
    """
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
    )
    
    def __init__(self, dataname, name, fromdate, todate, *args, **kwargs):
        super().__init__(dataname=dataname, name=name, fromdate=fromdate, todate=todate, *args, **kwargs)

class Backtest:
    def __init__(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        symbols: List[str],
        initial_cash: float = 1_000_000,
        commission: float = 0.001,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Backtest.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            symbols: List of symbols to trade
            initial_cash: Initial capital
            commission: Commission rate per trade
            cache_dir: Directory for caching data
        """
        self.logger = get_logger(self.__class__.__name__)
        
        # Convert dates if they're strings
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.commission = commission
        
        # Setup cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path(DATA_DIR) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.cerebro: Optional[bt.Cerebro] = None
        self.results: Optional[Dict] = None
        self.factor_pipeline: Optional[FactorPipeline] = None
        
        self.logger.info(f"Initialized backtest with {len(symbols)} symbols")
        
    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol"""
        return self.cache_dir / f"{symbol}_data.pkl"
        
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
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
        try:
            with open(self._get_cache_path(symbol), 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache data for {symbol}: {e}")
            
    def initialize_data(self) -> None:
        """
        Fetch and initialize data for all symbols
        """
        self.logger.info("Initializing data...")
        
        for symbol in self.symbols:
            try:
                # Try loading from cache first
                cached_data = self._load_from_cache(symbol)
                if cached_data is not None:
                    self.data[symbol] = cached_data
                    continue
                    
                # Fetch from yfinance if not in cache
                self.logger.info(f"Fetching data for {symbol}")
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True
                )
                
                if data.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    continue
                    
                # Cache the data
                self._save_to_cache(symbol, data)
                self.data[symbol] = data
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                continue
                
        self.logger.info(f"Successfully initialized data for {len(self.data)} symbols")
        
    def setup_factor_pipeline(self) -> None:
        """
        Setup and initialize the factor pipeline
        """
        try:
            self.logger.info("Setting up factor pipeline...")
            
            # Initialize ML model
            ml_model = MLModel()  # To be implemented
            
            # Initialize mean reversion analyzer
            mean_reversion = MeanReversionAnalyzer(
                lookback_periods=PORTFOLIO_STRATEGY_PARAMS.get('lookback_periods', 20)
            )  # To be implemented
            
            # Create factor pipeline
            self.factor_pipeline = FactorPipeline(
                ml_model=ml_model,
                mean_reversion=mean_reversion,
                data=self.data
            )  # To be implemented
            
            # Generate initial rankings
            self.factor_pipeline.update()
            
            self.logger.info("Factor pipeline setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up factor pipeline: {e}")
            raise
            
    def setup_cerebro(self) -> None:
        """
        Setup and configure Backtrader Cerebro
        """
        try:
            self.logger.info("Setting up Cerebro...")
            
            # Create Cerebro instance
            self.cerebro = bt.Cerebro()
            
            # Add strategy
            self.cerebro.addstrategy(
                TradingStrategy,
                factor_pipeline=self.factor_pipeline
            )
            
            # Set broker parameters
            self.cerebro.broker.setcash(self.initial_cash)
            self.cerebro.broker.setcommission(commission=self.commission)
            
            # Add data feeds
            valid_symbols = self.factor_pipeline.get_tradeable_symbols()
            for symbol in valid_symbols:
                if symbol in self.data:
                    data = self.data[symbol]
                    feed = CustomYahooFeed(
                        dataname=data,
                        name=symbol,
                        fromdate=self.start_date,
                        todate=self.end_date
                    )
                    self.cerebro.adddata(feed)
                    
            self.logger.info("Cerebro setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up Cerebro: {e}")
            raise
            
    def run_backtest(self) -> Dict:
        """
        Run the backtest and return results
        """
        try:
            self.logger.info("Starting backtest...")
            
            if self.cerebro is None:
                raise ValueError("Cerebro not initialized. Run setup_cerebro first.")
                
            # Run the backtest
            results = self.cerebro.run()
            
            # Process results
            self.results = self._process_results(results[0])
            
            self.logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
            
    def _process_results(self, strategy) -> Dict:
        """
        Process backtest results into a structured format
        """
        try:
            # Get portfolio value
            portfolio_value = self.cerebro.broker.getvalue()
            
            # Calculate returns
            returns = (portfolio_value - self.initial_cash) / self.initial_cash
            
            # Get trade history
            trade_history = strategy.get_trade_history()
            
            # Calculate metrics
            metrics = self._calculate_metrics(strategy)
            
            results = {
                'final_value': portfolio_value,
                'returns': returns,
                'trade_history': trade_history,
                'metrics': metrics
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing results: {e}")
            raise
            
    def _calculate_metrics(self, strategy) -> Dict:
        """
        Calculate performance metrics
        """
        try:
            metrics = {
                'total_trades': len(strategy.trades),
                'winning_trades': sum(1 for trade in strategy.trades if trade.pnl > 0),
                'losing_trades': sum(1 for trade in strategy.trades if trade.pnl <= 0),
                'total_pnl': sum(trade.pnl for trade in strategy.trades),
                'max_drawdown': strategy.stats.drawdown.max,
                'sharpe_ratio': strategy.stats.sharpe_ratio,
                'sortino_ratio': strategy.stats.sortino_ratio
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise
            
    def plot_results(self, filename: Optional[str] = None) -> None:
        """
        Plot backtest results
        """
        try:
            if self.cerebro is None:
                raise ValueError("No backtest results to plot")
                
            self.cerebro.plot(style='candlestick', filename=filename)
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")
            raise
            
    def save_results(self, filepath: str) -> None:
        """
        Save backtest results to file
        """
        try:
            if self.results is None:
                raise ValueError("No results to save")
                
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
                
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
            
def run_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 1_000_000,
    commission: float = 0.001
) -> Dict:
    """
    Convenience function to run a complete backtest
    """
    try:
        # Initialize backtest
        bt = Backtest(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_cash=initial_cash,
            commission=commission
        )
        
        # Setup and run
        bt.initialize_data()
        bt.setup_factor_pipeline()
        bt.setup_cerebro()
        results = bt.run_backtest()
        
        return results
        
    except Exception as e:
        logging.error(f"Error running backtest: {e}")
        raise