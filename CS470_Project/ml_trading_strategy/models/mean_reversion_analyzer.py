import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime

from config.logging_config import get_logger
from data.data_pipeline import DataPipeline
from exceptions import MeanReversionError

@dataclass
class MeanReversionSignals:
    """Container for mean reversion signals and metrics"""
    z_scores: pd.DataFrame
    longs: List[str]
    shorts: List[str]
    signal_strength: pd.Series
    metrics: Dict[str, float]

class MeanReversionAnalyzer:
    def __init__(
        self,
        data_pipeline: DataPipeline,
        lookback_periods: int = 20,
        z_score_threshold: float = 2.0,
        volume_percentile: float = 0.7,
        min_volatility_percentile: float = 0.3,
        max_positions: int = 10
    ):
        """
        Initialize MeanReversionAnalyzer.
        
        Args:
            data_pipeline: Instance of DataPipeline
            lookback_periods: Periods for calculating mean and std dev
            z_score_threshold: Threshold for identifying mean reversion opportunities
            volume_percentile: Minimum percentile for volume filtering
            min_volatility_percentile: Minimum percentile for volatility filtering
            max_positions: Maximum number of positions (long or short)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.data_pipeline = data_pipeline
        self.lookback_periods = lookback_periods
        self.z_score_threshold = z_score_threshold
        self.volume_percentile = volume_percentile
        self.min_volatility_percentile = min_volatility_percentile
        self.max_positions = max_positions
        
        # Storage for calculated signals
        self.current_signals: Optional[MeanReversionSignals] = None
        
    def calculate_z_scores(self, data: pd.DataFrame) -> pd.Series:
        """Calculate z-scores for mean reversion"""
        try:
            rolling_mean = data[self.data_pipeline.price_col].rolling(
                window=self.lookback_periods
            ).mean()
            
            rolling_std = data[self.data_pipeline.price_col].rolling(
                window=self.lookback_periods
            ).std()
            
            z_scores = (data[self.data_pipeline.price_col] - rolling_mean) / rolling_std
            
            return z_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating z-scores: {e}")
            raise MeanReversionError(f"Error calculating z-scores: {e}")
    
    def filter_universe(self) -> List[str]:
        """Filter universe based on volume and volatility criteria"""
        try:
            filtered_symbols = []
            
            for symbol in self.data_pipeline.universe:
                data = self.data_pipeline.processed_data[symbol]
                
                # Calculate metrics
                avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                volatility = data['Close'].pct_change().rolling(window=20).std().iloc[-1]
                
                # Get percentile ranks
                volume_rank = pd.Series([avg_volume] + [
                    self.data_pipeline.processed_data[s]['Volume'].rolling(window=20).mean().iloc[-1]
                    for s in self.data_pipeline.universe if s != symbol
                ]).rank(pct=True).iloc[0]
                
                volatility_rank = pd.Series([volatility] + [
                    self.data_pipeline.processed_data[s]['Close'].pct_change().rolling(window=20).std().iloc[-1]
                    for s in self.data_pipeline.universe if s != symbol
                ]).rank(pct=True).iloc[0]
                
                # Apply filters
                if (volume_rank >= self.volume_percentile and 
                    volatility_rank >= self.min_volatility_percentile):
                    filtered_symbols.append(symbol)
                    
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"Error filtering universe: {e}")
            raise MeanReversionError(f"Error filtering universe: {e}")
    
    def generate_signals(self) -> MeanReversionSignals:
        """Generate mean reversion signals for the current universe"""
        try:
            # Filter universe
            filtered_universe = self.filter_universe()
            self.logger.info(f"Filtered universe contains {len(filtered_universe)} symbols")
            
            # Calculate z-scores for all symbols
            z_scores_dict = {}
            signal_strength = {}
            
            # Get the latest date from the data
            latest_date = None
            for symbol in filtered_universe:
                data = self.data_pipeline.processed_data[symbol]
                if latest_date is None:
                    latest_date = data.index[-1]
                z_scores = self.calculate_z_scores(data)
                
                # Store latest z-score
                z_scores_dict[symbol] = z_scores.iloc[-1]
                
                # Calculate signal strength (inverse of z-score)
                signal_strength[symbol] = abs(1 / z_scores.iloc[-1]) if z_scores.iloc[-1] != 0 else 0
            
            z_scores_series = pd.Series(z_scores_dict)
            
            # Identify long and short positions
            longs = z_scores_series[z_scores_series <= -self.z_score_threshold]
            shorts = z_scores_series[z_scores_series >= self.z_score_threshold]
            
            # Sort by absolute z-score and take top positions
            longs = longs.sort_values().index[:self.max_positions].tolist()
            shorts = shorts.sort_values(ascending=False).index[:self.max_positions].tolist()
            
            # Calculate metrics
            metrics = {
                'avg_long_zscore': z_scores_series[longs].mean() if longs else 0,
                'avg_short_zscore': z_scores_series[shorts].mean() if shorts else 0,
                'long_positions': len(longs),
                'short_positions': len(shorts)
            }
            
            # Create DataFrame with proper index
            z_scores_df = pd.DataFrame(
                {symbol: [value] for symbol, value in z_scores_dict.items()},
                index=[latest_date]
            )
            
            # Create signals object
            self.current_signals = MeanReversionSignals(
                z_scores=z_scores_df,
                longs=longs,
                shorts=shorts,
                signal_strength=pd.Series(signal_strength),
                metrics=metrics
            )
            
            return self.current_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            raise MeanReversionAnalyzerError(f"Error generating signals: {e}")
    
    def get_position_sizes(self) -> Dict[str, float]:
        """Calculate position sizes based on signal strength"""
        if self.current_signals is None:
            raise MeanReversionError("No signals generated yet")
            
        try:
            position_sizes = {}
            
            # Calculate weights for longs
            if self.current_signals.longs:
                long_signals = self.current_signals.signal_strength[self.current_signals.longs]
                long_weights = long_signals / long_signals.sum()
                
                for symbol, weight in long_weights.items():
                    position_sizes[symbol] = weight
            
            # Calculate weights for shorts
            if self.current_signals.shorts:
                short_signals = self.current_signals.signal_strength[self.current_signals.shorts]
                short_weights = short_signals / short_signals.sum()
                
                for symbol, weight in short_weights.items():
                    position_sizes[symbol] = -weight
            
            return position_sizes
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizes: {e}")
            raise MeanReversionError(f"Error calculating position sizes: {e}")
    
    def get_signal_metrics(self) -> pd.DataFrame:
        """Get detailed metrics for current signals"""
        if self.current_signals is None:
            raise MeanReversionError("No signals generated yet")
            
        try:
            metrics_list = []
            
            for symbol in self.current_signals.longs + self.current_signals.shorts:
                data = self.data_pipeline.processed_data[symbol]
                
                metrics = {
                    'symbol': symbol,
                    'position': 'long' if symbol in self.current_signals.longs else 'short',
                    'z_score': self.current_signals.z_scores[symbol].iloc[-1],
                    'signal_strength': self.current_signals.signal_strength[symbol],
                    'volume': data['Volume'].iloc[-1],
                    'price': data[self.data_pipeline.price_col].iloc[-1],
                    'volatility': data[self.data_pipeline.price_col].pct_change().rolling(20).std().iloc[-1]
                }
                
                metrics_list.append(metrics)
            
            return pd.DataFrame(metrics_list)
            
        except Exception as e:
            self.logger.error(f"Error getting signal metrics: {e}")
            raise MeanReversionError(f"Error getting signal metrics: {e}")