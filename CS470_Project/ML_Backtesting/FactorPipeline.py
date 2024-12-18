import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime

import test

@dataclass
class SignalData:
    """Container for signal data"""
    ml_signals: pd.Series
    mean_rev_signals: pd.Series
    combined_signals: pd.Series
    raw_ml_predictions: pd.Series
    raw_mean_rev_scores: pd.Series
    timestamp: datetime

@dataclass
class RankingData:
    """Container for ranking data"""
    rankings: pd.Series
    percentiles: pd.Series
    filtered_scores: pd.Series
    timestamp: datetime

@dataclass
class PositionData:
    """Container for position data"""
    longs: List[str]
    shorts: List[str]
    long_sizes: Dict[str, float]
    short_sizes: Dict[str, float]
    timestamp: datetime

class FactorPipeline:
    def __init__(
        self,
        ml_model,
        mean_reversion,
        data: Dict[str, pd.DataFrame],
        config: dict
    ):
        """
        Initialize FactorPipeline.
        
        Args:
            ml_model: ML model instance
            mean_reversion: Mean reversion analyzer instance
            data: Dictionary of price data frames
            config: Configuration dictionary from YAML
        """
        self.logger = logging.getLogger(__name__)
        
        # Store components
        self.ml_model = ml_model
        self.mean_reversion = mean_reversion
        self.data = data
        self.config = config
        
        # Extract configuration parameters
        self.signal_config = config['factor_pipeline']
        self.ml_weight = self.signal_config['model_weight']
        self.mean_rev_weight = self.signal_config['mean_reversion_weight']
        self.min_score_threshold = self.signal_config['min_score_threshold']
        
        # Position limits from portfolio strategy config
        self.position_config = config['portfolio_strategy']
        self.max_position_size = self.position_config['max_position_size']
        self.min_positions = self.position_config['min_positions']
        
        # Initialize storage
        self.current_signals: Optional[SignalData] = None
        self.current_rankings: Optional[RankingData] = None
        self.current_positions: Optional[PositionData] = None
        
        # Historical storage
        self.signal_history: List[SignalData] = []
        self.ranking_history: List[RankingData] = []
        self.position_history: List[PositionData] = []
        
    def generate_signals(self, latest_data: Optional[Dict[str, pd.DataFrame]] = None) -> SignalData:
        """
        Generate signals from both models
        """
        try:
            # Update data if provided
            if latest_data is not None:
                self.data = latest_data
            
            # Get ML model predictions
            features = self._prepare_features()
            ml_predictions = self.ml_model.predict(features)
            ml_signals = pd.Series(ml_predictions, index=features.index)
            
            # Get mean reversion signals
            mean_rev_signals = self.mean_reversion.generate_signals(self.data)
            mean_rev_scores = mean_rev_signals.signal_strength
            
            # Normalize signals
            norm_ml = self._normalize_signals(ml_signals)
            norm_mean_rev = self._normalize_signals(mean_rev_scores)
            
            # Combine signals
            combined = self._combine_signals(norm_ml, norm_mean_rev)
            
            # Create signal data container
            signal_data = SignalData(
                ml_signals=norm_ml,
                mean_rev_signals=norm_mean_rev,
                combined_signals=combined,
                raw_ml_predictions=ml_signals,
                raw_mean_rev_scores=mean_rev_scores,
                timestamp=datetime.now()
            )
            
            # Store signals
            self.current_signals = signal_data
            self.signal_history.append(signal_data)
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            raise
            
    def _normalize_signals(self, signals: pd.Series) -> pd.Series:
        """
        Normalize signals using z-score method
        """
        return (signals - signals.mean()) / signals.std()
        
    def _combine_signals(self, ml_signals: pd.Series, mean_rev_signals: pd.Series) -> pd.Series:
        """
        Combine signals using weighted average
        """
        combined = (
            self.ml_weight * ml_signals +
            self.mean_rev_weight * mean_rev_signals
        )
        return combined
        
    def _prepare_features(self) -> pd.DataFrame:
        """
        Prepare features for ML model
        """
        # This would be implemented based on ML model requirements
        pass
        
    def rank_securities(self) -> RankingData:
        """
        Rank securities based on combined signals
        """
        try:
            if self.current_signals is None:
                raise ValueError("No signals available for ranking")
            
            # Get combined signals
            scores = self.current_signals.combined_signals
            
            # Apply filters
            filtered_scores = self._apply_filters(scores)
            
            # Calculate rankings and percentiles
            rankings = filtered_scores.rank(ascending=False)
            percentiles = rankings / len(rankings)
            
            # Create ranking data container
            ranking_data = RankingData(
                rankings=rankings,
                percentiles=percentiles,
                filtered_scores=filtered_scores,
                timestamp=datetime.now()
            )
            
            # Store rankings
            self.current_rankings = ranking_data
            self.ranking_history.append(ranking_data)
            
            return ranking_data
            
        except Exception as e:
            self.logger.error(f"Error ranking securities: {e}")
            raise
            
    def _apply_filters(self, scores: pd.Series) -> pd.Series:
        """
        Apply risk-based filters to scores
        """
        filtered_scores = scores.copy()
        
        for symbol in scores.index:
            data = self.data[symbol]
            
            # Volume filter
            volume = data['Volume'].rolling(20).mean().iloc[-1]
            volume_rank = pd.Series([volume] + [
                self.data[s]['Volume'].rolling(20).mean().iloc[-1]
                for s in self.data.keys() if s != symbol
            ]).rank(pct=True).iloc[0]
            
            # Volatility filter
            volatility = data['Close'].pct_change().rolling(20).std().iloc[-1]
            volatility_rank = pd.Series([volatility] + [
                self.data[s]['Close'].pct_change().rolling(20).std().iloc[-1]
                for s in self.data.keys() if s != symbol
            ]).rank(pct=True).iloc[0]
            
            # Apply filters
            if volume_rank < 0.2 or volatility_rank > 0.8:
                filtered_scores[symbol] = 0
                
        return filtered_scores
        
    def select_positions(self) -> PositionData:
        """
        Select positions based on rankings
        """
        try:
            if self.current_rankings is None:
                raise ValueError("No rankings available for position selection")
                
            scores = self.current_rankings.filtered_scores
            
            # Sort scores
            sorted_scores = scores.sort_values(ascending=False)
            
            # Determine number of positions
            num_positions = max(
                self.min_positions,
                int(len(sorted_scores) * self.max_position_size)
            )
            
            # Select longs (positive scores above threshold)
            long_candidates = sorted_scores[
                sorted_scores > self.min_score_threshold
            ]
            longs = long_candidates.head(num_positions).index.tolist()
            
            # Select shorts (negative scores below negative threshold)
            short_candidates = sorted_scores[
                sorted_scores < -self.min_score_threshold
            ]
            shorts = short_candidates.tail(num_positions).index.tolist()
            
            # Calculate position sizes
            long_sizes = self._calculate_position_sizes(longs, scores)
            short_sizes = self._calculate_position_sizes(shorts, scores)
            
            # Create position data container
            position_data = PositionData(
                longs=longs,
                shorts=shorts,
                long_sizes=long_sizes,
                short_sizes=short_sizes,
                timestamp=datetime.now()
            )
            
            # Store positions
            self.current_positions = position_data
            self.position_history.append(position_data)
            
            return position_data
            
        except Exception as e:
            self.logger.error(f"Error selecting positions: {e}")
            raise
            
    def _calculate_position_sizes(
        self,
        symbols: List[str],
        scores: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate position sizes based on signal strength
        """
        if not symbols:
            return {}
            
        # Get absolute scores for selected symbols
        symbol_scores = scores[symbols].abs()
        
        # Normalize to sum to 1
        total_score = symbol_scores.sum()
        if total_score == 0:
            # Equal weight if all scores are 0
            weights = pd.Series(1.0 / len(symbols), index=symbols)
        else:
            weights = symbol_scores / total_score
            
        return weights.to_dict()
        
    def update(self, latest_data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Update signals and positions with latest data
        """
        try:
            # Generate new signals
            self.generate_signals(latest_data)
            
            # Update rankings
            self.rank_securities()
            
            # Update positions
            self.select_positions()
            
        except Exception as e:
            self.logger.error(f"Error updating factor pipeline: {e}")
            raise
            
    def get_tradeable_symbols(self) -> List[str]:
        """
        Get list of symbols currently considered for trading
        """
        if self.current_positions is None:
            return []
            
        return list(set(self.current_positions.longs + self.current_positions.shorts))
        
    def get_signal_metrics(self) -> pd.DataFrame:
        """
        Get detailed metrics for current signals
        """
        if self.current_signals is None or self.current_rankings is None:
            raise ValueError("No signals or rankings available")
            
        try:
            metrics_list = []
            
            for symbol in self.get_tradeable_symbols():
                metrics = {
                    'symbol': symbol,
                    'position': 'long' if symbol in self.current_positions.longs else 'short',
                    'ml_signal': self.current_signals.ml_signals[symbol],
                    'mean_rev_signal': self.current_signals.mean_rev_signals[symbol],
                    'combined_signal': self.current_signals.combined_signals[symbol],
                    'ranking': self.current_rankings.rankings[symbol],
                    'percentile': self.current_rankings.percentiles[symbol],
                    'position_size': (
                        self.current_positions.long_sizes.get(symbol) or 
                        -self.current_positions.short_sizes.get(symbol) or 
                        0
                    )
                }
                metrics_list.append(metrics)
                
            return pd.DataFrame(metrics_list)
            
        except Exception as e:
            self.logger.error(f"Error getting signal metrics: {e}")
            raise

    def get_current_positions(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Get current positions with sizes
        """
        if self.current_positions is None:
            return {}, {}
            
        return self.current_positions.long_sizes, self.current_positions.short_sizes
    

def test_factor_pipeline():
    """Test the FactorPipeline implementation"""
    import yaml
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    
    
    # Create dummy ML model and mean reversion analyzer
    class DummyMLModel:
        def predict(self, features):
            return pd.Series(np.random.randn(len(features)))
            
    class DummyMeanReversion:
        def generate_signals(self, data):
            scores = pd.Series(np.random.randn(len(data)))
            return type('Signals', (), {'signal_strength': scores})()
    
    # Create sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    data = {}
    for symbol in symbols:
        data[symbol] = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
    
    # Initialize pipeline
    pipeline = FactorPipeline(
        ml_model=DummyMLModel(),
        mean_reversion=DummyMeanReversion(),
        data=data,
        config=config
    )

    # Test update
    pipeline.update()
    
    # Print results
    print("\nCurrent Positions:")
    longs, shorts = pipeline.get_current_positions()
    print(f"Longs: {longs}")
    print(f"Shorts: {shorts}")
    
    print("\nSignal Metrics:")
    metrics = pipeline.get_signal_metrics()
    print(metrics)
    
    return pipeline

if __name__ == "__main__":
    pipeline = test_factor_pipeline()