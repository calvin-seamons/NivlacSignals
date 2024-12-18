import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

from config.logging_config import get_logger
from config.settings import (
    PORTFOLIO_STRATEGY_PARAMS,
    OPTIMIZATION_PARAMS,
    PREDICTION_PARAMS
)

@dataclass
class FactorSignals:
    """Container for factor signals and metrics"""
    combined_scores: pd.Series
    rankings: pd.Series
    longs: List[str]
    shorts: List[str]
    signal_metrics: Dict[str, float]

class FactorPipeline:
    def __init__(
        self,
        mean_reversion_analyzer,
        model_pipeline,
        data_pipeline,
        mean_reversion_weight: float = 0.5,
        model_weight: float = 0.5,
        min_score_threshold: float = 0.1
    ):
        """
        Initialize FactorPipeline.
        
        Args:
            mean_reversion_analyzer: Instance of MeanReversionAnalyzer
            model_pipeline: Instance of ModelPipeline
            data_pipeline: Instance of DataPipeline
            mean_reversion_weight: Weight for mean reversion signals
            model_weight: Weight for ML model signals
            min_score_threshold: Minimum score threshold for position selection
        """
        self.logger = get_logger(self.__class__.__name__)
        
        # Store components
        self.mean_reversion_analyzer = mean_reversion_analyzer
        self.model_pipeline = model_pipeline
        self.data_pipeline = data_pipeline
        
        # Store weights
        self.mean_reversion_weight = mean_reversion_weight
        self.model_weight = model_weight
        self.min_score_threshold = min_score_threshold
        
        # Load settings
        self.max_position_size = PORTFOLIO_STRATEGY_PARAMS['max_position_size']
        self.min_positions = PORTFOLIO_STRATEGY_PARAMS['min_positions']
        self.max_sector_exposure = PORTFOLIO_STRATEGY_PARAMS['max_sector_exposure']
        self.position_limits = PORTFOLIO_STRATEGY_PARAMS['position_limits']
        self.confidence_threshold = PREDICTION_PARAMS['confidence_threshold']
        
        # Initialize storage
        self.current_signals: Optional[FactorSignals] = None
        self.longs: List[str] = []
        self.shorts: List[str] = []
        
    def _normalize_signals(self, signals: pd.Series) -> pd.Series:
        """
        Normalize signals using z-score method
        """
        return (signals - signals.mean()) / signals.std()
        
    def _combine_signals(
        self,
        mean_rev_signals: pd.Series,
        model_signals: pd.Series
    ) -> pd.Series:
        """
        Combine signals using weighted average
        """
        # Normalize both signals
        norm_mean_rev = self._normalize_signals(mean_rev_signals)
        norm_model = self._normalize_signals(model_signals)
        
        # Combine signals
        combined = (
            self.mean_reversion_weight * norm_mean_rev +
            self.model_weight * norm_model
        )
        
        return combined
        
    def _apply_risk_filters(
        self,
        scores: pd.Series,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Apply risk-based filters to scores
        """
        filtered_scores = scores.copy()
        
        # Apply volume filter
        volume_threshold = data['Volume'].mean() * 0.5
        low_volume_mask = data['Volume'] < volume_threshold
        filtered_scores[low_volume_mask] = 0
        
        # Apply volatility filter
        volatility = data['Close'].pct_change().rolling(20).std()
        high_vol_mask = volatility > volatility.quantile(0.9)
        filtered_scores[high_vol_mask] = 0
        
        return filtered_scores
        
    def generate_combined_signals(self, latest_data: Dict[str, pd.DataFrame]) -> FactorSignals:
        """
        Generate combined signals from mean reversion and ML models
        """
        try:
            # Get mean reversion signals
            mr_signals = self.mean_reversion_analyzer.generate_signals()
            mr_scores = mr_signals.signal_strength
            
            # Get ML model predictions
            features = self.data_pipeline.get_latest_data()
            model_predictions = self.model_pipeline.predict(features)
            model_scores = pd.Series(model_predictions, index=features.index)
            
            # Combine signals
            combined_scores = self._combine_signals(mr_scores, model_scores)
            
            # Apply risk filters
            filtered_scores = self._apply_risk_filters(combined_scores, latest_data)
            
            # Generate rankings
            rankings = filtered_scores.rank(ascending=False)
            
            # Select positions
            positions = self._select_positions(filtered_scores, rankings)
            
            # Calculate signal metrics
            signal_metrics = {
                'mean_score': filtered_scores.mean(),
                'score_std': filtered_scores.std(),
                'num_longs': len(positions['longs']),
                'num_shorts': len(positions['shorts'])
            }
            
            # Create and store signals
            self.current_signals = FactorSignals(
                combined_scores=filtered_scores,
                rankings=rankings,
                longs=positions['longs'],
                shorts=positions['shorts'],
                signal_metrics=signal_metrics
            )
            
            # Update class properties
            self.longs = positions['longs']
            self.shorts = positions['shorts']
            
            return self.current_signals
            
        except Exception as e:
            self.logger.error(f"Error generating combined signals: {e}")
            raise
            
    def _select_positions(
        self,
        scores: pd.Series,
        rankings: pd.Series
    ) -> Dict[str, List[str]]:
        """
        Select long and short positions based on scores and rankings
        """
        try:
            positions = {'longs': [], 'shorts': []}
            
            # Sort scores
            sorted_scores = scores.sort_values(ascending=False)
            
            # Select top and bottom positions
            num_positions = max(
                self.min_positions,
                int(len(sorted_scores) * self.max_position_size)
            )
            
            # Select longs (positive scores above threshold)
            long_candidates = sorted_scores[sorted_scores > self.min_score_threshold]
            positions['longs'] = long_candidates.head(num_positions).index.tolist()
            
            # Select shorts (negative scores below negative threshold)
            short_candidates = sorted_scores[sorted_scores < -self.min_score_threshold]
            positions['shorts'] = short_candidates.tail(num_positions).index.tolist()
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error selecting positions: {e}")
            raise
            
    def update(self, latest_data: Dict[str, pd.DataFrame]) -> None:
        """
        Update signals and positions with latest data
        """
        try:
            self.generate_combined_signals(latest_data)
        except Exception as e:
            self.logger.error(f"Error updating factor pipeline: {e}")
            raise
            
    def get_signal_metrics(self) -> pd.DataFrame:
        """
        Get detailed metrics for current signals
        """
        if self.current_signals is None:
            raise ValueError("No signals generated yet")
            
        try:
            metrics_list = []
            
            for symbol in self.longs + self.shorts:
                metrics = {
                    'symbol': symbol,
                    'position': 'long' if symbol in self.longs else 'short',
                    'score': self.current_signals.combined_scores[symbol],
                    'ranking': self.current_signals.rankings[symbol]
                }
                metrics_list.append(metrics)
                
            return pd.DataFrame(metrics_list)
            
        except Exception as e:
            self.logger.error(f"Error getting signal metrics: {e}")
            raise