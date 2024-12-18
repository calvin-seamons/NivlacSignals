import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import minimize

from config.logging_config import get_logger
from config.settings import (
    PORTFOLIO_STRATEGY_PARAMS as STRATEGY_PARAMS,
    OPTIMIZATION_PARAMS
)
from data.data_pipeline import DataPipeline
from models.mean_reversion_analyzer import MeanReversionSignals
from exceptions import StrategyError

@dataclass
class PortfolioState:
    """Container for portfolio state information"""
    positions: Dict[str, float]  # Current position sizes
    returns: pd.Series  # Historical returns
    metrics: Dict[str, float]  # Performance metrics
    last_rebalance: datetime  # Last rebalance date
    cash: float  # Available cash

class PortfolioStrategy:
    def __init__(
        self,
        data_pipeline: DataPipeline,
        initial_capital: float = 1_000_000,
        strategy_params: Optional[Dict] = None,
        optimization_params: Optional[Dict] = None
    ):
        """
        Initialize Portfolio Strategy
        
        Args:
            data_pipeline: Instance of DataPipeline
            initial_capital: Initial portfolio capital
            strategy_params: Optional override for STRATEGY_PARAMS
            optimization_params: Optional override for OPTIMIZATION_PARAMS
        """
        self.logger = get_logger(self.__class__.__name__)
        self.data_pipeline = data_pipeline
        self.initial_capital = initial_capital
        
        # Load parameters from settings with optional overrides
        self.strategy_params = strategy_params or STRATEGY_PARAMS
        self.optimization_params = optimization_params or OPTIMIZATION_PARAMS
        
        # Extract commonly used parameters
        self.rebalance_days = self.strategy_params['rebalance_frequency']
        self.max_position_size = self.strategy_params['max_position_size']
        self.min_positions = self.strategy_params['min_positions']
        self.target_leverage = self.strategy_params['target_leverage']
        self.risk_free_rate = self.strategy_params['risk_free_rate']
        
        # Set transaction cost parameters
        self.transaction_costs = self.strategy_params['transaction_costs']
        
        # Set risk management parameters
        self.stop_loss = self.strategy_params['stop_loss']
        self.profit_target = self.strategy_params['profit_target']
        self.portfolio_stop_loss = self.strategy_params['portfolio_stop_loss']
        
        # Initialize portfolio state
        self.portfolio_state = PortfolioState(
            positions={},
            returns=pd.Series(),
            metrics={},
            last_rebalance=None,
            cash=initial_capital
        )
        
        # Performance tracking
        self.monthly_returns = pd.DataFrame()
        self.monthly_volumes = pd.DataFrame()
        
    def update_monthly_metrics(self):
        """Update monthly return and volume metrics"""
        try:
            returns_dict = {}
            volumes_dict = {}
            
            for symbol in self.data_pipeline.universe:
                data = self.data_pipeline.processed_data[symbol]
                
                # Calculate monthly returns
                monthly_returns = data['Close'].resample('M').last().pct_change()
                returns_dict[symbol] = monthly_returns
                
                # Calculate monthly volumes
                monthly_volumes = data['Volume'].resample('M').mean()
                volumes_dict[symbol] = monthly_volumes
            
            self.monthly_returns = pd.DataFrame(returns_dict)
            self.monthly_volumes = pd.DataFrame(volumes_dict)
            
        except Exception as e:
            self.logger.error(f"Error updating monthly metrics: {e}")
            raise StrategyError(f"Error updating monthly metrics: {e}")
    
    def optimize_portfolio(
        self,
        signals: MeanReversionSignals,
        returns: pd.DataFrame,
        constraints: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization
        
        Args:
            signals: Mean reversion signals
            returns: Historical returns matrix
            constraints: Additional optimization constraints
            
        Returns:
            Dictionary of optimal position sizes
        """
        try:
            # Combine long and short signals
            symbols = signals.longs + signals.shorts
            if len(symbols) < 2:
                raise StrategyError("Insufficient symbols for optimization")
            
            # Get relevant returns
            portfolio_returns = returns[symbols].dropna()
            
            # Calculate expected returns (use signal strength as proxy)
            expected_returns = pd.Series(index=symbols)
            for symbol in symbols:
                if symbol in signals.longs:
                    expected_returns[symbol] = signals.signal_strength[symbol]
                else:
                    expected_returns[symbol] = -signals.signal_strength[symbol]
            
            # Calculate covariance matrix
            cov_matrix = portfolio_returns.cov()
            
            # Define objective function (negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                return -sharpe
            
            # Define constraints
            constraints = []
            
            # Sum of absolute weights equals target leverage
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(np.abs(x)) - self.target_leverage
            })
            
            # Position limits
            bounds = []
            for symbol in symbols:
                if symbol in signals.longs:
                    bounds.append((0, self.max_position_size))
                else:
                    bounds.append((-self.max_position_size, 0))
            
            # Initial guess
            n_assets = len(symbols)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Run optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result.success:
                raise StrategyError("Portfolio optimization failed")
            
            # Convert results to dictionary
            optimal_weights = {
                symbol: weight 
                for symbol, weight in zip(symbols, result.x)
            }
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            raise StrategyError(f"Error optimizing portfolio: {e}")
    
    def rebalance_portfolio(
        self,
        signals: MeanReversionSignals,
        current_date: datetime
    ) -> Dict[str, float]:
        """
        Rebalance portfolio based on signals and optimization
        
        Args:
            signals: Mean reversion signals
            current_date: Current trading date
            
        Returns:
            Dictionary of new position sizes
        """
        try:
            # Check if rebalance is needed
            if (self.portfolio_state.last_rebalance and 
                (current_date - self.portfolio_state.last_rebalance).days < self.rebalance_days):
                return self.portfolio_state.positions
            
            # Update monthly metrics
            self.update_monthly_metrics()
            
            # Ensure minimum positions per side
            if len(signals.longs) < self.min_positions or len(signals.shorts) < self.min_positions:
                self.logger.warning("Insufficient positions for balanced portfolio")
                return self.portfolio_state.positions
            
            # Get optimal weights
            optimal_weights = self.optimize_portfolio(
                signals,
                self.monthly_returns
            )
            
            # Update portfolio state
            self.portfolio_state.positions = optimal_weights
            self.portfolio_state.last_rebalance = current_date
            
            # Calculate trade sizes
            trades = {
                symbol: optimal_weights.get(symbol, 0) - 
                self.portfolio_state.positions.get(symbol, 0)
                for symbol in set(optimal_weights) | set(self.portfolio_state.positions)
            }
            
            # Log rebalance summary
            self.logger.info(f"Portfolio rebalanced on {current_date}")
            self.logger.info(f"New positions: {optimal_weights}")
            self.logger.info(f"Trades executed: {trades}")
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            raise StrategyError(f"Error rebalancing portfolio: {e}")
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        try:
            if not self.portfolio_state.positions:
                return {}
            
            # Calculate portfolio returns
            position_returns = pd.Series(0, index=self.monthly_returns.index)
            for symbol, weight in self.portfolio_state.positions.items():
                position_returns += weight * self.monthly_returns[symbol]
            
            # Calculate metrics
            metrics = {
                'total_return': (1 + position_returns).prod() - 1,
                'annual_return': position_returns.mean() * 12,
                'annual_volatility': position_returns.std() * np.sqrt(12),
                'sharpe_ratio': (position_returns.mean() * 12 - self.risk_free_rate) / 
                               (position_returns.std() * np.sqrt(12)),
                'max_drawdown': (1 + position_returns).cumprod().div(
                    (1 + position_returns).cumprod().cummax()
                ).min() - 1,
                'current_leverage': sum(abs(w) for w in self.portfolio_state.positions.values())
            }
            
            self.portfolio_state.metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            raise StrategyError(f"Error calculating portfolio metrics: {e}")