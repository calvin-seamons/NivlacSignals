# models/model_pipeline.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost, Pool
import shap

from config.settings import (
    MODEL_PARAMS, TRAIN_PARAMS, FEATURE_PARAMS, 
    EVALUATION_METRICS, PERFORMANCE_THRESHOLDS,
    DEFAULT_MODEL_TYPE
)
from config.logging_config import get_logger

class ModelPipelineError(Exception):
    """Custom exception for model pipeline errors"""
    pass

class ModelPipeline:
    def __init__(
    self,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_params: Optional[dict] = None,
    train_params: Optional[dict] = None
    ):
        """
        Initialize model pipeline
        
        Args:
            model_type: Type of model from settings.DEFAULT_MODEL_TYPE
            model_params: Override default MODEL_PARAMS if provided
            train_params: Override default TRAIN_PARAMS if provided
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_type = model_type.lower()

        self.logger.info(f"Initializing model pipeline with {self.model_type} model")
        
        # Use settings or override with provided params
        self.model_params = model_params or MODEL_PARAMS[self.model_type]
        self.train_params = train_params or TRAIN_PARAMS
        
        # Initialize from settings
        self.feature_params = FEATURE_PARAMS
        self.evaluation_metrics = EVALUATION_METRICS
        self.performance_thresholds = PERFORMANCE_THRESHOLDS
        
        # Initialize model and storage
        self.model = None
        self.feature_importance = None
        self.validation_metrics = {}

    def create_time_series_splits(
        self,
        X: np.ndarray,
        n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-series cross-validation splits
        
        Args:
            X: Feature matrix
            n_splits: Number of splits
            
        Returns:
            List of (train, test) indices
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X))

    def _initialize_model(self):
        """Initialize the selected model type"""
        try:
            if self.model_type == 'lightgbm':
                self.model = lgb.LGBMRegressor(**self.model_params)
            elif self.model_type == 'xgboost':
                self.model = xgb.XGBRegressor(**self.model_params)
            elif self.model_type == 'catboost':
                self.model = CatBoost(self.model_params)
            else:
                raise ModelPipelineError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            raise ModelPipelineError(f"Error initializing model: {e}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Train the model
        """
        try:
            self._initialize_model()
            
            # Generate default feature names if none provided
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                
            # Convert feature_names to list if it's not already
            feature_names = list(feature_names)
            
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

            if self.model_type == 'lightgbm':
                fit_params = {
                    'eval_set': eval_set,
                    'feature_name': feature_names,
                    'callbacks': [lgb.early_stopping(
                        stopping_rounds=self.model_params.get('early_stopping_rounds', 50),
                        verbose=False
                    )]
                }
                self.model.fit(X_train, y_train, **fit_params)
                
            elif self.model_type == 'xgboost':
                # Create DMatrix with feature names
                dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names) if X_val is not None else None
                
                # Set up watchlist
                watchlist = [(dtrain, 'train')]
                if dval is not None:
                    watchlist.append((dval, 'eval'))
                
                # Train model
                self.model = xgb.train(
                    self.model_params,
                    dtrain,
                    evals=watchlist,
                    verbose_eval=False
                )
                
            elif self.model_type == 'catboost':
                train_pool = Pool(X_train, y_train, feature_names=feature_names)
                val_pool = Pool(X_val, y_val, feature_names=feature_names) if X_val is not None else None
                
                self.model.fit(
                    train_pool,
                    eval_set=val_pool,
                    verbose=False
                )

            # Calculate feature importance
            self._calculate_feature_importance()
            
            # Return training metrics
            return self.calculate_metrics(y_train, self.predict(X_train))
            
        except Exception as e:
            raise ModelPipelineError(f"Error during training: {e}")

    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        try:
            if self.model_type == 'lightgbm':
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=self.model.feature_name_
                )
            elif self.model_type == 'xgboost':
                # For XGBoost, feature importance is stored differently
                self.feature_importance = pd.Series(
                    self.model.get_score(importance_type='gain'),
                    name='importance'
                ).sort_values(ascending=False)
            elif self.model_type == 'catboost':
                self.feature_importance = pd.Series(
                    self.model.get_feature_importance(),
                    index=self.model.feature_names_
                )
        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            raise ModelPipelineError("Model not trained. Call train() first.")
            
        try:
            if self.model_type == 'xgboost':
                # Create DMatrix for prediction
                dtest = xgb.DMatrix(X)
                return self.model.predict(dtest)
            else:
                return self.model.predict(X)
        except Exception as e:
            raise ModelPipelineError(f"Error during prediction: {e}")

    def calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for feature importance"""
        try:
            if self.model_type == 'lightgbm':
                explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'xgboost':
                explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'catboost':
                explainer = shap.TreeExplainer(self.model)
            
            return explainer.shap_values(X)
            
        except Exception as e:
            self.logger.warning(f"Error calculating SHAP values: {e}")
            return None

    def save_model(self, path: str):
        """Save model to file"""
        if self.model is None:
            raise ModelPipelineError("No model to save")
            
        try:
            if self.model_type == 'lightgbm':
                self.model.booster_.save_model(path)
            elif self.model_type == 'xgboost':
                self.model.save_model(path)
            elif self.model_type == 'catboost':
                self.model.save_model(path)
        except Exception as e:
            raise ModelPipelineError(f"Error saving model: {e}")

    def load_model(self, path: str):
        """Load model from file"""
        try:
            self._initialize_model()
            if self.model_type == 'lightgbm':
                self.model = lgb.Booster(model_file=path)
            elif self.model_type == 'xgboost':
                self.model.load_model(path)
            elif self.model_type == 'catboost':
                self.model.load_model(path)
        except Exception as e:
            raise ModelPipelineError(f"Error loading model: {e}")
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for model evaluation
        """
        try:
            # Ensure arrays are the correct shape
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
            
            metrics = {}
            
            # Basic regression metrics
            metrics.update({
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            })
            
            # Directional accuracy (ensure binary classification)
            y_true_direction = np.where(y_true > 0, 1, 0)
            y_pred_direction = np.where(y_pred > 0, 1, 0)
            
            metrics.update({
                'directional_accuracy': accuracy_score(y_true_direction, y_pred_direction),
                'precision': precision_score(y_true_direction, y_pred_direction, 
                                        average='binary', zero_division=0),
                'recall': recall_score(y_true_direction, y_pred_direction, 
                                    average='binary', zero_division=0)
            })
            
            if returns is not None:
                # Ensure returns is 1D
                returns = returns.reshape(-1)
                
                # Calculate strategy returns
                strategy_returns = np.where(y_pred > 0, returns, -returns)
                
                # Trading metrics calculations
                metrics.update(self._calculate_trading_metrics(strategy_returns))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise ModelPipelineError(f"Error calculating metrics: {e}")

    def _calculate_trading_metrics(self, strategy_returns: np.ndarray) -> Dict[str, float]:
        """Helper function to calculate trading-specific metrics"""
        metrics = {}
        
        try:
            # Sharpe Ratio (annualized)
            sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if len(strategy_returns) > 1 else 0
            
            # Cumulative returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            # Maximum Drawdown
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns / running_max - 1
            max_drawdown = drawdowns.min()
            
            # Hit Ratio
            profitable_trades = np.sum(strategy_returns > 0)
            total_trades = len(strategy_returns)
            hit_ratio = profitable_trades / total_trades if total_trades > 0 else 0
            
            # Profit Factor
            gross_profits = np.sum(strategy_returns[strategy_returns > 0])
            gross_losses = abs(np.sum(strategy_returns[strategy_returns < 0]))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            metrics.update({
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'hit_ratio': hit_ratio,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'avg_return': strategy_returns.mean(),
                'return_std': strategy_returns.std() if len(strategy_returns) > 1 else 0,
                'cumulative_return': cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
            })
            
        except Exception as e:
            self.logger.warning(f"Error calculating trading metrics: {e}")
            
        return metrics

    def evaluate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            X: Feature matrix (must have same number of features as training data)
            y: True values
            returns: Optional returns for financial metrics
        """
        try:
            # Check feature dimensions
            if self.model is None:
                raise ModelPipelineError("Model not trained. Call train() first.")
                
            expected_features = self.model.n_features_in_  # Get expected number of features
            
            # Ensure X is 2D with correct number of features
            if X.ndim == 1:
                if len(X) != expected_features:
                    raise ModelPipelineError(
                        f"Input has {len(X)} features but model expects {expected_features} features"
                    )
                X = X.reshape(1, -1)
            elif X.shape[1] != expected_features:
                raise ModelPipelineError(
                    f"Input has {X.shape[1]} features but model expects {expected_features} features"
                )
                
            # Generate predictions
            predictions = self.predict(X)
            
            # Ensure y and predictions are the right shape
            y = y.reshape(-1)
            predictions = predictions.reshape(-1)
            
            if returns is not None:
                returns = returns.reshape(-1)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y, predictions, returns)
            
            # Store metrics
            self.validation_metrics = metrics
            
            # Log key metrics
            self.logger.info("Model Evaluation Results:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"{metric}: {value:.4f}")
                else:
                    self.logger.info(f"{metric}: {value}")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise ModelPipelineError(f"Error evaluating model: {e}")