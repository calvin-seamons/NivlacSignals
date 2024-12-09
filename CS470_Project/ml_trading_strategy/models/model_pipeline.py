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
    MODEL_TYPE, TRAIN_TEST_SPLIT, VALIDATION_SIZE, 
    LOOKBACK_PERIODS, TARGET_HORIZON
)
from config.logging_config import get_logger

class ModelPipelineError(Exception):
    """Custom exception for model pipeline errors"""
    pass

class ModelPipeline:
    def __init__(
        self,
        model_type: str = MODEL_TYPE,
        model_params: Optional[dict] = None,
        validation_size: float = VALIDATION_SIZE
    ):
        """
        Initialize model pipeline
        
        Args:
            model_type: Type of model to use ('lightgbm', 'xgboost', 'catboost')
            model_params: Model hyperparameters
            validation_size: Size of validation set
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_type = model_type.lower()
        self.validation_size = validation_size
        self.model = None
        self.feature_importance = None
        
        # Set default parameters based on model type
        self.model_params = model_params or self._get_default_params()

    def _get_default_params(self) -> dict:
        """Get default parameters for selected model type"""
        if self.model_type == 'lightgbm':
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'n_estimators': 100,
                'early_stopping_rounds': 50
            }
        elif self.model_type == 'xgboost':
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'early_stopping_rounds': 50
            }
        elif self.model_type == 'catboost':
            return {
                'loss_function': 'RMSE',
                'iterations': 100,
                'learning_rate': 0.05,
                'depth': 6,
                'early_stopping_rounds': 50
            }
        else:
            raise ModelPipelineError(f"Unsupported model type: {self.model_type}")

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
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of training metrics
        """
        try:
            self._initialize_model()
            
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

            if self.model_type == 'lightgbm':
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
                
            elif self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
                
            elif self.model_type == 'catboost':
                train_pool = Pool(X_train, y_train)
                val_pool = Pool(X_val, y_val) if X_val is not None else None
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            raise ModelPipelineError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        try:
            if self.model_type == 'lightgbm':
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=self.model.feature_name_
                )
            elif self.model_type == 'xgboost':
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=self.model.feature_names_in_
                )
            elif self.model_type == 'catboost':
                self.feature_importance = pd.Series(
                    self.model.get_feature_importance(),
                    index=self.model.feature_names_
                )
        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {e}")

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

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