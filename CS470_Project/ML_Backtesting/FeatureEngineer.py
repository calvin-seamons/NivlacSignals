from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import logging
from dataclasses import dataclass
from enum import Enum

# Import feature group classes
from Features import PriceFeatures, ReturnFeatures, MomentumFeatures, VolatilityFeatures, VolumeFeatures

class ScalingMethod(Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"

@dataclass
class FeatureGroup:
    """Container for feature group configuration"""
    name: str
    scaler_type: ScalingMethod
    enabled: bool = True
    
class FeatureEngineering:
    """
    Coordinates feature generation and scaling across different feature groups.
    Each feature group is implemented in a separate class.
    """
    
    def __init__(self, config: Dict) -> None:
        """
        Initialize feature engineering with configuration.
        
        Args:
            config: Configuration dictionary containing feature parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize feature groups
        self.feature_groups = [
            FeatureGroup("price", ScalingMethod.ROBUST),
            FeatureGroup("returns", ScalingMethod.ROBUST),
            FeatureGroup("momentum", ScalingMethod.STANDARD),
            FeatureGroup("volatility", ScalingMethod.STANDARD),
            FeatureGroup("volume", ScalingMethod.ROBUST)
        ]
        
        # Initialize feature generators
        self.feature_generators = {
            "price": PriceFeatures(config.get("price_features", {})),
            "returns": ReturnFeatures(config.get("return_features", {})),
            "momentum": MomentumFeatures(config.get("momentum_features", {})),
            "volatility": VolatilityFeatures(config.get("volatility_features", {})),
            "volume": VolumeFeatures(config.get("volume_features", {}))
        }
        
        # Initialize scalers
        self.scalers = {}
        self._initialize_scalers()
        
        self.logger.info("FeatureEngineering initialized successfully")
        
    def _initialize_scalers(self) -> None:
        """Initialize scalers for each feature group."""
        for group in self.feature_groups:
            if group.scaler_type == ScalingMethod.STANDARD:
                self.scalers[group.name] = StandardScaler()
            else:
                self.scalers[group.name] = RobustScaler()
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data structure and content.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if len(data) < self.config.get("min_samples", 100):
            raise ValueError(
                f"Insufficient data. Need at least "
                f"{self.config.get('min_samples', 100)} samples"
            )
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the input data.
        
        Args:
            data: Input OHLCV DataFrame
            
        Returns:
            DataFrame containing all generated features
        """
        print("\nFeatureEngineering process started...")
        print(f"Input data shape: {data.shape}")
        print(f"Input data columns: {data.columns.tolist()}")
        print(f"Input data index type: {type(data.index)}")
        print(f"First few dates in index: {data.index[:5]}")
        
        try:
            # Check if we have a multi-index
            if isinstance(data.index, pd.MultiIndex):
                # If we have a multi-index, we already have processed features
                return data
            
            # Validate input data
            print(f"\nValidating data with min_samples = {self.config.get('min_samples', 100)}")
            self._validate_data(data)
            
            # Initialize with original OHLCV data
            all_features = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Generate features for each enabled group
            for group in self.feature_groups:
                if not group.enabled:
                    continue
                    
                self.logger.info(f"Generating {group.name} features")
                
                try:
                    # Get features from appropriate generator
                    generator = self.feature_generators[group.name]
                    features = generator.generate(data)
                    
                    # Add prefix to avoid name collisions
                    features = features.add_prefix(f"{group.name}_")
                    
                    # Add to results, preserving original OHLCV columns
                    all_features = pd.concat([all_features, features], axis=1)
                    
                    self.logger.info(
                        f"Generated {len(features.columns)} {group.name} features"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error generating {group.name} features: {str(e)}")
                    raise
            
            # Handle any invalid values
            all_features = self._handle_invalid_values(all_features)
            
            self.logger.info(f"Completed feature generation. Shape: {all_features.shape}")
            return all_features
            
        except Exception as e:
            self.logger.error(f"Feature generation failed: {str(e)}")
            raise
    
    def _handle_invalid_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Handle invalid values in features DataFrame.
        
        Args:
            features: DataFrame of generated features
            
        Returns:
            DataFrame with invalid values handled
        """
        # Replace infinite values with NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        features = features.ffill()
        
        # Backward fill any remaining NaN values
        features = features.bfill()
        
        # Replace any remaining NaN values with 0
        features = features.fillna(0)
        
        return features
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate features and fit scalers on the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Numpy array of scaled features
        """
        # Generate features
        features = self.process(data)
        
        # Fit and transform each feature group
        scaled_features = pd.DataFrame(index=features.index)
        
        for group in self.feature_groups:
            if not group.enabled:
                continue
                
            # Get columns for this group
            group_cols = [col for col in features.columns if col.startswith(f"{group.name}_")]
            
            if group_cols:
                # Fit scaler and transform
                scaler = self.scalers[group.name]
                scaled = scaler.fit_transform(features[group_cols])
                scaled_features[group_cols] = scaled
        
        return scaled_features.values
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate features and transform using fitted scalers.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Numpy array of scaled features
        """
        # Generate features
        features = self.process(data)
        
        # Transform each feature group
        scaled_features = pd.DataFrame(index=features.index)
        
        for group in self.feature_groups:
            if not group.enabled:
                continue
                
            # Get columns for this group
            group_cols = [col for col in features.columns if col.startswith(f"{group.name}_")]
            
            if group_cols:
                # Transform using fitted scaler
                scaler = self.scalers[group.name]
                scaled = scaler.transform(features[group_cols])
                scaled_features[group_cols] = scaled
        
        return scaled_features.values
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        features = []
        for group in self.feature_groups:
            if group.enabled:
                generator = self.feature_generators[group.name]
                group_features = [
                    f"{group.name}_{feature}"
                    for feature in generator.get_feature_names()
                ]
                features.extend(group_features)
        return features
    
    def get_num_features(self) -> int:
        """Get total number of features."""
        return len(self.get_feature_names())