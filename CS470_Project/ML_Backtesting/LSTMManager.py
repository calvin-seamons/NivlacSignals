from typing import Dict, Any
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import torch
import yaml
import logging
from pathlib import Path
import numpy as np
from typing import Tuple
from LSTM import DirectionalLSTM
from FeatureEngineer import FeatureEngineering
from LSTM import LSTMConfig

CONFIG_PATH = Path("config/config.yaml")

class TimeSeriesSplitter:
    """
    Handles time series data splitting with proper temporal ordering and gap handling.
    """
    def __init__(self, validation_ratio: float = 0.2, gap_days: int = 5):
        self.validation_ratio = validation_ratio
        self.gap_days = gap_days
        
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time series data while maintaining temporal order and adding a gap.
        
        Args:
            data: DataFrame with DatetimeIndex
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation splits
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
            
        # Calculate split point
        split_idx = int(len(data) * (1 - self.validation_ratio))
        split_date = data.index[split_idx]
        
        # Create gap
        gap_start = split_date
        gap_end = split_date + pd.Timedelta(days=self.gap_days)
        
        # Split data
        train_data = data[:gap_start]
        val_data = data[gap_end:]
        
        return train_data, val_data


class LSTMManager:
    """
    Manages a directional prediction LSTM model for financial time series data.
    Handles model training, prediction, and feature engineering.
    
    The class is configured via a config.yaml file and maintains minimal state.
    """
    
    def __init__(self) -> None:
        """
        Initialize the LSTM manager by loading configuration and setting up components.
        
        Raises:
            FileNotFoundError: If config.yaml is not found
            yaml.YAMLError: If config.yaml is invalid
            ValueError: If required config parameters are missing
        """
        # Set up logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self.feature_engineering = FeatureEngineering(self.config['feature_params'])
        # Create LSTM config object

        lstm_config = LSTMConfig(
            input_size=self.config['model_params']['input_size'],
            hidden_size=self.config['model_params']['hidden_size'],
            num_layers=self.config['model_params']['num_layers'],
            dropout=self.config['model_params']['dropout'],
            bidirectional=self.config['model_params']['bidirectional'],
            attention_heads=self.config['model_params']['attention_heads'],
            use_layer_norm=self.config['model_params']['use_layer_norm'],
            residual_connections=self.config['model_params']['residual_connections'],
            confidence_threshold=self.config['model_params']['confidence_threshold']
        )
        
        # Initialize model with config object
        self.model = DirectionalLSTM(lstm_config)
        
        logging.info("LSTMManager initialized successfully")
    
    def _setup_logging(self) -> None:
        """Configure logging with appropriate format and level."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load and parse the config.yaml file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config.yaml is not found
            yaml.YAMLError: If config.yaml is invalid
        """
        try:
            with open(CONFIG_PATH) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {CONFIG_PATH}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise
    
    def _validate_config(self) -> None:
        """
        Validate that all required configuration parameters are present.
        
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required_sections = ['model_params', 'feature_params', 'training_params', 
                           'early_stopping_params', 'prediction_params']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate specific required parameters
        if 'sequence_length' not in self.config['model_params']:
            raise ValueError("Missing required parameter: model_params.sequence_length")
        
        if 'confidence_threshold' not in self.config['prediction_params']:
            raise ValueError("Missing required parameter: prediction_params.confidence_threshold")
    
    def train(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Enhanced training with proper data handling and scaling"""
        try:
            print("\nStarting train method...")
            print(f"Number of symbols in historical_data: {len(historical_data)}")
            print("Data shapes for each symbol:")
            for symbol, df in historical_data.items():
                print(f"{symbol}: {df.shape}")

            self._validate_historical_data(historical_data)
            
            # Sort all data by timestamp first
            sorted_data = {
                symbol: df.sort_index() 
                for symbol, df in historical_data.items()
            }
            
            # Initialize TimeSeriesSplitter
            splitter = TimeSeriesSplitter(
                validation_ratio=self.config['training_params'].get('validation_ratio', 0.2),
                gap_days=self.config['training_params'].get('gap_days', 5)
            )
            
            # Process and split features by symbol
            train_features = []
            val_features = []
            
            for symbol, data in sorted_data.items():
                logging.info(f"Processing features for {symbol}")
                
                # Generate raw features
                features = self.feature_engineering.process(data)
                
                # Split features into train/val
                train_df, val_df = splitter.split(features)
                
                train_features.append(train_df)
                val_features.append(val_df)
            
            # Combine features
            train_features = pd.concat(train_features)
            val_features = pd.concat(val_features)
            
            # Fit scalers on training data and transform both sets
            logging.info("Fitting and applying scalers...")
            scaled_train_features = self.feature_engineering.fit_transform(train_features)
            scaled_val_features = self.feature_engineering.transform(val_features)
            
            # Create sequences
            logging.info("Creating sequences...")
            train_sequences, train_labels = self._create_sequences(scaled_train_features)
            val_sequences, val_labels = self._create_sequences(scaled_val_features)

            # Add validation here
            self._validate_sequence_data(train_sequences, train_labels)
            self._validate_sequence_data(val_sequences, val_labels)
            
            # Compute class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            
            self.config['training_params']['class_weights'] = dict(
                enumerate(class_weights)
            )
            
            # Convert to tensors and create datasets
            train_data = self._prepare_torch_data(train_sequences, train_labels)
            val_data = self._prepare_torch_data(val_sequences, val_labels)
            
            # Train model
            metrics = self.model.train(
                train_data,
                validation_data=val_data,
                **self.config['training_params'],
                early_stopping_params=self.config['early_stopping_params']
            )
            
            return metrics
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}") from e
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate directional prediction for a single symbol with market regime awareness.
        
        Args:
            data: DataFrame containing OHLCV data for prediction
            
        Returns:
            Dictionary containing:
                - direction: 1 for up, 0 for down
                - probability: adjusted confidence in prediction
                - timestamp: prediction timestamp
                - market_regime: current market regime
                - volatility: current volatility level
                - prediction_horizon: number of days ahead prediction is for
                
        Raises:
            ValueError: If input data is invalid or insufficient
        """
        try:
            logging.info(f"Starting prediction with data shape: {data.shape}")
            
            # Validate input data
            self._validate_prediction_data(data)
            
            # Process features
            logging.info("Processing features...")
            features = self.feature_engineering.process(data)
            
            # Scale features using fitted scalers
            logging.info("Scaling features...")
            scaled_features = self.feature_engineering.transform(features)
            
            # Detect market regime
            market_regime = self._detect_market_regime(data)
            logging.info(f"Detected market regime: {market_regime}")
            
            # Calculate current volatility
            volatility = self._calculate_volatility(data)
            logging.info(f"Current volatility: {volatility:.4f}")
            
            # Create sequence
            sequence = self._create_prediction_sequence(scaled_features)
            logging.info(f"Created sequence with shape: {sequence.shape}")

            # Validate sequence
            self._validate_prediction_sequence(sequence)
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence).to(self.model.device)
            
            # Get base prediction
            logging.info("Getting base prediction from model...")
            with torch.no_grad():
                raw_probability = self.model(sequence_tensor).sigmoid().cpu().numpy()[0]
            logging.info(f"Raw prediction probability: {raw_probability:.4f}")
            
            # Adjust probability based on market regime
            adjusted_probability = self._adjust_probability_for_regime(
                raw_probability,
                market_regime,
                volatility
            )
            
            # Determine direction and final probability
            if adjusted_probability < 0.5:
                direction = 0
                final_probability = 1 - adjusted_probability
            else:
                direction = 1
                final_probability = adjusted_probability
                
            # Create prediction dictionary
            prediction = {
                'direction': direction,
                'probability': float(final_probability),
                'timestamp': data.index[-1],
                'market_regime': market_regime,
                'volatility': float(volatility),
                'metadata': {
                    'raw_probability': float(raw_probability),
                    'sequence_length': self.config['model_params']['sequence_length'],
                    'prediction_horizon': self.config['model_params']['prediction_horizon'],
                    'feature_count': scaled_features.shape[1]
                }
            }
            
            # Apply confidence threshold
            if abs(adjusted_probability - 0.5) < self.config['prediction_params']['confidence_threshold']:
                prediction['direction'] = None
                logging.info("Prediction below confidence threshold - returning None direction")
            
            return prediction
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect the current market regime based on recent price action.
        
        Args:
            data: DataFrame of OHLCV data
            
        Returns:
            str: One of ['trending_up', 'trending_down', 'ranging', 'high_volatility']
        """
        logging.info("Detecting market regime...")
        
        # Calculate required metrics
        returns = data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend metrics
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        
        current_price = data['Close'].iloc[-1]
        current_sma20 = sma_20.iloc[-1]
        current_sma50 = sma_50.iloc[-1]
        
        # Define volatility threshold from config
        vol_threshold = self.config['prediction_params'].get('volatility_threshold', 0.25)
        
        logging.info(f"Current volatility: {volatility:.4f}, Threshold: {vol_threshold}")
        logging.info(f"Price: {current_price:.4f}, SMA20: {current_sma20:.4f}, SMA50: {current_sma50:.4f}")
        
        # Determine regime
        if volatility > vol_threshold:
            regime = 'high_volatility'
        elif current_price > current_sma20 > current_sma50:
            regime = 'trending_up'
        elif current_price < current_sma20 < current_sma50:
            regime = 'trending_down'
        else:
            regime = 'ranging'
            
        logging.info(f"Detected regime: {regime}")
        return regime

    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate recent volatility.
        
        Args:
            data: DataFrame of OHLCV data
            window: Rolling window for volatility calculation
            
        Returns:
            float: Annualized volatility
        """
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
        logging.info(f"Calculated {window}-day annualized volatility: {volatility:.4f}")
        return volatility

    def _adjust_probability_for_regime(
        self,
        base_probability: float,
        market_regime: str,
        volatility: float
    ) -> float:
        """
        Adjust prediction probability based on market regime and volatility.
        
        Args:
            base_probability: Raw model probability
            market_regime: Detected market regime
            volatility: Current volatility level
            
        Returns:
            float: Adjusted probability
        """
        logging.info("Adjusting probability for market regime...")
        logging.info(f"Base probability: {base_probability:.4f}")
        
        # Get adjustment factors from config
        regime_factors = self.config['prediction_params'].get('regime_factors', {
            'trending_up': 1.1,
            'trending_down': 1.1,
            'ranging': 0.9,
            'high_volatility': 0.8
        })
        
        # Get volatility adjustment threshold
        vol_threshold = self.config['prediction_params'].get('volatility_threshold', 0.25)
        
        # Apply regime adjustment
        regime_factor = regime_factors.get(market_regime, 1.0)
        adjusted_probability = base_probability * regime_factor
        
        logging.info(f"After regime adjustment ({regime_factor}): {adjusted_probability:.4f}")
        
        # Apply volatility dampening if above threshold
        if volatility > vol_threshold:
            vol_dampening = 1.0 - min((volatility - vol_threshold) / vol_threshold, 0.5)
            adjusted_probability = 0.5 + (adjusted_probability - 0.5) * vol_dampening
            logging.info(f"After volatility dampening ({vol_dampening}): {adjusted_probability:.4f}")
        
        # Ensure probability stays in [0,1]
        adjusted_probability = min(max(adjusted_probability, 0.0), 1.0)
        
        logging.info(f"Final adjusted probability: {adjusted_probability:.4f}")
        return adjusted_probability
    
    def _validate_historical_data(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Validate historical data format and content."""
        if not historical_data:
            raise ValueError("Empty historical data provided")
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'Dividends', 'Stock Splits']
        
        for symbol, data in historical_data.items():
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data for symbol {symbol} must be a DataFrame")
                
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
                
            if len(data) < self.config['model_params']['sequence_length']:
                raise ValueError(
                    f"Insufficient data for {symbol}. Need at least "
                    f"{self.config['model_params']['sequence_length']} rows"
                )
    
    def _validate_prediction_data(self, data: pd.DataFrame) -> None:
        """Validate prediction data format and content."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Prediction data must be a DataFrame")
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'Dividends', 'Stock Splits']
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if len(data) < self.config['model_params']['sequence_length']:
            raise ValueError(
                f"Insufficient data. Need at least "
                f"{self.config['model_params']['sequence_length']} rows"
            )
    
    def _create_sequences(self, features: pd.DataFrame) -> tuple:
        """
        Create sequences for LSTM training with proper validation and logging.
        """
        logging.info(f"Creating sequences from features of shape: {features.shape}")
        
        sequence_length = self.config['model_params']['sequence_length']
        prediction_horizon = self.config['model_params']['prediction_horizon']
        threshold = self.config['model_params'].get('movement_threshold', 0.0)
        
        # Validate that features are properly formatted
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Features must be a DataFrame")
        
        # Get feature columns (excluding any target/label columns)
        feature_cols = [col for col in features.columns if col != 'target']
        
        sequences = []
        labels = []
        timestamps = []  # Track timestamps for proper ordering
        
        for i in range(len(features) - sequence_length - prediction_horizon):
            # Get the sequence of features
            seq = features[feature_cols].iloc[i:i+sequence_length].values
            
            # Validate sequence
            if np.isnan(seq).any():
                logging.warning(f"Skipping sequence at index {i} due to NaN values")
                continue
                
            # Calculate future return (assuming 'Close' is in features)
            current_price = features['Close'].iloc[i+sequence_length-1]
            future_price = features['Close'].iloc[i+sequence_length+prediction_horizon-1]
            
            future_return = (future_price - current_price) / current_price
            
            # Apply threshold for more meaningful moves
            if abs(future_return) < threshold:
                continue
                
            label = int(future_return > 0)
            
            sequences.append(seq)
            labels.append(label)
            timestamps.append(features.index[i+sequence_length])
        
        if not sequences:
            raise ValueError("No valid sequences created after applying criteria")
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        logging.info(f"Created sequences with shape: {sequences.shape}")
        logging.info(f"Label distribution: {np.bincount(labels)}")
        
        return sequences, labels
    
    def _create_prediction_sequence(self, features: pd.DataFrame) -> np.ndarray:
        """Create sequence for prediction."""
        sequence_length = self.config['model_params']['sequence_length']
        return features.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
    
    def _train_val_split(self, sequences: np.ndarray, labels: np.ndarray, 
                        val_ratio: float) -> tuple:
        """Split data into training and validation sets."""
        split_idx = int(len(sequences) * (1 - val_ratio))
        return (
            sequences[:split_idx], 
            sequences[split_idx:],
            labels[:split_idx], 
            labels[split_idx:]
        )
    
    def _prepare_torch_data(self, sequences: np.ndarray, labels: np.ndarray) -> torch.utils.data.DataLoader:
        """Prepare data for PyTorch training"""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(sequences),
            torch.LongTensor(labels)
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['training_params'].get('batch_size', 32),
            shuffle=True
        )

    def _validate_sequence_data(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        """Validate sequence data before training"""
        if len(sequences) != len(labels):
            raise ValueError("Sequence and label lengths do not match")
        
        if sequences.shape[1] != self.config['model_params']['sequence_length']:
            raise ValueError("Sequences have incorrect sequence length")
        
        if not np.isfinite(sequences).all():
            raise ValueError("Sequences contain non-finite values")
        
    def _validate_prediction_sequence(self, sequence: np.ndarray) -> None:
        """Validate a single prediction sequence"""
        if sequence.shape[0] != 1:
            raise ValueError("Prediction sequence should have batch size 1")
            
        if sequence.shape[1] != self.config['model_params']['sequence_length']:
            raise ValueError("Sequence has incorrect sequence length")
            
        if not np.isfinite(sequence).all():
            raise ValueError("Sequence contains non-finite values")