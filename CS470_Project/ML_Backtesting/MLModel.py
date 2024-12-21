import os
import json
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
from FeatureEngineering import FeatureEngineering

from utils import calculate_financial_metrics

class TimeSeriesDataset(Dataset):
    """Custom Dataset for LSTM input"""
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        """
        Initialize the dataset with proper reshaping
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, sequence_length, n_features)
            y (np.ndarray): Target values
            sequence_length (int): Length of each sequence
        """
        # Ensure X is 3D
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D, got {X.ndim}D instead")
            
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (sequence, target) where sequence is of shape (sequence_length, n_features)
        """
        return self.X[idx], self.y[idx]

class LSTM(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Shape verification
        if x.dim() != 3:
            raise ValueError(f"Expected input to be 3D (batch_size, sequence_length, input_size), got {x.dim()}D instead")
        
        batch_size, seq_len, input_size = x.size()
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class MLModel:
    """
    Machine Learning Model Manager
    Handles model training, prediction, and persistence
    """
    def __init__(self, config: Dict):
        """
        Initialize MLModel with configuration
        
        Args:
            config (Dict): Configuration dictionary containing model parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = None
        self.feature_engineering = FeatureEngineering(config)
        self.scaler = None
        
        # Setup paths
        self.model_dir = config['paths']['model_dir']
        self.model_path = os.path.join(
            self.model_dir,
            f"model_{config['model']['version']}.pt"
        )
        self.metadata_path = os.path.join(
            self.model_dir,
            f"metadata_{config['model']['version']}.json"
        )
        self.scaler_path = os.path.join(
            self.model_dir,
            f"scaler_{config['model']['version']}.joblib"
        )
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # LSTM parameters
        self.sequence_length = config['model'].get('sequence_length', 10)
        self.hidden_size = config['model'].get('hidden_size', 64)
        self.num_layers = config['model'].get('num_layers', 2)
        self.batch_size = config['model'].get('batch_size', 32)
        self.num_epochs = config['model'].get('num_epochs', 100)
        self.learning_rate = config['model'].get('learning_rate', 0.001)
        
        # Try to load existing model
        self._load_model()

    def train(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Train the model using provided historical data"""
        print("\n=== Starting Model Training Process ===")
        print(f"Number of symbols in input data: {len(historical_data)}")
        
        # Print sample of input data structure
        sample_symbol = list(historical_data.keys())[0]
        print(f"\nSample of input data for {sample_symbol}:")
        print(historical_data[sample_symbol].head())
        print(f"Input data shape for {sample_symbol}: {historical_data[sample_symbol].shape}")
        
        try:
            print("\n=== Feature Engineering Phase ===")
            # Get features and create train loader
            print("Starting feature transformation...")
            X_train, X_test, y_train, y_test = self.feature_engineering.transform(
                historical_data, 
                train_size=self.config['model'].get('train_test_split', 0.8)
            )
            
            # Print shapes after feature engineering
            print("\nData shapes after feature engineering:")
            print(f"X_train shape: {X_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"y_test shape: {y_test.shape}")
            
            # Create dataset and dataloader
            print("\n=== Creating DataLoader ===")
            train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            print(f"Number of batches in train_loader: {len(train_loader)}")
            
            # Initialize model
            print("\n=== Initializing Model ===")
            input_size = X_train.shape[2]  # Number of features
            print(f"Input size (number of features): {input_size}")
            print(f"Hidden size: {self.hidden_size}")
            print(f"Number of layers: {self.num_layers}")
            
            self.model = LSTM(input_size, self.hidden_size, self.num_layers)
            
            # Print model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Total number of model parameters: {total_params}")
            
            # Log a sample batch
            sample_batch = next(iter(train_loader))
            print("\nSample batch shapes:")
            print(f"Batch X shape: {sample_batch[0].shape}")
            print(f"Batch y shape: {sample_batch[1].shape}")
            
            # Training setup
            print("\n=== Setting Up Training ===")
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.config['model'].get('weight_decay', 0.0001)
            )
            
            # Initialize best loss for early stopping
            best_loss = float('inf')
            patience_counter = 0
            
            # Training loop
            print("\n=== Starting Training Loop ===")
            self.model.train()
            
            for epoch in range(self.num_epochs):
                total_loss = 0
                batch_count = 0
                
                for batch_X, batch_y in train_loader:
                    # Print shapes for first batch of first epoch
                    if epoch == 0 and batch_count == 0:
                        print(f"\nFirst batch shapes:")
                        print(f"batch_X shape: {batch_X.shape}")
                        print(f"batch_y shape: {batch_y.shape}")
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    
                    # Print shapes for first batch of first epoch
                    if epoch == 0 and batch_count == 0:
                        print(f"outputs shape: {outputs.shape}")
                        print(f"batch_y unsqueezed shape: {batch_y.unsqueeze(1).shape}")
                    
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                avg_loss = total_loss / len(train_loader)
                
                if (epoch + 1) % 5 == 0:  # Print every 5 epochs
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {avg_loss:.6f}')
                
                # Early stopping check
                if avg_loss < best_loss - self.config['model'].get('min_delta', 0.001):
                    best_loss = avg_loss
                    patience_counter = 0
                    print(f"New best loss: {best_loss:.6f}")
                    # Save best model
                    self._save_model()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['model'].get('early_stopping_patience', 10):
                        print(f'\nEarly stopping triggered at epoch {epoch+1}')
                        break
            
            print("\n=== Training Complete ===")
            print(f"Final best loss: {best_loss:.6f}")
            
        except Exception as e:
            print(f"\n!!! Error during model training !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate predictions for input data
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping symbols to OHLCV DataFrames
            
        Returns:
            Dict[str, pd.Series]: Dictionary mapping symbols to their predictions
        """
        self.logger.info(f"Starting prediction process for {len(data)} symbols")
        
        try:
            # Load model if not loaded
            if self.model is None:
                self._load_model()
                if self.model is None:
                    self.logger.warning("No trained model found. Please train the model first.")
                    raise ValueError("No trained model available")
            
            # Feature engineering
            features = self.feature_engineering.transform_predict(data)
            
            # Make predictions
            self.model.eval()
            predictions = {}
            
            with torch.no_grad():
                for symbol in data.keys():
                    # Get features for this symbol
                    symbol_features = torch.FloatTensor(features[symbol]).unsqueeze(0)
                    
                    # Generate prediction
                    pred = self.model(symbol_features)
                    predictions[symbol] = pd.Series(
                        pred.numpy().flatten(),
                        index=data[symbol].index[-len(pred):]
                    )
            
            return predictions
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Load saved model and metadata if they exist"""
        try:
            if os.path.exists(self.model_path):
                print("\nFound existing model checkpoint")
                try:
                    # Try to load the model
                    self.model = LSTM(
                        self.config['model']['input_size'],
                        self.hidden_size,
                        self.num_layers
                    )
                    self.model.load_state_dict(torch.load(self.model_path))
                    self.model.eval()
                    
                    # Load metadata
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    
                    # Load scaler
                    if os.path.exists(self.scaler_path):
                        self.scaler = joblib.load(self.scaler_path)
                    
                    print("Successfully loaded existing model and metadata")
                except Exception as e:
                    print(f"Warning: Could not load existing model due to: {str(e)}")
                    print("Will train a new model instead")
                    # Remove old model files since they're incompatible
                    if os.path.exists(self.model_path):
                        os.remove(self.model_path)
                    if os.path.exists(self.metadata_path):
                        os.remove(self.metadata_path)
                    if os.path.exists(self.scaler_path):
                        os.remove(self.scaler_path)
                    self.model = None
            else:
                print("\nNo existing model found - will train a new one")
                self.model = None
                    
        except Exception as e:
            print(f"\nError during model loading: {str(e)}")
            print("Will train a new model instead")
            self.model = None

    def _save_model(self) -> None:
        """Save model, metadata, and scaler"""
        try:
            # Save model
            torch.save(self.model.state_dict(), self.model_path)
            
            # Save metadata
            metadata = {
                'version': self.config['model']['version'],
                'training_date': datetime.now().isoformat(),
                'model_type': 'LSTM',
                'parameters': {
                    'sequence_length': self.sequence_length,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers
                }
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Save scaler
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
            
            self.logger.info("Successfully saved model and metadata")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise