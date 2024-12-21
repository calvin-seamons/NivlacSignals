import os
import json
import logging
import traceback
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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch normalization for inputs
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Additional dropout before final layer
        self.dropout = nn.Dropout(dropout)
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.size()
        
        # Apply batch normalization to each timestep
        x = x.reshape(-1, features)
        x = self.batch_norm(x)
        x = x.reshape(batch_size, seq_len, features)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get last timestep output and apply dropout
        out = self.dropout(lstm_out[:, -1, :])
        
        # Final prediction
        out = self.fc(out)
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
        
        # Get model parameters for filename
        model_version = config['model']['version']
        n_layers = config['model'].get('num_layers', 2)
        hidden_size = config['model'].get('hidden_size', 64)
        sequence_length = config['model'].get('sequence_length', 10)
        
        # Create descriptive filename
        model_name = f"lstm_layers{n_layers}_hidden{hidden_size}_seq{sequence_length}_v{model_version}"
        
        # Setup paths
        self.model_dir = config['paths']['model_dir']
        self.model_path = os.path.join(
            self.model_dir,
            f"{model_name}.pt"
        )
        self.metadata_path = os.path.join(
            self.model_dir,
            f"{model_name}_metadata.json"
        )
        self.scaler_path = os.path.join(
            self.model_dir,
            f"{model_name}_scaler.joblib"
        )
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # LSTM parameters
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.batch_size = config['model'].get('batch_size', 32)
        self.num_epochs = config['model'].get('num_epochs', 100)
        self.learning_rate = config['model'].get('learning_rate', 0.001)
        
        # Try to load existing model
        self._load_model()

    def train(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Train the model using provided historical data with financial metrics"""
        print("\n=== Starting Model Training Process ===")
        try:
            # Get features and create loaders
            X_train, X_val, y_train, y_val = self.feature_engineering.transform(
                historical_data, 
                train_size=self.config['model'].get('train_test_split', 0.8)
            )
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
            val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Initialize model with dropout
            input_size = X_train.shape[2]
            self.model = LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.config['model'].get('dropout', 0.2)
            )
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.config['model'].get('weight_decay', 0.0001)
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            # Initialize best metrics
            best_val_metrics = {
                'loss': float('inf'),
                'direction_accuracy': 0,
                'sharpe_ratio': -float('inf'),
                'information_coefficient': -float('inf')
            }
            patience_counter = 0
            
            # Training loop
            for epoch in range(self.num_epochs):
                # Training phase
                self.model.train()
                train_metrics = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                self.model.eval()
                val_metrics = self._validate_epoch(val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step(val_metrics['loss'])
                
                # Early stopping check based on multiple metrics
                improvement = (
                    val_metrics['loss'] < best_val_metrics['loss'] - self.config['model'].get('min_delta', 0.001) or
                    val_metrics['direction_accuracy'] > best_val_metrics['direction_accuracy'] + 0.01 or
                    val_metrics['sharpe_ratio'] > best_val_metrics['sharpe_ratio'] + 0.1
                )
                
                if improvement:
                    best_val_metrics = val_metrics.copy()
                    patience_counter = 0
                    self._save_model()
                    print(f"New best model saved with metrics: {val_metrics}")
                else:
                    patience_counter += 1
                
                # Print epoch metrics
                if (epoch + 1) % 5 == 0:
                    print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
                    print(f"Train metrics: {train_metrics}")
                    print(f"Val metrics: {val_metrics}")
                
                # Early stopping check
                if patience_counter >= self.config['model'].get('early_stopping_patience', 10):
                    print(f'\nEarly stopping triggered at epoch {epoch+1}')
                    break
            
            print("\n=== Training Complete ===")
            print(f"Best validation metrics: {best_val_metrics}")
            
        except Exception as e:
            print(f"\n!!! Error during model training !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            raise

    def _train_epoch(self, train_loader, optimizer, criterion):
        """Run one epoch of training"""
        total_loss = 0
        all_y_true = []
        all_y_pred = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            
            # Store predictions and actual values
            all_y_true.append(batch_y)
            all_y_pred.append(outputs.squeeze())
            
            # Calculate MSE loss
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Combine all batches
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred)
        
        # Calculate average loss and other metrics
        avg_loss = total_loss / len(train_loader)
        metrics = calculate_financial_metrics(y_true, y_pred)
        metrics['loss'] = avg_loss
        
        return metrics

    def _validate_epoch(self, val_loader, criterion):
        """Run one epoch of validation"""
        total_loss = 0
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                
                # Store predictions and actual values
                all_y_true.append(batch_y)
                all_y_pred.append(outputs.squeeze())
                
                # Calculate MSE loss
                loss = criterion(outputs, batch_y.unsqueeze(1))
                total_loss += loss.item()
        
        # Combine all batches
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred)
        
        # Calculate average loss and other metrics
        avg_loss = total_loss / len(val_loader)
        metrics = calculate_financial_metrics(y_true, y_pred)
        metrics['loss'] = avg_loss
        
        return metrics

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