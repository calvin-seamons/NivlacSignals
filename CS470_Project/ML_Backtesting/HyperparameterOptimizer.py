import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import json
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class OptimizationTrial:
    """Data class to store trial results"""
    parameters: Dict
    metrics: Dict
    trial_id: int

class HyperparameterOptimizer:
    """Class to handle hyperparameter optimization for LSTM models"""
    
    def __init__(self, config: Dict):
        """
        Initialize the optimizer with configuration
        
        Args:
            config (Dict): Configuration dictionary containing optimization parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Define parameter search spaces
        self.param_spaces = {
            'num_layers': [1, 2, 3, 4],
            'hidden_size': [32, 64, 128, 256],
            'learning_rate': [0.0001, 0.001, 0.01],
            'dropout': [0.1, 0.2, 0.3, 0.4],
            'sequence_length': [5, 10, 20, 30],
            'batch_size': [16, 32, 64, 128],
            'weight_decay': [0.0001, 0.001, 0.01]
        }
        
        # Optimization settings
        self.n_trials = config.get('optimization', {}).get('n_trials', 20)
        self.epochs_per_trial = config.get('optimization', {}).get('epochs_per_trial', 10)
        
        # Storage for trials
        self.trials: List[OptimizationTrial] = []
        self.best_trial: Optional[OptimizationTrial] = None
        
        # Setup results directory
        self.results_dir = os.path.join(
            config['paths']['model_dir'],
            'hyperparameter_optimization'
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Metrics to optimize (in order of importance)
        self.optimization_metrics = [
            ('sharpe_ratio', 'max'),
            ('direction_accuracy', 'max'),
            ('loss', 'min')
        ]

    def generate_parameters(self) -> Dict:
        """Generate random parameters from search spaces"""
        return {
            param: random.choice(values)
            for param, values in self.param_spaces.items()
        }

    def evaluate_trial(
        self,
        model,
        parameters: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate a single trial with given parameters
        
        Args:
            model: LSTM model instance
            parameters (Dict): Parameters to test
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Update model parameters
        model.update_parameters(parameters)
        
        best_val_metrics = {
            'loss': float('inf'),
            'direction_accuracy': 0,
            'sharpe_ratio': -float('inf'),
            'information_coefficient': -float('inf')
        }
        
        # Train for specified number of epochs
        for epoch in range(self.epochs_per_trial):
            # Train one epoch
            train_metrics = model.train_epoch(train_loader)
            
            # Validate
            val_metrics = model.validate(val_loader)
            
            # Update best metrics
            if self.is_better_trial(val_metrics, best_val_metrics):
                best_val_metrics = val_metrics.copy()
                
            # Early stopping check
            if val_metrics['loss'] > best_val_metrics['loss'] * 1.5:  # If loss is 50% worse than best
                break
                
        return best_val_metrics

    def is_better_trial(self, current_metrics: Dict, best_metrics: Dict) -> bool:
        """
        Compare trials based on multiple metrics
        
        Args:
            current_metrics (Dict): Metrics from current trial
            best_metrics (Dict): Metrics from best trial so far
            
        Returns:
            bool: True if current trial is better
        """
        for metric_name, direction in self.optimization_metrics:
            current_value = current_metrics[metric_name]
            best_value = best_metrics[metric_name]
            
            # Skip if values are too close
            if abs(current_value - best_value) < 1e-6:
                continue
                
            # Return True if current is better for this metric
            if direction == 'max':
                if current_value > best_value:
                    return True
                if current_value < best_value:
                    return False
            else:  # direction == 'min'
                if current_value < best_value:
                    return True
                if current_value > best_value:
                    return False
        
        return False  # If all metrics are equal

    def optimize(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        seed: Optional[int] = None
    ) -> Tuple[Dict, Dict]:
        """
        Run hyperparameter optimization
        
        Args:
            model: LSTM model instance
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            Tuple[Dict, Dict]: Best parameters and their metrics
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        print("\n=== Starting Hyperparameter Optimization ===")
        print(f"Number of trials: {self.n_trials}")
        print(f"Epochs per trial: {self.epochs_per_trial}")
        
        try:
            # Run trials
            for trial_id in tqdm(range(self.n_trials), desc="Optimization Progress"):
                # Generate parameters
                parameters = self.generate_parameters()
                
                # Evaluate trial
                metrics = self.evaluate_trial(model, parameters, train_loader, val_loader)
                
                # Create trial object
                trial = OptimizationTrial(
                    parameters=parameters,
                    metrics=metrics,
                    trial_id=trial_id
                )
                
                # Update best trial if better
                if (self.best_trial is None or 
                    self.is_better_trial(metrics, self.best_trial.metrics)):
                    self.best_trial = trial
                    print(f"\nNew best trial {trial_id}:")
                    print(f"Parameters: {parameters}")
                    print(f"Metrics: {metrics}")
                
                # Store trial
                self.trials.append(trial)
                
                # Save intermediate results
                self._save_results()
            
            print("\n=== Optimization Complete ===")
            print(f"Best parameters found: {self.best_trial.parameters}")
            print(f"Best metrics achieved: {self.best_trial.metrics}")
            
            return self.best_trial.parameters, self.best_trial.metrics
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise

    def _save_results(self) -> None:
        """Save optimization results to file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_trials': self.n_trials,
            'epochs_per_trial': self.epochs_per_trial,
            'param_spaces': self.param_spaces,
            'trials': [
                {
                    'trial_id': trial.trial_id,
                    'parameters': trial.parameters,
                    'metrics': trial.metrics
                }
                for trial in self.trials
            ],
            'best_trial': {
                'trial_id': self.best_trial.trial_id,
                'parameters': self.best_trial.parameters,
                'metrics': self.best_trial.metrics
            } if self.best_trial else None
        }
        
        # Save to file
        filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)