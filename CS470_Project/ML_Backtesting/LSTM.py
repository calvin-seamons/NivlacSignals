import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class LSTMConfig:
    """Configuration for LSTM model"""
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    attention_heads: int = 4
    use_layer_norm: bool = True
    residual_connections: bool = True
    confidence_threshold: float = 0.6  # Threshold for high-confidence predictions
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.input_size > 0, "Input size must be positive"
        assert self.hidden_size > 0, "Hidden size must be positive"
        assert self.num_layers > 0, "Number of layers must be positive"
        assert 0 <= self.dropout < 1, "Dropout must be between 0 and 1"
        assert self.attention_heads > 0, "Number of attention heads must be positive"
        assert 0.5 <= self.confidence_threshold < 1, "Confidence threshold must be between 0.5 and 1"

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Linear layers for query, key, value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        
        # Apply layer normalization first (pre-norm formulation)
        x_norm = self.layer_norm(x)
        
        # Linear projections and reshape for multi-head attention
        q = self.query(x_norm).view(batch_size, seq_length, self.num_heads, self.head_size)
        k = self.key(x_norm).view(batch_size, seq_length, self.num_heads, self.head_size)
        v = self.value(x_norm).view(batch_size, seq_length, self.num_heads, self.head_size)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_size)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Compute context vectors
        context = torch.matmul(attention, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.output(context)
        
        # Residual connection
        return x + self.dropout(output)

class LSTMLayer(nn.Module):
    """Single LSTM layer with layer normalization and residual connection"""
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1, 
                 bidirectional: bool = True, use_layer_norm: bool = True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.hidden_size = hidden_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2 if bidirectional else hidden_size,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, 
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Apply LSTM
        output, (h_n, c_n) = self.lstm(x, hx)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, (h_n, c_n)

class DirectionalLSTM(nn.Module):
    """
    Improved LSTM model with:
    - Bidirectional LSTM layers
    - Multi-head self-attention
    - Layer normalization
    - Residual connections
    - Gradient clipping
    """
    def __init__(self, config: LSTMConfig):
        super().__init__()
        config.validate()
        self.config = config
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(config.input_size)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        input_size = config.input_size
        
        for _ in range(config.num_layers):
            lstm_layer = LSTMLayer(
                input_size=input_size,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                bidirectional=config.bidirectional,
                use_layer_norm=config.use_layer_norm
            )
            self.lstm_layers.append(lstm_layer)
            input_size = config.hidden_size
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.attention_heads,
            dropout=config.dropout
        )
        
        # Final prediction layers
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size // 2)
        
        # Change output to 2 nodes (up/down probability)
        self.output = nn.Linear(config.hidden_size // 2, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.size()
        
        # Apply input normalization
        x = x.reshape(-1, features)
        x = self.input_norm(x)
        x = x.reshape(batch_size, seq_len, features)
        
        # Initial hidden states
        h_states = []
        c_states = []
        
        # Process LSTM layers with residual connections
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i > 0 and self.config.residual_connections:
                residual = x
            
            # Apply LSTM layer
            x, (h_n, c_n) = lstm_layer(x)
            
            # Store hidden states
            h_states.append(h_n)
            c_states.append(c_n)
            
            # Add residual connection if enabled
            if i > 0 and self.config.residual_connections:
                x = x + residual
        
        # Apply attention mechanism
        x = self.attention(x)
        
        # Final processing
        x = self.final_norm(x)
        x = self.dropout(x)
        
        # Get last sequence output
        x = x[:, -1, :]
        
        # Dense layers
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        
        # Output probabilities using softmax
        x = self.output(x)
        probabilities = F.softmax(x, dim=-1)
        
        return probabilities

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores
        
        Returns:
            Tuple containing:
            - predicted direction (0 for down, 1 for up)
            - confidence score (probability of predicted direction)
            - raw probabilities for both directions
        """
        probabilities = self(x)
        
        # Get predicted direction and confidence
        predicted_direction = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities, dim=1).values
        
        return predicted_direction, confidence, probabilities

    def configure_optimizers(self, learning_rate: float = 1e-3, 
                           weight_decay: float = 1e-5) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer and learning rate scheduler"""
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=100,
            steps_per_epoch=100,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        return optimizer, scheduler