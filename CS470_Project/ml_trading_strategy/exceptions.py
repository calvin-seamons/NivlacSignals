"""
Custom exceptions for ML trading strategy implementation.
Defines specific error types for different components of the system.
"""

class DataPipelineError(Exception):
    """Base exception for errors in data pipeline operations."""
    def __init__(self, message: str = "Error in data pipeline operation"):
        self.message = message
        super().__init__(self.message)

class FeatureEngineeringError(Exception):
    """Exception for errors during feature engineering process."""
    def __init__(self, message: str = "Error in feature engineering process"):
        self.message = message
        super().__init__(self.message)

class MeanReversionError(Exception):
    """Exception for errors in mean reversion calculations."""
    def __init__(self, message: str = "Error in mean reversion calculation"):
        self.message = message
        super().__init__(self.message)

class ModelError(Exception):
    """Exception for errors in model operations."""
    def __init__(self, message: str = "Error in model operation"):
        self.message = message
        super().__init__(self.message)

class ValidationError(Exception):
    """Exception for data validation errors."""
    def __init__(self, message: str = "Data validation error"):
        self.message = message
        super().__init__(self.message)

class ConfigurationError(Exception):
    """Exception for configuration-related errors."""
    def __init__(self, message: str = "Configuration error"):
        self.message = message
        super().__init__(self.message)

class DataQualityError(Exception):
    """Exception for data quality issues."""
    def __init__(self, message: str = "Data quality error"):
        self.message = message
        super().__init__(self.message)

class UniverseCreationError(Exception):
    """Exception for errors in creating trading universe."""
    def __init__(self, message: str = "Error creating trading universe"):
        self.message = message
        super().__init__(self.message)

class SignalGenerationError(Exception):
    """Exception for errors in signal generation."""
    def __init__(self, message: str = "Error generating trading signals"):
        self.message = message
        super().__init__(self.message)

class BacktestError(Exception):
    """Exception for errors during backtesting."""
    def __init__(self, message: str = "Error during backtesting"):
        self.message = message
        super().__init__(self.message)

class PositionSizingError(Exception):
    """Exception for errors in position sizing calculations."""
    def __init__(self, message: str = "Error in position sizing calculation"):
        self.message = message
        super().__init__(self.message)

class RiskManagementError(Exception):
    """Exception for risk management related errors."""
    def __init__(self, message: str = "Risk management error"):
        self.message = message
        super().__init__(self.message)

class DataFetchError(Exception):
    """Exception for errors in fetching data."""
    def __init__(self, message: str = "Error fetching data"):
        self.message = message
        super().__init__(self.message)

class CacheError(Exception):
    """Exception for caching related errors."""
    def __init__(self, message: str = "Cache operation error"):
        self.message = message
        super().__init__(self.message)

class StrategyError(Exception):
    """Exception for errors in trading strategy implementation."""
    def __init__(self, message: str = "Trading strategy error"):
        self.message = message
        super().__init__(self.message)

class ModelPipelineError(Exception):
    """Exception for errors in model pipeline operations."""
    def __init__(self, message: str = "Error in model pipeline operation"):
        self.message = message
        super().__init__(self.message)