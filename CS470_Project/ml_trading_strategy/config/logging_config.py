"""
Logging configuration for ML trading strategy.
Provides centralized logging setup with both console and file handlers.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import settings
from config.settings import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
    BASE_DIR
)

class MLTradingLogger:
    """Singleton logger class for ML trading application"""
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, name: str = 'ml_trading') -> logging.Logger:
        """Get or create logger instance"""
        if cls._instance is None:
            cls._instance = cls._setup_logger(name)
        return cls._instance
    
    @staticmethod
    def _setup_logger(name: str) -> logging.Logger:
        """Configure and setup logger with handlers"""
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(LOG_LEVEL)
        
        # Prevent duplicate logging
        if logger.handlers:
            return logger
            
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            LOG_FORMAT,
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(console_handler)
        
        # File Handler
        if LOG_FILE:
            log_dir = Path(BASE_DIR) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f'{name}_{datetime.now():%Y%m%d}.log'
            )
            file_handler.setFormatter(logging.Formatter(
                LOG_FORMAT,
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(file_handler)
        
        return logger

# Convenience function
def get_logger(name: str = 'ml_trading') -> logging.Logger:
    """Get logger instance"""
    return MLTradingLogger.get_logger(name)