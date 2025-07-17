import logging
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

from config.settings import APP_CONFIG


def setup_logging(log_level: str = None, log_file: str = None) -> logging.Logger:
    """Set up logging configuration"""
    
    # Use config defaults if not provided
    if log_level is None:
        log_level = APP_CONFIG.log_level
    
    if log_file is None:
        log_file = APP_CONFIG.log_file
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logger
    logger = logging.getLogger('address_matcher')
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler for {log_file}: {e}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(f'address_matcher.{name}')