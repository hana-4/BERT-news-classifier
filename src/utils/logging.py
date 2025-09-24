"""
Logging utilities for BERT News Recommender
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    use_json: bool = False,
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (if None, uses timestamp)
        log_dir: Directory for log files
        use_json: Whether to use JSON formatting
        logger_name: Name of the logger (if None, uses root logger)
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Choose formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"bert_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_path / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time"""
    import time
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    
    return wrapper


class ModelMetricsLogger:
    """Logger for model training and inference metrics"""
    
    def __init__(self, logger_name: str = "model_metrics"):
        self.logger = logging.getLogger(logger_name)
    
    def log_training_metrics(self, epoch: int, step: int, metrics: dict):
        """Log training metrics"""
        self.logger.info(
            "Training metrics",
            extra={
                'extra_fields': {
                    'epoch': epoch,
                    'step': step,
                    'metrics': metrics,
                    'type': 'training'
                }
            }
        )
    
    def log_validation_metrics(self, epoch: int, metrics: dict):
        """Log validation metrics"""
        self.logger.info(
            "Validation metrics",
            extra={
                'extra_fields': {
                    'epoch': epoch,
                    'metrics': metrics,
                    'type': 'validation'
                }
            }
        )
    
    def log_inference_metrics(self, text: str, prediction: dict, latency: float):
        """Log inference metrics"""
        self.logger.info(
            "Inference completed",
            extra={
                'extra_fields': {
                    'text_length': len(text),
                    'prediction': prediction,
                    'latency_ms': latency * 1000,
                    'type': 'inference'
                }
            }
        )
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log errors with context"""
        self.logger.error(
            f"{error_type}: {error_message}",
            extra={
                'extra_fields': {
                    'error_type': error_type,
                    'context': context or {},
                    'type': 'error'
                }
            }
        )


# Initialize default logger
default_logger = setup_logging()
