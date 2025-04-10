import logging
import os
from logging.handlers import RotatingFileHandler

def configure_logging(log_directory='logs'):
    """
    Configures logging for the application.

    Args:
        log_directory (str): Directory to store log files. Defaults to 'logs'.

    Returns:
        dict: A dictionary of configured loggers for each module.
    """
    # Ensure the log directory exists
    os.makedirs(log_directory, exist_ok=True)
    
    # Define the logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()]  # Log to stdout for Docker
    )
    
    # Create module-specific loggers
    modules = ['data_processing', 'training', 'predict', 'predict_api']
    loggers = {}

    for module in modules:
        # Create a logger for the module
        logger = logging.getLogger(f'vehicle_classifier.{module}')
        
        # Add a rotating file handler for the module
        file_handler = RotatingFileHandler(
            filename=os.path.join(log_directory, f'{module}.log'),
            maxBytes=10485760,  # 10MB per file
            backupCount=5,      # Keep up to 5 backup files
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Add the handler to the logger
        logger.addHandler(file_handler)
        
        # Store the logger in the dictionary
        loggers[module] = logger

    return loggers
