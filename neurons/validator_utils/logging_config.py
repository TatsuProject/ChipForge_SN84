# neurons/validator_utils/logging_config.py

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from datetime import datetime
import sys


def setup_validator_logging(log_level: str = "INFO"):
    """
    Configure validator logging with daily rotation
    All logs saved to file, all logs shown in console/nohup output
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    
    # Console handler - shows all logs (for nohup output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Try to set up file logging with daily rotation
    try:
        # Create dated log filename
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = log_dir / f"validator_{today}.log"
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=50*1024*1024,  # 50MB per file
            backupCount=30,  # Keep 30 backup files
            encoding='utf-8',
            delay=True
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger = logging.getLogger(__name__)
        logger.info(f"📝 Validator logging configured - Level: {log_level}")
        logger.info(f"📂 Log file: {log_file.absolute()}")
        logger.info(f"📋 Console output: enabled (captured by nohup)")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error setting up file logging: {e}")
        logger.warning("📝 Using console logging only")
    
    # Suppress noisy third-party loggers
    logging.getLogger('async_substrate_interface').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('substrate_interface').setLevel(logging.WARNING)
    logging.getLogger('scalecodec').setLevel(logging.WARNING)
    logging.getLogger('bittensor').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    return root_logger