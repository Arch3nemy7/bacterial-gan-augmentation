"""
Script for running training pipeline with easy configuration.

This script should:
1. Parse command line arguments
2. Load configuration files
3. Setup logging and monitoring
4. Initialize MLflow tracking
5. Run training pipeline with error handling
6. Provide progress updates
7. Handle interruption gracefully
"""

import argparse
import logging
import sys
from pathlib import Path
import signal
import traceback

sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_settings
from pipelines.train_pipeline import run as train_run
from utils import setup_logging

def parse_arguments():
    """Parse command line arguments for training script."""
    parser = argparse.ArgumentParser(
        description="Run bacterial GAN training pipeline"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Resume training from checkpoint (provide run_id)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Perform dry run without actual training"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logging.info("Received interrupt signal. Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def validate_environment():
    """Validate environment and dependencies before training."""
    try:
        import tensorflow as tf
        import mlflow
        import numpy as np
        import PIL
        logging.info("All required dependencies are available")

        if tf.config.list_physical_devices('GPU'):
            logging.info("GPU detected and available")
        else:
            logging.warning("No GPU detected. Training will use CPU")
            
        return True
    except ImportError as e:
        logging.error(f"Missing dependency: {e}")
        return False

def main():
    """Main execution function."""
    args = parse_arguments()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    setup_signal_handlers()
    
    if not validate_environment():
        sys.exit(1)
    
    try:
        settings = get_settings(args.config)
        logging.info(f"Loaded configuration from {args.config}")
        
        if args.dry_run:
            logging.info("Dry run mode - configuration validation only")
            logging.info("Configuration is valid. Exiting.")
            sys.exit(0)
        
        logging.info("Starting training pipeline...")
        train_run(settings, resume_from=args.resume)
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
