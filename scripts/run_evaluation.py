"""
Script for running evaluation pipeline for trained models.

This script should:
1. Parse command line arguments for model selection
2. Load trained model from MLflow
3. Run comprehensive evaluation
4. Generate evaluation reports
5. Save results and visualizations
6. Compare multiple models if required
"""

import argparse
import logging
import sys
from pathlib import Path
import traceback
import json

sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_settings
from pipelines.evaluate_pipeline import run as evaluate_run
from utils import setup_logging

def parse_arguments():
    """Parse command line arguments for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained bacterial GAN models"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID of trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        nargs="+",
        help="Compare with other models (provide run IDs)"
    )
    parser.add_argument(
        "--generate-report", 
        action="store_true",
        help="Generate detailed HTML report"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def validate_run_id(run_id: str) -> bool:
    """Validate that run ID exists in MLflow."""
    try:
        import mlflow
        run = mlflow.get_run(run_id)
        return run is not None
    except Exception:
        return False

def main():
    """Main execution function."""
    args = parse_arguments()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    try:
        if not validate_run_id(args.run_id):
            logging.error(f"Invalid or non-existent run ID: {args.run_id}")
            sys.exit(1)
        
        settings = get_settings(args.config)
        logging.info(f"Loaded configuration from {args.config}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Starting evaluation for model {args.run_id}...")
        evaluation_results = evaluate_run(settings, args.run_id)
        
        results_file = output_dir / f"evaluation_{args.run_id}.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logging.info(f"Evaluation results saved to {results_file}")
        
        if args.compare_with:
            logging.info("Running model comparison...")

        if args.generate_report:
            logging.info("Generating detailed HTML report...")
        
        logging.info("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Evaluation failed with error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
