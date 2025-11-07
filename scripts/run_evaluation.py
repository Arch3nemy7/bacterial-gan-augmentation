"""
Script untuk menjalankan evaluation pipeline untuk trained models.

Script ini harus:
1. Parse command line arguments untuk model selection
2. Load trained model dari MLflow
3. Run comprehensive evaluation
4. Generate evaluation reports
5. Save results dan visualizations
6. Compare multiple models jika diperlukan
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
    """Parse command line arguments untuk evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained bacterial GAN models"
    )
    parser.add_argument(
        "--run-id", 
        type=str, 
        required=True,
        help="MLflow run ID dari trained model"
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
        help="Directory untuk menyimpan evaluation results"
    )
    parser.add_argument(
        "--compare-with", 
        type=str, 
        nargs="+",
        help="Compare dengan model lain (provide run IDs)"
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
    """Validate bahwa run ID exists di MLflow."""
    try:
        import mlflow
        run = mlflow.get_run(run_id)
        return run is not None
    except Exception:
        return False

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    try:
        # Validate run ID
        if not validate_run_id(args.run_id):
            logging.error(f"Invalid or non-existent run ID: {args.run_id}")
            sys.exit(1)
        
        # Load configuration
        settings = get_settings(args.config)
        logging.info(f"Loaded configuration from {args.config}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation pipeline
        logging.info(f"Starting evaluation for model {args.run_id}...")
        evaluation_results = evaluate_run(settings, args.run_id)
        
        # Save results
        results_file = output_dir / f"evaluation_{args.run_id}.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logging.info(f"Evaluation results saved to {results_file}")
        
        # Compare with other models if requested
        if args.compare_with:
            logging.info("Running model comparison...")
            # Implementation untuk model comparison
        
        # Generate detailed report if requested
        if args.generate_report:
            logging.info("Generating detailed HTML report...")
            # Implementation untuk HTML report generation
        
        logging.info("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Evaluation failed with error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
