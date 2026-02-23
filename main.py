"""
Main entry script for the time-series forecasting pipeline.
Provides command-line interface for various pipeline tasks.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generation import TimeSeriesDataGenerator, generate_train_test_split, normalize_data
from src.train import TrainingPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_data(args):
    """Generate synthetic or load real time-series data."""
    logger.info("Starting data generation...")
    
    generator = TimeSeriesDataGenerator(seed=args.seed)
    
    if args.source == 'synthetic':
        df = generator.generate_synthetic_data(
            n_points=args.n_points,
            n_features=args.n_features,
            seasonality_period=args.seasonality_period,
            trend=args.trend,
            noise_level=args.noise_level
        )
        logger.info(f"Generated synthetic data with shape: {df.shape}")
    else:
        df = generator.load_data(args.data_path)
        logger.info(f"Loaded data from {args.data_path} with shape: {df.shape}")
    
    # Save raw data
    generator.save_data(df, args.output_raw)
    
    # Split and normalize
    train_df, test_df = generate_train_test_split(df, test_size=args.test_size)
    train_norm, test_norm, mean, std = normalize_data(train_df, test_df)
    
    # Save processed data
    train_norm.to_csv(args.output_train)
    test_norm.to_csv(args.output_test)
    
    import json
    scaling_params = {
        "mean": mean.to_dict(),
        "std": std.to_dict()
    }
    with open("data/scaling_params.json", "w") as f:
        json.dump(scaling_params, f, indent=2)
    
    logger.info(f"Data generation complete")
    logger.info(f"  Raw data: {args.output_raw}")
    logger.info(f"  Train data: {args.output_train}")
    logger.info(f"  Test data: {args.output_test}")


def train(args):
    """Train time-series forecasting model."""
    logger.info(f"Starting training for {args.model_type} model...")
    
    pipeline = TrainingPipeline(
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    
    pipeline.run(
        train_path=args.train_data,
        test_path=args.test_data
    )
    
    logger.info("Training completed successfully!")
    print("\nFinal Metrics:")
    for metric, value in pipeline.metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")


def predict(args):
    """Make predictions using a trained model."""
    import pickle
    import pandas as pd
    import numpy as np
    
    logger.info(f"Loading model from {args.model_path}...")
    
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded model: {type(model).__name__}")
    
    # Load input data
    if args.input_data.endswith('.csv'):
        data = pd.read_csv(args.input_data)
        logger.info(f"Loaded data from {args.input_data} with shape: {data.shape}")
    else:
        # Parse JSON input
        data = np.array(eval(args.input_data))
        logger.info(f"Parsed input data with shape: {data.shape}")
    
    # Make predictions
    predictions = model.predict(data)
    
    logger.info(f"Predictions shape: {predictions.shape}")
    print(f"\nPredictions (first 10):")
    print(predictions[:10])
    
    # Save predictions if specified
    if args.output:
        pd.DataFrame(predictions, columns=['prediction']).to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Time-Series Forecasting Pipeline'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data generation command
    data_parser = subparsers.add_parser('generate', help='Generate or load data')
    data_parser.add_argument('--source', choices=['synthetic', 'real'], default='synthetic',
                            help='Data source type')
    data_parser.add_argument('--n-points', type=int, default=2000,
                            help='Number of time points (synthetic data)')
    data_parser.add_argument('--n-features', type=int, default=3,
                            help='Number of features')
    data_parser.add_argument('--seasonality-period', type=int, default=24,
                            help='Seasonality period')
    data_parser.add_argument('--trend', action='store_true', default=True,
                            help='Add trend component')
    data_parser.add_argument('--noise-level', type=float, default=0.5,
                            help='Noise level')
    data_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
    data_parser.add_argument('--test-size', type=float, default=0.2,
                            help='Test set proportion')
    data_parser.add_argument('--data-path', type=str,
                            help='Path to real data file')
    data_parser.add_argument('--output-raw', default='data/raw_data.csv',
                            help='Output path for raw data')
    data_parser.add_argument('--output-train', default='data/train_data.csv',
                            help='Output path for training data')
    data_parser.add_argument('--output-test', default='data/test_data.csv',
                            help='Output path for test data')
    data_parser.set_defaults(func=generate_data)
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model-type', choices=['arima', 'lstm'], default='arima',
                             help='Model type to train')
    train_parser.add_argument('--train-data', default='data/train_data.csv',
                             help='Path to training data')
    train_parser.add_argument('--test-data', default='data/test_data.csv',
                             help='Path to test data')
    train_parser.add_argument('--output-dir', default='models',
                             help='Output directory for models')
    train_parser.set_defaults(func=train)
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model-path', required=True,
                               help='Path to trained model file')
    predict_parser.add_argument('--input-data', required=True,
                               help='Input data (CSV file path or JSON string)')
    predict_parser.add_argument('--output', type=str,
                               help='Output CSV file for predictions')
    predict_parser.set_defaults(func=predict)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
