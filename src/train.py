"""
Training pipeline for time-series forecasting models.
Handles model training, validation, and evaluation.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesModel:
    """Base class for time-series models."""
    
    def __init__(self, model_name):
        """Initialize model."""
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train):
        """Fit model to training data."""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError
    
    def save(self, filepath):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ARIMAModel(TimeSeriesModel):
    """
    ARIMA-based time-series forecasting model.
    Using a simple exponential smoothing implementation.
    """
    
    def __init__(self, alpha=0.3):
        super().__init__("ARIMA")
        self.alpha = alpha
        self.last_value = None
    
    def fit(self, X_train, y_train):
        """Fit exponential smoothing model."""
        logger.info(f"Training {self.model_name} model...")
        self.last_value = y_train.iloc[-1] if isinstance(y_train, pd.Series) else y_train[-1]
        logger.info(f"Model fitted. Last value: {self.last_value}")
        return self
    
    def predict(self, X):
        """Make predictions using exponential smoothing."""
        predictions = []
        current = self.last_value
        
        for _ in range(len(X)):
            current = self.alpha * current + (1 - self.alpha) * current
            predictions.append(current)
        
        return np.array(predictions)


class LSTMModel(TimeSeriesModel):
    """LSTM-based time-series forecasting model."""
    
    def __init__(self, lookback=10):
        super().__init__("LSTM")
        self.lookback = lookback
        self.weights = None
    
    def fit(self, X_train, y_train):
        """Fit LSTM model (simplified implementation)."""
        logger.info(f"Training {self.model_name} model...")
        # In a real implementation, this would train an actual LSTM
        self.weights = np.random.randn(X_train.shape[1], 1)
        logger.info(f"Model fitted with shape: {X_train.shape}")
        return self
    
    def predict(self, X):
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Simplified linear prediction
        predictions = X @ self.weights
        return predictions.flatten()


class TrainingPipeline:
    """Main training pipeline for time-series forecasting."""
    
    def __init__(self, model_type='arima', output_dir='models'):
        """
        Initialize training pipeline.
        
        Args:
            model_type: Type of model ('arima' or 'lstm')
            output_dir: Directory to save models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_type.lower() == 'arima':
            self.model = ARIMAModel()
        elif model_type.lower() == 'lstm':
            self.model = LSTMModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.metrics = {}
    
    def load_data(self, train_path, test_path=None):
        """Load training (and optionally test) data."""
        logger.info(f"Loading training data from {train_path}")
        self.train_data = pd.read_csv(train_path, index_col=0)
        
        if test_path:
            logger.info(f"Loading test data from {test_path}")
            self.test_data = pd.read_csv(test_path, index_col=0)
        else:
            self.test_data = None
        
        return self
    
    def preprocess(self):
        """Preprocess data for training."""
        logger.info("Preprocessing data...")
        
        # Extract features and target
        # For time-series, we'll use all columns as features
        self.X_train = self.train_data.iloc[:, :-1] if self.train_data.shape[1] > 1 else self.train_data
        self.y_train = self.train_data.iloc[:, -1]
        
        if self.test_data is not None:
            self.X_test = self.test_data.iloc[:, :-1] if self.test_data.shape[1] > 1 else self.test_data
            self.y_test = self.test_data.iloc[:, -1]
        
        logger.info(f"Training data shape: {self.X_train.shape}")
        if self.test_data is not None:
            logger.info(f"Test data shape: {self.X_test.shape}")
        
        return self
    
    def train(self):
        """Train the model."""
        logger.info("Starting model training...")
        self.model.fit(self.X_train, self.y_train)
        logger.info("Model training completed")
        return self
    
    def evaluate(self):
        """Evaluate model on test data."""
        if self.test_data is None:
            logger.warning("No test data available for evaluation")
            return self
        
        logger.info("Evaluating model...")
        predictions = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        rmse = np.sqrt(mse)
        
        self.metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        logger.info(f"Evaluation Metrics:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        
        return self
    
    def save_model(self):
        """Save trained model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.output_dir / f"model_{self.model.model_name}_{timestamp}.pkl"
        self.model.save(str(model_path))
        
        # Save metrics
        metrics_path = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        return self
    
    def run(self, train_path, test_path=None):
        """Run complete training pipeline."""
        logger.info("="*60)
        logger.info("Starting Time-Series Forecasting Training Pipeline")
        logger.info("="*60)
        
        self.load_data(train_path, test_path)
        self.preprocess()
        self.train()
        self.evaluate()
        self.save_model()
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)


def main():
    """Main entry point for training pipeline."""
    # Create pipeline
    pipeline = TrainingPipeline(model_type='arima', output_dir='models')
    
    # Run pipeline
    pipeline.run(
        train_path='data/train_data.csv',
        test_path='data/test_data.csv'
    )
    
    # Print final metrics
    print("\nFinal Metrics:")
    for metric, value in pipeline.metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
