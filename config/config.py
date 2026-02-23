"""
Configuration file for time-series forecasting pipeline.
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Data generation config
DATA_CONFIG = {
    'n_points': 2000,
    'n_features': 3,
    'seasonality_period': 24,
    'trend': True,
    'noise_level': 0.5,
    'seed': 42,
    'test_size': 0.2,
}

# Model training config
TRAINING_CONFIG = {
    'models': ['arima', 'lstm'],
    'arima': {
        'alpha': 0.3,
    },
    'lstm': {
        'lookback': 10,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
    },
}

# API config
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'models_dir': str(MODELS_DIR),
}

# Airflow config
AIRFLOW_CONFIG = {
    'dag_id': 'time_series_forecasting_pipeline',
    'owner': 'data-science-team',
    'email': ['airflow@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay_minutes': 5,
    'schedule_interval': '0 0 * * *',  # Daily at midnight
}

# Logging config
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': 'logs/pipeline.log',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

# Feature configuration
FEATURE_CONFIG = {
    'scaler': 'standard',  # 'standard' or 'minmax'
    'window_size': 24,  # lookback window size
    'forecast_horizon': 12,  # number of steps to forecast
}

# Evaluation metrics
METRICS = {
    'regression': ['mse', 'rmse', 'mae', 'r2'],
    'time_series': ['mape', 'mase', 'rmsse'],
}

# Database config (if using)
DATABASE_CONFIG = {
    'engine': 'sqlite',  # or 'postgresql', 'mysql', etc.
    'path': str(PROJECT_ROOT / 'data' / 'pipeline.db'),
    'log_predictions': False,
}

if __name__ == '__main__':
    print("Configuration loaded successfully")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
