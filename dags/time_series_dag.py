"""
Apache Airflow DAG for time-series forecasting pipeline orchestration.
Manages data generation, training, and predictions.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation import TimeSeriesDataGenerator, generate_train_test_split, normalize_data
from src.train import TrainingPipeline


# Default arguments for the DAG
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['airflow@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG Definition
dag = DAG(
    'time_series_forecasting_pipeline',
    default_args=default_args,
    description='Time-series forecasting pipeline with data generation, training, and evaluation',
    schedule_interval='0 0 * * *',  # Daily at midnight
    catchup=False,
    tags=['time-series', 'forecasting', 'ml'],
)


def generate_data_task():
    """Generate synthetic time-series data."""
    print("Starting data generation task...")
    
    generator = TimeSeriesDataGenerator(seed=42)
    df = generator.generate_synthetic_data(
        n_points=2000,
        n_features=3,
        seasonality_period=24,
        trend=True,
        noise_level=0.5
    )
    
    # Save raw data
    generator.save_data(df, "data/raw_data.csv")
    
    # Split into train and test
    train_df, test_df = generate_train_test_split(df, test_size=0.2)
    
    # Normalize data
    train_norm, test_norm, mean, std = normalize_data(train_df, test_df)
    
    # Save processed data
    train_norm.to_csv("data/train_data.csv")
    test_norm.to_csv("data/test_data.csv")
    
    import json
    scaling_params = {
        "mean": mean.to_dict(),
        "std": std.to_dict()
    }
    with open("data/scaling_params.json", "w") as f:
        json.dump(scaling_params, f, indent=2)
    
    print(f"✓ Data generation complete. Generated {df.shape[0]} samples with {df.shape[1]} features")
    return {"status": "success", "rows": df.shape[0], "features": df.shape[1]}


def train_arima_task():
    """Train ARIMA model."""
    print("Starting ARIMA model training...")
    
    pipeline = TrainingPipeline(model_type='arima', output_dir='models')
    pipeline.run(
        train_path='data/train_data.csv',
        test_path='data/test_data.csv'
    )
    
    print(f"✓ ARIMA training complete")
    print(f"  RMSE: {pipeline.metrics.get('rmse', 'N/A'):.4f}")
    print(f"  MAE: {pipeline.metrics.get('mae', 'N/A'):.4f}")
    print(f"  R²: {pipeline.metrics.get('r2', 'N/A'):.4f}")
    
    return pipeline.metrics


def train_lstm_task():
    """Train LSTM model."""
    print("Starting LSTM model training...")
    
    pipeline = TrainingPipeline(model_type='lstm', output_dir='models')
    pipeline.run(
        train_path='data/train_data.csv',
        test_path='data/test_data.csv'
    )
    
    print(f"✓ LSTM training complete")
    print(f"  RMSE: {pipeline.metrics.get('rmse', 'N/A'):.4f}")
    print(f"  MAE: {pipeline.metrics.get('mae', 'N/A'):.4f}")
    print(f"  R²: {pipeline.metrics.get('r2', 'N/A'):.4f}")
    
    return pipeline.metrics


def validate_models_task():
    """Validate trained models."""
    print("Starting model validation...")
    
    import os
    import json
    
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        raise ValueError(f"Models directory not found: {models_dir}")
    
    # Check for model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        raise ValueError("No trained models found")
    
    # Check for metrics files
    metrics_files = [f for f in os.listdir(models_dir) if f.startswith('metrics') and f.endswith('.json')]
    
    print(f"✓ Found {len(model_files)} trained models")
    print(f"✓ Found {len(metrics_files)} metrics files")
    
    # Print metrics summary
    for metrics_file in metrics_files:
        with open(os.path.join(models_dir, metrics_file), 'r') as f:
            metrics = json.load(f)
            print(f"  {metrics_file}: RMSE={metrics.get('rmse', 'N/A'):.4f}")
    
    return {"models_found": len(model_files), "metrics_found": len(metrics_files)}


def generate_report_task():
    """Generate a summary report."""
    print("Generating pipeline report...")
    
    import os
    from datetime import datetime
    
    report = []
    report.append("=" * 60)
    report.append("Time-Series Forecasting Pipeline Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    
    # Data summary
    if os.path.exists("data/train_data.csv"):
        import pandas as pd
        train_df = pd.read_csv("data/train_data.csv")
        test_df = pd.read_csv("data/test_data.csv")
        
        report.append("DATA SUMMARY")
        report.append(f"  Training samples: {len(train_df)}")
        report.append(f"  Testing samples: {len(test_df)}")
        report.append(f"  Features: {train_df.shape[1] - 1}")
        report.append("")
    
    # Models summary
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        report.append("TRAINED MODELS")
        report.append(f"  Total models: {len(model_files)}")
        for model_file in sorted(model_files)[-3:]:  # Show last 3 models
            report.append(f"    - {model_file}")
        report.append("")
    
    report.append("=" * 60)
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    os.makedirs("results", exist_ok=True)
    with open("results/pipeline_report.txt", "w") as f:
        f.write(report_text)
    
    print("✓ Report saved to results/pipeline_report.txt")


# Define tasks
task_generate_data = PythonOperator(
    task_id='generate_data',
    python_callable=generate_data_task,
    dag=dag,
)

task_train_arima = PythonOperator(
    task_id='train_arima_model',
    python_callable=train_arima_task,
    dag=dag,
)

task_train_lstm = PythonOperator(
    task_id='train_lstm_model',
    python_callable=train_lstm_task,
    dag=dag,
)

task_validate_models = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models_task,
    dag=dag,
)

task_generate_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report_task,
    dag=dag,
)

# Define task dependencies
# Data generation should run first
# Then both models can train in parallel
# Validation and reporting run after training is complete
task_generate_data >> [task_train_arima, task_train_lstm]
[task_train_arima, task_train_lstm] >> task_validate_models
task_validate_models >> task_generate_report


if __name__ == '__main__':
    print("DAG file loaded successfully")
    print(f"DAG ID: {dag.dag_id}")
    print(f"Start date: {dag.start_date}")
    print(f"Schedule interval: {dag.schedule_interval}")
