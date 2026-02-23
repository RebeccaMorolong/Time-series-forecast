# Time-Series Forecasting Pipeline

A comprehensive end-to-end machine learning pipeline for time-series forecasting with multiple models, REST API, and Apache Airflow orchestration.

## Project Structure

```
├── data/                          # Data storage
│   ├── raw_data.csv              # Original raw data
│   ├── train_data.csv            # Normalized training data
│   ├── test_data.csv             # Normalized test data
│   └── scaling_params.json       # Normalization parameters
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_generation.py        # Data generation and loading
│   ├── train.py                  # Model training pipeline
│   ├── api.py                    # Flask REST API
│   ├── models/                   # Model implementations
│   ├── preprocessing/            # Data preprocessing utilities
│   ├── utils/                    # Utility functions
│   └── evaluation/               # Evaluation metrics
│
├── dags/                          # Airflow DAGs
│   └── time_series_dag.py        # Main orchestration DAG
│
├── notebooks/                     # Jupyter notebooks
│   └── (explorations and analysis)
│
├── config/                        # Configuration files
│   └── config.py                 # Pipeline configuration
│
├── models/                        # Trained model storage
│   ├── model_*.pkl               # Trained models
│   └── metrics_*.json            # Performance metrics
│
├── results/                       # Pipeline results
│   ├── predictions/              # Model predictions
│   └── evaluation_reports/       # Evaluation reports
│
├── tests/                         # Unit tests
│   └── (test files)
│
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Features

### 1. **Data Generation** (`src/data_generation.py`)
- Generate synthetic time-series data with configurable:
  - Number of time points
  - Number of features
  - Seasonality period
  - Trend components
  - Noise levels
- Load real time-series data from CSV files
- Automatic train-test splitting
- Data normalization with scaling parameters

### 2. **Model Training** (`src/train.py`)
- **ARIMA Model**: Exponential smoothing-based forecasting
- **LSTM Model**: Deep learning-based sequential forecasting
- Training pipeline with:
  - Automatic preprocessing
  - Model fitting
  - Evaluation on test set
  - Metrics calculation (RMSE, MAE, R²)
  - Model persistence (pickle)

### 3. **REST API** (`src/api.py`)
Flask-based REST API with endpoints:
- `GET /health` - Health check
- `GET /models` - List loaded models
- `POST /models/load` - Load a specific model
- `POST /models/load-latest` - Load latest trained model
- `POST /predict` - Make predictions with numpy arrays
- `POST /predict-series` - Make predictions with time-series data
- `GET /model-info` - Get model metadata

### 4. **Orchestration** (`dags/time_series_dag.py`)
Apache Airflow DAG that:
1. Generates training data
2. Trains ARIMA and LSTM models in parallel
3. Validates trained models
4. Generates summary reports
- Scheduled to run daily
- Includes error handling and retry logic
- Email notifications on failure

### 5. **CLI Interface** (`main.py`)
Command-line tool with subcommands:
```bash
# Generate data
python main.py generate --source synthetic --n-points 2000 --n-features 3

# Train model
python main.py train --model-type arima --train-data data/train_data.csv

# Make predictions
python main.py predict --model-path models/model_ARIMA_*.pkl --input-data data/test_data.csv
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Time-series-forecast
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize directories**
```bash
python config/config.py
```

## Usage

### Quick Start

1. **Generate synthetic data**
```bash
python main.py generate --source synthetic --n-points 2000 --n-features 3
```

2. **Train models**
```bash
python main.py train --model-type arima
python main.py train --model-type lstm
```

3. **Make predictions**
```bash
python main.py predict --model-path models/model_ARIMA_*.pkl --input-data data/test_data.csv --output predictions.csv
```

### Run with Airflow

1. **Initialize Airflow**
```bash
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

2. **Copy DAG to Airflow home**
```bash
cp dags/time_series_dag.py ~/airflow/dags/
```

3. **Start Airflow services**
```bash
airflow webserver --port 8080
airflow scheduler
```

Visit `http://localhost:8080` and trigger the `time_series_forecasting_pipeline` DAG.

### Start REST API

```bash
python -m src.api
# API runs on http://localhost:5000
```

**Example API calls:**
```bash
# Load latest model
curl -X POST http://localhost:5000/models/load-latest \
  -H "Content-Type: application/json"

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "latest",
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  }'

# Get model info
curl http://localhost:5000/model-info?model_name=latest
```

## Configuration

Edit `config/config.py` to customize:
- Data generation parameters
- Model training settings
- API server settings
- Airflow scheduler settings
- Evaluation metrics
- Database connections

## Model Performance

After training, metrics are saved in `models/metrics_*.json`:
```json
{
  "mse": 0.0234,
  "rmse": 0.1529,
  "mae": 0.0987,
  "r2": 0.9456
}
```

## Development

### Running Tests
```bash
pytest tests/ -v
pytest tests/ --cov=src
```

### Code Quality
```bash
# Format code
black src/ dags/ main.py

# Lint code
flake8 src/ dags/ main.py

# Type checking
mypy src/ dags/ main.py

# Sort imports
isort src/ dags/ main.py
```

### Jupyter Notebooks
```bash
jupyter lab notebooks/
```

## Extending the Pipeline

### Add New Models
1. Create new model class in `src/models/`
2. Inherit from `TimeSeriesModel`
3. Implement `fit()` and `predict()` methods
4. Register in `TrainingPipeline`

### Add New Preprocessing Steps
1. Create functions in `src/preprocessing/`
2. Update `TrainingPipeline.preprocess()`
3. Document parameters in `config.py`

### Add New API Endpoints
1. Create route in `src/api.py`
2. Use appropriate Flask decorators
3. Add error handling
4. Document in README

## Monitoring and Logging

Logs are stored in:
- Console output (INFO level and above)
- `logs/pipeline.log` (all levels)
- Airflow logs in `$AIRFLOW_HOME/logs/`

Enable debug logging in `config.py`:
```python
LOGGING_CONFIG['root']['level'] = 'DEBUG'
```

## Database Integration

Configure database in `config.py`:
```python
DATABASE_CONFIG = {
    'engine': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'timeseries_db',
    'user': 'postgres',
    'password': 'password',
}
```

## Deployment

### Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "src.api"]
```

Build and run:
```bash
docker build -t timeseries-forecast .
docker run -p 5000:5000 timeseries-forecast
```

### Kubernetes
See `kubernetes/` directory for deployment manifests.

## Contributing

1. Create a feature branch
2. Make changes and commit
3. Submit pull request
4. Ensure all tests pass

## License

[Your License Here]

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## References

- [Statsmodels Time Series](https://www.statsmodels.org/stable/tsa.html)
- [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Apache Airflow](https://airflow.apache.org/)
- [Flask REST API](https://flask-restful.readthedocs.io/)
