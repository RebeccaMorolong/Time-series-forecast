# Quick Start Guide

Get the time-series forecasting pipeline up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- pip or conda
- Git

## Step 1: Clone and Setup (2 min)

```bash
# Clone the repository
git clone <your-repo-url>
cd Time-series-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup project directories
python config/config.py
```

Or use the Makefile:
```bash
make install setup
```

## Step 2: Generate Data (1 min)

Generate synthetic time-series data:

```bash
python main.py generate --source synthetic --n-points 2000 --n-features 3
```

Or use Makefile:
```bash
make generate
```

**Output:**
- `data/raw_data.csv` - Original synthetic data
- `data/train_data.csv` - Normalized training set
- `data/test_data.csv` - Normalized test set
- `data/scaling_params.json` - Normalization parameters

## Step 3: Train Models (1 min)

Train forecasting models:

```bash
# Train ARIMA model
python main.py train --model-type arima

# Train LSTM model
python main.py train --model-type lstm
```

Or train all models at once:
```bash
make train-all
```

**Output:**
- `models/model_ARIMA_*.pkl` - Trained ARIMA model
- `models/model_LSTM_*.pkl` - Trained LSTM model
- `models/metrics_*.json` - Performance metrics

View metrics:
```bash
cat models/metrics_*.json
```

## Step 4: Use the API (1 min)

Start the Flask API:

```bash
python -m src.api
```

Or:
```bash
make api
```

The API runs on `http://localhost:5000`

### Test the API

In a new terminal:

```bash
# Health check
curl http://localhost:5000/health

# Load latest model
curl -X POST http://localhost:5000/models/load-latest \
  -H "Content-Type: application/json"

# List models
curl http://localhost:5000/models

# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "latest",
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  }'
```

## Complete Workflow in One Command

```bash
# Fastest way to get started
make quick-start
```

This will:
1. Setup project structure
2. Generate synthetic data
3. Train ARIMA model

Then:
```bash
make api
```

## Run Tests

```bash
make test
```

## Docker Setup

For isolated environment with PostgreSQL:

```bash
# Build and start containers
docker-compose up -d

# API available at: http://localhost:5000
# PgAdmin available at: http://localhost:5050
# Postgres: localhost:5432
```

## Using Airflow for Orchestration

```bash
# Initialize Airflow
make airflow-init

# Start scheduler in one terminal
make airflow-scheduler

# Start webserver in another terminal
make airflow-web

# Access at http://localhost:8080 (admin/admin)
```

## Project Structure

```
Time-series-forecast/
├── data/              # Training and test data
├── models/            # Trained models
├── src/               # Source code
│   ├── data_generation.py    # Data generation
│   ├── train.py              # Training pipeline
│   ├── api.py                # REST API
│   └── models/               # Model implementations
├── dags/              # Airflow DAGs
├── config/            # Configuration
├── main.py            # CLI entry point
└── Makefile           # Useful commands
```

## Available Commands

```bash
# Development
make install          # Install dependencies
make setup           # Setup directories
make generate        # Generate data
make train-all       # Train all models
make test            # Run tests
make lint            # Code quality checks

# Running
make api             # Start API server
make predict         # Make predictions
make jupyter         # Start Jupyter Lab

# Cleaning
make clean           # Clean all generated files
make reset           # Reset to initial state
```

See `Makefile` for all available commands.

## Troubleshooting

### Missing dependencies
```bash
pip install -r requirements.txt
```

### Data not found
```bash
python main.py generate --source synthetic
```

### API port already in use
```bash
# Change port in src/api.py or config
python -m src.api --port 5001
```

### Permission errors on macOS/Linux
```bash
source venv/bin/activate
```

## Next Steps

1. **Explore notebooks**: `jupyter lab notebooks/`
2. **Review full documentation**: `README_PIPELINE.md`
3. **Modify configuration**: `config/config.py`
4. **Add custom models**: `src/models/`
5. **Deploy with Docker**: See `docker-compose.yml`

## Getting Help

- Check logs: `tail -f logs/pipeline.log`
- Review configuration: `cat config/config.py`
- Run tests: `make test`
- Check documentation: `README_PIPELINE.md`

## Common Workflows

### Experiment with Different Data
```bash
python main.py generate --n-points 5000 --n-features 5 --noise-level 1.0
make train-all
```

### Make Predictions
```bash
python main.py predict \
  --model-path models/model_ARIMA_*.pkl \
  --input-data data/test_data.csv \
  --output predictions.csv
```

### Run Pipeline with Airflow
```bash
make airflow-init
make airflow-scheduler &
make airflow-web
# Trigger DAG in http://localhost:8080
```

### Deploy with Docker
```bash
docker-compose up -d
curl http://localhost:5000/health
```

Happy Forecasting! 🚀
