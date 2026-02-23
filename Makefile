.PHONY: help install setup generate train api test lint format clean docker-up docker-down

# Colors for output
BLUE=\033[0;34m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NC=\033[0m # No Color

help: ## Display this help message
	@echo "$(BLUE)Time-Series Forecasting Pipeline - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup: ## Setup project structure and initialize directories
	@echo "$(BLUE)Setting up project...$(NC)"
	python config/config.py
	mkdir -p logs
	@echo "$(GREEN)✓ Project setup complete$(NC)"

generate: ## Generate synthetic training data
	@echo "$(BLUE)Generating synthetic data...$(NC)"
	python main.py generate --source synthetic --n-points 2000 --n-features 3
	@echo "$(GREEN)✓ Data generation complete$(NC)"

train-arima: ## Train ARIMA model
	@echo "$(BLUE)Training ARIMA model...$(NC)"
	python main.py train --model-type arima
	@echo "$(GREEN)✓ ARIMA training complete$(NC)"

train-lstm: ## Train LSTM model
	@echo "$(BLUE)Training LSTM model...$(NC)"
	python main.py train --model-type lstm
	@echo "$(GREEN)✓ LSTM training complete$(NC)"

train-all: train-arima train-lstm ## Train all models

api: ## Start REST API server
	@echo "$(BLUE)Starting API server...$(NC)"
	python -m src.api

predict: ## Make predictions (requires MODEL_PATH)
	@echo "$(BLUE)Making predictions...$(NC)"
	python main.py predict \
		--model-path $(MODEL_PATH) \
		--input-data data/test_data.csv \
		--output results/predictions.csv
	@echo "$(GREEN)✓ Predictions saved to results/predictions.csv$(NC)"

test: ## Run unit tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=src
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-quick: ## Run quick tests without coverage
	@echo "$(BLUE)Running quick tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

lint: ## Run code linting
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 src/ dags/ main.py config/
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ dags/ main.py config/ tests/
	isort src/ dags/ main.py config/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy src/ dags/ main.py
	@echo "$(GREEN)✓ Type checking complete$(NC)"

quality: lint type-check ## Run all code quality checks

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t timeseries-forecast:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-up: ## Start Docker containers (API + PostgreSQL)
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Containers started$(NC)"
	@echo "  API: http://localhost:5000"
	@echo "  PgAdmin: http://localhost:5050"
	@echo "  Postgres: localhost:5432"

docker-down: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Containers stopped$(NC)"

docker-logs: ## View Docker container logs
	docker-compose logs -f

airflow-init: ## Initialize Airflow
	@echo "$(BLUE)Initializing Airflow...$(NC)"
	airflow db init
	airflow users create --username admin --password admin \
		--firstname Admin --lastname User --role Admin --email admin@example.com
	@echo "$(GREEN)✓ Airflow initialized$(NC)"

airflow-web: ## Start Airflow web server
	@echo "$(BLUE)Starting Airflow webserver...$(NC)"
	airflow webserver --port 8080

airflow-scheduler: ## Start Airflow scheduler
	@echo "$(BLUE)Starting Airflow scheduler...$(NC)"
	airflow scheduler

jupyter: ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	jupyter lab notebooks/

clean-data: ## Remove generated data files
	@echo "$(YELLOW)Removing data files...$(NC)"
	rm -f data/raw_data.csv data/train_data.csv data/test_data.csv data/scaling_params.json
	@echo "$(GREEN)✓ Data cleaned$(NC)"

clean-models: ## Remove trained models
	@echo "$(YELLOW)Removing trained models...$(NC)"
	rm -f models/*.pkl models/*.json
	@echo "$(GREEN)✓ Models cleaned$(NC)"

clean-results: ## Remove result files
	@echo "$(YELLOW)Removing result files...$(NC)"
	rm -f results/*.csv results/*.json results/*.txt
	@echo "$(GREEN)✓ Results cleaned$(NC)"

clean: clean-data clean-models clean-results ## Clean all generated files
	@echo "$(YELLOW)Removing Python cache...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Project cleaned$(NC)"

reset: clean ## Reset project to initial state
	@echo "$(BLUE)Resetting project...$(NC)"
	rm -rf models/ results/ logs/
	mkdir -p models results logs data
	@echo "$(GREEN)✓ Project reset$(NC)"

status: ## Show current project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo ""
	@echo "Data files:"
	@ls -lh data/ 2>/dev/null || echo "  No data files found"
	@echo ""
	@echo "Trained models:"
	@ls -lh models/ 2>/dev/null || echo "  No models found"
	@echo ""
	@echo "Results:"
	@ls -lh results/ 2>/dev/null || echo "  No results found"

quick-start: setup generate train-arima ## Quick start: setup, generate data, train model
	@echo ""
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo "Next steps:"
	@echo "  1. Start API:  make api"
	@echo "  2. Try prediction: curl -X POST http://localhost:5000/health"

.DEFAULT_GOAL := help
