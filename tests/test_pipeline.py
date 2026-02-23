"""
Unit tests for time-series forecasting pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import TimeSeriesDataGenerator, generate_train_test_split, normalize_data
from src.train import ARIMAModel, LSTMModel, TrainingPipeline


class TestDataGeneration:
    """Test data generation functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create a data generator for testing."""
        return TimeSeriesDataGenerator(seed=42)
    
    def test_synthetic_data_generation(self, generator):
        """Test synthetic data generation."""
        df = generator.generate_synthetic_data(
            n_points=100,
            n_features=3,
            seasonality_period=12,
            trend=True,
            noise_level=0.1
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 100
        assert df.shape[1] == 3
        assert not df.isna().any().any()
    
    def test_train_test_split(self, generator):
        """Test train-test split."""
        df = generator.generate_synthetic_data(n_points=100, n_features=2)
        train_df, test_df = generate_train_test_split(df, test_size=0.2)
        
        assert len(train_df) == 80
        assert len(test_df) == 20
        assert len(train_df) + len(test_df) == len(df)
    
    def test_data_normalization(self, generator):
        """Test data normalization."""
        df = generator.generate_synthetic_data(n_points=100, n_features=2)
        train_df, test_df = generate_train_test_split(df, test_size=0.2)
        
        train_norm, test_norm, mean, std = normalize_data(train_df, test_df)
        
        assert isinstance(train_norm, pd.DataFrame)
        assert isinstance(test_norm, pd.DataFrame)
        assert isinstance(mean, pd.Series)
        assert isinstance(std, pd.Series)
        
        # Check that normalized training data has approximately zero mean
        assert np.allclose(train_norm.mean(), 0, atol=1e-10)
    
    def test_save_and_load_data(self, generator):
        """Test saving and loading data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_data.csv"
            
            # Generate and save
            df = generator.generate_synthetic_data(n_points=50, n_features=2)
            generator.save_data(df, str(filepath))
            
            # Load
            loaded_df = generator.load_data(str(filepath))
            
            # Check
            assert loaded_df.shape == df.shape
            assert loaded_df.shape[0] == 50
            assert loaded_df.shape[1] == 2


class TestModels:
    """Test model implementations."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        generator = TimeSeriesDataGenerator(seed=42)
        df = generator.generate_synthetic_data(n_points=100, n_features=3)
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        return X.iloc[:80], y.iloc[:80], X.iloc[80:], y.iloc[80:]
    
    def test_arima_model(self, sample_data):
        """Test ARIMA model."""
        X_train, y_train, X_test, y_test = sample_data
        
        model = ARIMAModel(alpha=0.3)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
    
    def test_lstm_model(self, sample_data):
        """Test LSTM model."""
        X_train, y_train, X_test, y_test = sample_data
        
        model = LSTMModel(lookback=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
    
    def test_model_save_load(self):
        """Test model persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.pkl"
            
            # Create and save model
            model = ARIMAModel(alpha=0.3)
            model.save(str(filepath))
            
            # Load model
            loaded_model = model.load(str(filepath))
            
            assert type(loaded_model) == type(model)
            assert loaded_model.alpha == 0.3


class TestTrainingPipeline:
    """Test the training pipeline."""
    
    @pytest.fixture
    def sample_data_files(self):
        """Create sample data files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Generate data
            generator = TimeSeriesDataGenerator(seed=42)
            df = generator.generate_synthetic_data(n_points=100, n_features=3)
            
            train_df, test_df = generate_train_test_split(df, test_size=0.2)
            train_norm, test_norm, _, _ = normalize_data(train_df, test_df)
            
            # Save files
            train_path = tmpdir / "train_data.csv"
            test_path = tmpdir / "test_data.csv"
            
            train_norm.to_csv(train_path)
            test_norm.to_csv(test_path)
            
            yield str(train_path), str(test_path), tmpdir
    
    def test_pipeline_arima(self, sample_data_files):
        """Test ARIMA training pipeline."""
        train_path, test_path, tmpdir = sample_data_files
        
        pipeline = TrainingPipeline(model_type='arima', output_dir=str(tmpdir / 'models'))
        pipeline.load_data(train_path, test_path)
        pipeline.preprocess()
        pipeline.train()
        pipeline.evaluate()
        
        assert len(pipeline.metrics) > 0
        assert 'rmse' in pipeline.metrics
        assert 'mae' in pipeline.metrics
    
    def test_pipeline_lstm(self, sample_data_files):
        """Test LSTM training pipeline."""
        train_path, test_path, tmpdir = sample_data_files
        
        pipeline = TrainingPipeline(model_type='lstm', output_dir=str(tmpdir / 'models'))
        pipeline.load_data(train_path, test_path)
        pipeline.preprocess()
        pipeline.train()
        pipeline.evaluate()
        
        assert len(pipeline.metrics) > 0
        assert 'r2' in pipeline.metrics


class TestAPI:
    """Test REST API functionality."""
    
    @pytest.fixture
    def api_client(self):
        """Create a test client for the Flask API."""
        from src.api import app
        
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = api_client.get('/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
    
    def test_list_models(self, api_client):
        """Test list models endpoint."""
        response = api_client.get('/models')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'models' in data
        assert 'count' in data
    
    def test_predict_endpoint(self, api_client):
        """Test prediction endpoint."""
        # This would require mocking the model
        response = api_client.post('/predict', json={
            'data': [[1, 2, 3], [4, 5, 6]]
        })
        
        # Should fail if no model loaded
        assert response.status_code in [404, 400]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
