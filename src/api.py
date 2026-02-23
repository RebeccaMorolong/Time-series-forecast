"""
REST API for time-series forecasting models.
Provides endpoints for making predictions and managing models.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model storage
models = {}
model_metadata = {}


class ModelManager:
    """Manage loaded models."""
    
    @staticmethod
    def load_model(model_path):
        """Load a pickle model from disk."""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            return None
    
    @staticmethod
    def get_latest_model(models_dir='models'):
        """Get the most recently trained model."""
        models_dir = Path(models_dir)
        model_files = sorted(models_dir.glob('model_*.pkl'))
        
        if not model_files:
            return None
        
        return model_files[-1]


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'loaded_models': list(models.keys())
    }), 200


@app.route('/models', methods=['GET'])
def list_models():
    """List all available models."""
    model_info = []
    for name, model in models.items():
        info = {
            'name': name,
            'type': type(model).__name__,
            'loaded_at': model_metadata.get(name, {}).get('loaded_at')
        }
        model_info.append(info)
    
    return jsonify({
        'models': model_info,
        'count': len(model_info)
    }), 200


@app.route('/models/load', methods=['POST'])
def load_model():
    """Load a model from disk."""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        model_name = data.get('model_name', Path(model_path).stem)
        
        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400
        
        model = ModelManager.load_model(model_path)
        
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 404
        
        models[model_name] = model
        model_metadata[model_name] = {
            'loaded_at': datetime.utcnow().isoformat(),
            'path': model_path
        }
        
        return jsonify({
            'message': f'Model {model_name} loaded successfully',
            'model_name': model_name,
            'model_type': type(model).__name__
        }), 200
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/models/load-latest', methods=['POST'])
def load_latest_model():
    """Load the latest trained model."""
    try:
        data = request.get_json() if request.is_json else {}
        models_dir = data.get('models_dir', 'models')
        model_name = data.get('model_name', 'latest')
        
        latest_model_path = ModelManager.get_latest_model(models_dir)
        
        if latest_model_path is None:
            return jsonify({'error': 'No models found'}), 404
        
        model = ModelManager.load_model(str(latest_model_path))
        
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 404
        
        models[model_name] = model
        model_metadata[model_name] = {
            'loaded_at': datetime.utcnow().isoformat(),
            'path': str(latest_model_path)
        }
        
        return jsonify({
            'message': f'Latest model loaded as {model_name}',
            'model_name': model_name,
            'model_path': str(latest_model_path),
            'model_type': type(model).__name__
        }), 200
    
    except Exception as e:
        logger.error(f"Error loading latest model: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the loaded model."""
    try:
        data = request.get_json()
        
        # Get model name (default to 'latest')
        model_name = data.get('model_name', 'latest')
        
        if model_name not in models:
            return jsonify({
                'error': f'Model {model_name} not loaded. Available models: {list(models.keys())}'
            }), 404
        
        model = models[model_name]
        
        # Get input data
        input_data = data.get('data')
        
        if not input_data:
            return jsonify({'error': 'input data is required'}), 400
        
        # Convert to appropriate format
        if isinstance(input_data, list):
            X = np.array(input_data)
        else:
            return jsonify({'error': 'input data must be a list or array'}), 400
        
        # Make predictions
        predictions = model.predict(X)
        
        return jsonify({
            'model_name': model_name,
            'predictions': predictions.tolist(),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict-series', methods=['POST'])
def predict_series():
    """Make predictions for a time series."""
    try:
        data = request.get_json()
        
        model_name = data.get('model_name', 'latest')
        
        if model_name not in models:
            return jsonify({
                'error': f'Model {model_name} not loaded. Available models: {list(models.keys())}'
            }), 404
        
        model = models[model_name]
        
        # Get input data as DataFrame
        input_data = data.get('data')
        
        if not input_data:
            return jsonify({'error': 'data is required'}), 400
        
        # Convert to DataFrame
        if isinstance(input_data, list):
            X = pd.DataFrame(input_data)
        elif isinstance(input_data, dict):
            X = pd.DataFrame(input_data)
        else:
            return jsonify({'error': 'data must be a list or dict'}), 400
        
        # Make predictions
        predictions = model.predict(X)
        
        return jsonify({
            'model_name': model_name,
            'input_shape': list(X.shape),
            'predictions': predictions.tolist(),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error making series prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about a specific model."""
    model_name = request.args.get('model_name', 'latest')
    
    if model_name not in models:
        return jsonify({
            'error': f'Model {model_name} not found. Available models: {list(models.keys())}'
        }), 404
    
    model = models[model_name]
    metadata = model_metadata.get(model_name, {})
    
    return jsonify({
        'model_name': model_name,
        'model_type': type(model).__name__,
        'loaded_at': metadata.get('loaded_at'),
        'path': metadata.get('path')
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


def run_api(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask API server."""
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_api(host='0.0.0.0', port=5000, debug=True)
