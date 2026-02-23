"""
Data generation script for time-series forecasting.
Generates synthetic or loads real time-series data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json


class TimeSeriesDataGenerator:
    """Generate synthetic time-series data for forecasting tasks."""
    
    def __init__(self, seed=42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def generate_synthetic_data(self, 
                                n_points=1000,
                                n_features=3,
                                seasonality_period=24,
                                trend=True,
                                noise_level=0.1):
        """
        Generate synthetic time-series data with trend and seasonality.
        
        Args:
            n_points: Number of time points
            n_features: Number of features
            seasonality_period: Period of seasonality
            trend: Whether to add trend
            noise_level: Standard deviation of noise
            
        Returns:
            DataFrame with time-series data
        """
        # Time index
        start_date = datetime(2020, 1, 1)
        time_index = [start_date + timedelta(hours=i) for i in range(n_points)]
        
        data = {}
        
        for feature_idx in range(n_features):
            # Create base time series
            t = np.arange(n_points)
            
            # Trend component
            if trend:
                trend_component = 0.01 * t * (feature_idx + 1)
            else:
                trend_component = 0
            
            # Seasonality component
            seasonality = 10 * np.sin(2 * np.pi * t / seasonality_period) * (feature_idx + 1)
            
            # Noise
            noise = np.random.normal(0, noise_level, n_points)
            
            # Combine components
            series = 50 * (feature_idx + 1) + trend_component + seasonality + noise
            data[f'feature_{feature_idx}'] = series
        
        df = pd.DataFrame(data, index=time_index)
        return df
    
    def load_data(self, filepath):
        """
        Load time-series data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with time-series data
        """
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    
    def save_data(self, df, filepath):
        """
        Save time-series data to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")


def generate_train_test_split(df, test_size=0.2):
    """
    Split time-series data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of test data
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def normalize_data(train_df, test_df=None):
    """
    Normalize data using training set statistics.
    
    Args:
        train_df: Training data
        test_df: Testing data (optional)
        
    Returns:
        Normalized dataframes and scaling parameters
    """
    train_mean = train_df.mean()
    train_std = train_df.std()
    
    train_normalized = (train_df - train_mean) / train_std
    
    if test_df is not None:
        test_normalized = (test_df - train_mean) / train_std
        return train_normalized, test_normalized, train_mean, train_std
    
    return train_normalized, train_mean, train_std


def main():
    """Generate and save time-series data."""
    # Create output directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
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
    
    # Save normalization parameters
    scaling_params = {
        "mean": mean.to_dict(),
        "std": std.to_dict()
    }
    with open("data/scaling_params.json", "w") as f:
        json.dump(scaling_params, f, indent=2)
    
    print(f"Generated data with shape: {df.shape}")
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")


if __name__ == "__main__":
    main()
