"""
Data Preprocessing Module
- Load data from various sources
- Handle missing values
- Encode categorical variables
- Scale features
- Split train/test sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV, Excel, or other formats.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        DataFrame containing the data
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        df = df.fillna(df.mode().iloc[0])
    
    return df


def encode_categorical(df: pd.DataFrame, columns: list = None) -> tuple:
    """
    Encode categorical variables using LabelEncoder.
    
    Args:
        df: Input DataFrame
        columns: List of column names to encode (if None, auto-detect)
        
    Returns:
        Tuple of (encoded DataFrame, dict of label encoders)
    """
    df = df.copy()
    encoders = {}
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    return df, encoders


def scale_features(X_train: np.ndarray, X_test: np.ndarray, 
                    method: str = 'standard') -> tuple:
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
        method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, scaler object)
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, 
                            random_state=random_state, stratify=y)


def preprocess(filepath: str, target_column: str, 
               test_size: float = 0.2, 
               handle_missing: str = 'mean',
               scale: bool = True) -> dict:
    """
    Full preprocessing pipeline.
    
    Args:
        filepath: Path to data file
        target_column: Name of target column
        test_size: Proportion for test set
        handle_missing: Strategy for missing values
        scale: Whether to scale features
        
    Returns:
        Dictionary containing preprocessed data and metadata
    """
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df, strategy=handle_missing)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    
    # Scale features
    scaler = None
    if scale:
        X_train, X_test, scaler = scale_features(X_train.values, X_test.values)
    else:
        X_train, X_test = X_train.values, X_test.values
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler
    }
