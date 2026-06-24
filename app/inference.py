"""
Inference Module
- Make predictions on new data
- Batch prediction
- API-like interface
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Union, List


def load_model(model_path: str):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model (.pkl file)
        
    Returns:
        Loaded model
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_single(model, X: np.ndarray, scaler=None) -> np.ndarray:
    """
    Make prediction on a single sample or batch.
    
    Args:
        model: Trained model
        X: Input features (1D array for single, 2D for batch)
        scaler: Scaler used during training (optional)
        
    Returns:
        Predictions
    """
    if scaler is not None:
        X = scaler.transform(X.reshape(1, -1)) if X.ndim == 1 else scaler.transform(X)
    else:
        X = X.reshape(1, -1) if X.ndim == 1 else X
    
    return model.predict(X)


def predict_batch(model, X: np.ndarray, scaler=None, batch_size: int = 32) -> np.ndarray:
    """
    Make predictions on large batches with memory efficiency.
    
    Args:
        model: Trained model
        X: Input features (2D array)
        scaler: Scaler used during training (optional)
        batch_size: Batch size for prediction
        
    Returns:
        All predictions
    """
    n_samples = X.shape[0]
    predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch = X[i:i+batch_size]
        if scaler is not None:
            batch = scaler.transform(batch)
        pred = model.predict(batch)
        predictions.append(pred)
    
    return np.concatenate(predictions)


def predict_from_file(model_path: str, data_path: str, 
                       scaler_path: str = None, 
                       output_path: str = None) -> np.ndarray:
    """
    Make predictions from a data file.
    
    Args:
        model_path: Path to the saved model
        data_path: Path to input data (CSV)
        scaler_path: Path to the scaler (optional)
        output_path: Path to save predictions (optional)
        
    Returns:
        Predictions array
    """
    # Load model
    model = load_model(model_path)
    
    # Load scaler if provided
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.values
    
    # Predict
    predictions = predict_batch(model, X, scaler)
    
    # Save if output path provided
    if output_path:
        pd.DataFrame({'prediction': predictions}).to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
    
    return predictions


def create_prediction_api(model, scaler=None, class_labels: list = None):
    """
    Create a simple API-like interface for predictions.
    
    Args:
        model: Trained model
        scaler: Scaler used during training
        class_labels: Labels for classification (optional)
        
    Returns:
        Prediction function
    """
    def predict_api(X):
        """
        Make prediction.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with prediction results
        """
        pred = predict_single(model, np.array(X), scaler)[0]
        
        result = {'prediction': int(pred) if isinstance(pred, (np.integer,)) else float(pred)}
        
        if class_labels is not None:
            result['label'] = class_labels[int(pred)]
        
        return result
    
    return predict_api


def compare_models(models: dict, X: np.ndarray, scaler=None) -> pd.DataFrame:
    """
    Compare predictions from multiple models.
    
    Args:
        models: Dictionary of {model_name: model}
        X: Input features
        scaler: Scaler used during training
        
    Returns:
        DataFrame with predictions from all models
    """
    results = {}
    
    for name, model in models.items():
        preds = predict_batch(model, X, scaler)
        results[name] = preds
    
    return pd.DataFrame(results)


def main():
    """Demo usage of inference module."""
    # Example usage
    print("Inference Module")
    print("=" * 50)
    print("""
Usage examples:
    
1. Single prediction:
    model = load_model('models/my_model.pkl')
    prediction = predict_single(model, X_new, scaler)
    
2. Batch prediction:
    predictions = predict_batch(model, X_batch, scaler)
    
3. From file:
    preds = predict_from_file('models/my_model.pkl', 'data/test.csv')
    
4. API-like:
    predict_fn = create_prediction_api(model, scaler, class_labels=['cat', 'dog'])
    result = predict_fn([1.5, 2.3, 0.8])
    """)


if __name__ == "__main__":
    main()
