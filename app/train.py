"""
Model Training Module
- Train various ML models
- Hyperparameter tuning
- Save trained models
"""

import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


# Classification models
CLASSIFICATION_MODELS = {
    'logistic_regression': LogisticRegression,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'gradient_boosting': GradientBoostingClassifier,
    'svm': SVC
}

# Regression models
REGRESSION_MODELS = {
    'linear_regression': LinearRegression,
    'decision_tree': DecisionTreeRegressor,
    'random_forest': RandomForestRegressor,
    'svr': SVR
}


def get_model(model_name: str, task: str = 'classification', **kwargs):
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model
        task: 'classification' or 'regression'
        **kwargs: Model hyperparameters
        
    Returns:
        Model instance
    """
    if task == 'classification':
        models = CLASSIFICATION_MODELS
    else:
        models = REGRESSION_MODELS
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def train_model(X_train, y_train, model_name: str = 'random_forest',
                task: str = 'classification', **model_params):
    """
    Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model to train
        task: 'classification' or 'regression'
        **model_params: Additional model parameters
        
    Returns:
        Trained model
    """
    model = get_model(model_name, task, **model_params)
    model.fit(X_train, y_train)
    return model


def train_multiple_models(X_train, y_train, model_names: list = None,
                          task: str = 'classification', **common_params):
    """
    Train multiple models and compare performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_names: List of model names to train (default: all available)
        task: 'classification' or 'regression'
        **common_params: Common model parameters
        
    Returns:
        Dictionary of trained models
    """
    if model_names is None:
        model_names = list(CLASSIFICATION_MODELS.keys()) if task == 'classification' \
                      else list(REGRESSION_MODELS.keys())
    
    trained_models = {}
    
    for name in model_names:
        print(f"Training {name}...")
        try:
            model = train_model(X_train, y_train, name, task, **common_params)
            trained_models[name] = model
            print(f"  ✓ {name} trained successfully")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
    
    return trained_models


def save_model(model, filepath: str):
    """
    Save trained model to file.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def load_saved_model(filepath: str):
    """
    Load a trained model from file.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def quick_train(X_train, y_train, X_test, y_test, 
                model_name: str = 'random_forest',
                task: str = 'classification'):
    """
    Quick training with immediate evaluation.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Model to use
        task: 'classification' or 'regression'
        
    Returns:
        Trained model and metrics
    """
    model = train_model(X_train, y_train, model_name, task)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    if task == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred)
        }
    else:
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    
    return model, metrics
