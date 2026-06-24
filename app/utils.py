"""
Utility Functions
- Helper functions used across the project
- File operations
- Data visualization helpers
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List
import numpy as np


def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'log_{timestamp}.txt')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_json(data: Dict, filepath: str):
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yaml(data: Dict, filepath: str):
    """
    Save data to YAML file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"YAML saved to: {filepath}")


def load_yaml(filepath: str) -> Dict:
    """
    Load data from YAML file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_directories(dirs: List[str]):
    """
    Create multiple directories if they don't exist.
    
    Args:
        dirs: List of directory paths
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_metrics(y_true, y_pred, task: str = 'classification') -> Dict[str, float]:
    """
    Calculate common metrics for evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task: 'classification' or 'regression'
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_absolute_error, mean_squared_error, r2_score
    )
    
    if task == 'classification':
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    else:
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }


def ensure_dir(filepath: str):
    """
    Ensure directory exists for a file path.
    
    Args:
        filepath: File path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str):
        """Load configuration from file."""
        if config_path.endswith('.json'):
            self.config = load_json(config_path)
        elif config_path.endswith(('.yaml', '.yml')):
            self.config = load_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value == default:
                break
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def save(self, config_path: str):
        """Save configuration to file."""
        if config_path.endswith('.json'):
            save_json(self.config, config_path)
        elif config_path.endswith(('.yaml', '.yml')):
            save_yaml(self.config, config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
