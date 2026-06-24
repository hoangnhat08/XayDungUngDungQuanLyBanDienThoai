"""
Model Evaluation Module
- Evaluate model performance
- Generate various metrics
- Create visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score,
    MAE, MSE, RMSE
)
import os


def evaluate_classification(y_true, y_pred, y_prob=None, class_names=None):
    """
    Comprehensive evaluation for classification models.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for ROC curve)
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Classification Report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    
    # ROC Curve (for binary classification)
    if y_prob is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
    
    return metrics


def evaluate_regression(y_true, y_pred):
    """
    Comprehensive evaluation for regression models.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    # MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    if mask.any():
        metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return metrics


def print_metrics(metrics, task: str = 'classification'):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        task: 'classification' or 'regression'
    """
    print("\n" + "="*50)
    print(f"Model Evaluation ({task})")
    print("="*50)
    
    if task == 'classification':
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
    else:
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"MSE:  {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²:   {metrics['r2']:.4f}")
        if 'mape' in metrics:
            print(f"MAPE: {metrics['mape']:.2f}%")
    
    print("="*50)


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_regression_results(y_true, y_pred, save_path=None):
    """
    Plot regression results (actual vs predicted, residual plot).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Actual vs Predicted')
    
    # Residual plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regression plot saved to: {save_path}")
    
    plt.show()


def evaluate_model(model, X_test, y_test, task='classification', 
                   y_prob=None, save_dir=None, class_names=None):
    """
    Complete model evaluation with optional visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        task: 'classification' or 'regression'
        y_prob: Prediction probabilities (for ROC curve)
        save_dir: Directory to save plots
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    if task == 'classification':
        metrics = evaluate_classification(y_test, y_pred, y_prob, class_names)
    else:
        metrics = evaluate_regression(y_test, y_pred)
    
    # Print metrics
    print_metrics(metrics, task)
    
    # Create plots
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        if task == 'classification':
            plot_confusion_matrix(metrics['confusion_matrix'], class_names,
                                  os.path.join(save_dir, 'confusion_matrix.png'))
            
            if 'roc_curve' in metrics:
                plot_roc_curve(
                    metrics['roc_curve']['fpr'],
                    metrics['roc_curve']['tpr'],
                    metrics['roc_curve']['auc'],
                    os.path.join(save_dir, 'roc_curve.png')
                )
        else:
            plot_regression_results(y_test, y_pred,
                                    os.path.join(save_dir, 'regression_results.png'))
    
    return metrics
