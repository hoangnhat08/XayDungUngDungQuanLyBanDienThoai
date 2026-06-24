"""
Demo Inference Script
Quick demonstration of model inference on sample data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from app.preprocess import load_data, scale_features, encode_categorical
from app.train import load_saved_model
from app.inference import predict_single, predict_batch


def demo_with_sample_data():
    """
    Demo: Make predictions on sample/test data.
    """
    print("=" * 60)
    print("DEMO: Inference on Sample Data")
    print("=" * 60)
    
    # Example: Mobile Price Classification
    # Features: battery_power, blue, clock_speed, dual_sim, fc, four_g, 
    #           int_memory, m_dep, mobile_wt, n_cores, pc, px_height, 
    #           px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi
    
    sample_data = np.array([
        [842, 0, 2.2, 0, 1, 0, 7, 0.6, 188, 2, 2, 20, 756, 2549, 9, 7, 19, 0, 0, 1],  # Sample 1
        [1021, 1, 0.5, 1, 6, 1, 53, 0.9, 136, 3, 6, 905, 1988, 2631, 17, 3, 14, 1, 1, 1],  # Sample 2
        [1021, 1, 0.5, 1, 6, 1, 53, 0.9, 136, 3, 6, 905, 1988, 2631, 17, 3, 14, 1, 1, 1],  # Sample 3
    ])
    
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Features: {sample_data.shape[1]}")
    
    # If you have a trained model:
    # model_path = 'models/trained_model.pkl'
    # if os.path.exists(model_path):
    #     model = load_saved_model(model_path)
    #     predictions = predict_batch(model, sample_data)
    #     print(f"\nPredictions: {predictions}")
    # else:
    print("\n[INFO] No trained model found. Showing expected output format.")
    print("To run inference:")
    print("1. Train a model first: python app/train.py")
    print("2. Update model_path in this script")
    print("3. Run: python demo/demo_inference.py")
    
    # Example output format
    print("\nExpected prediction format (Mobile Price):")
    print("  0: Low Price (0-10000)")
    print("  1: Medium Price (10001-20000)")
    print("  2: High Price (20001-30000)")
    print("  3: Very High Price (>30000)")
    
    return sample_data


def demo_from_csv():
    """
    Demo: Load data from CSV and make predictions.
    """
    print("\n" + "=" * 60)
    print("DEMO: Inference from CSV File")
    print("=" * 60)
    
    # Check if sample data exists
    sample_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.csv')
    
    if os.path.exists(sample_csv):
        print(f"\nLoading sample data from: {sample_csv}")
        df = pd.read_csv(sample_csv)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        # Make predictions
        # model = load_saved_model('models/trained_model.pkl')
        # predictions = predict_batch(model, df.values)
        # print(f"Predictions: {predictions}")
    else:
        print(f"\n[INFO] Sample CSV not found at: {sample_csv}")
        print("Create a sample.csv file in the data/ folder to test.")
    
    return None


def demo_api_style():
    """
    Demo: API-style prediction function.
    """
    print("\n" + "=" * 60)
    print("DEMO: API-Style Prediction")
    print("=" * 60)
    
    # Example usage of API-style inference
    print("""
# Import and setup
from app.inference import create_prediction_api
from app.train import load_saved_model

# Load model
model = load_saved_model('models/trained_model.pkl')

# Create API
predict_fn = create_prediction_api(
    model, 
    scaler=None,
    class_labels=['class_0', 'class_1', 'class_2']
)

# Make prediction
result = predict_fn([1.5, 2.3, 0.8, ...])  # Your features
print(result)
# Output: {'prediction': 1, 'label': 'class_1'}
""")


def create_sample_predictions():
    """
    Create sample predictions for demonstration.
    """
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Simulate predictions for demonstration
    sample_predictions = np.array([1, 0, 2, 1, 3, 0, 1, 2])
    sample_probabilities = np.array([
        [0.1, 0.7, 0.1, 0.1],
        [0.8, 0.1, 0.05, 0.05],
        [0.05, 0.05, 0.85, 0.05],
        [0.1, 0.6, 0.2, 0.1],
        [0.05, 0.05, 0.1, 0.8],
        [0.9, 0.05, 0.03, 0.02],
        [0.15, 0.7, 0.1, 0.05],
        [0.05, 0.1, 0.75, 0.1],
    ])
    
    print("\nSimulated Predictions:")
    print(f"  Class predictions: {sample_predictions}")
    print(f"\nPrediction probabilities:")
    
    for i, (pred, probs) in enumerate(zip(sample_predictions, sample_probabilities)):
        print(f"  Sample {i+1}: Class {pred} | Confidence: {max(probs):.2%}")
    
    return sample_predictions, sample_probabilities


def main():
    """
    Main demo function.
    """
    print("\n" + "=" * 60)
    print("        MACHINE LEARNING INFERENCE DEMO")
    print("=" * 60)
    
    # Run all demos
    demo_with_sample_data()
    demo_from_csv()
    demo_api_style()
    create_sample_predictions()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
Next steps:
1. Place your trained model at: models/trained_model.pkl
2. Update feature columns in demo_inference.py to match your dataset
3. Run: python demo/demo_inference.py
4. Or use Jupyter notebook: jupyter notebook demo/demo.ipynb
""")


if __name__ == "__main__":
    main()
