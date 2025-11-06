"""
Test predictions with example data
"""
import numpy as np
import pickle
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Load model and preprocessors
print("Loading model artifacts...")
model = keras.models.load_model('stress_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_selector.pkl', 'rb') as f:
    feature_selector = pickle.load(f)
print("✓ Model, scaler, and feature selector loaded successfully")

def predict(features, label):
    """Make prediction and display result"""
    features_array = np.array(features).reshape(1, -1)
    features_selected = feature_selector.transform(features_array)
    features_scaled = scaler.transform(features_selected)
    prediction = model.predict(features_scaled, verbose=0)[0][0]
    probability = prediction * 100
    
    # Classify with moderate range
    if probability > 55:
        classification = "HIGH"
        emoji = "😰"
    elif probability > 45:
        classification = "MODERATE"
        emoji = "😐"
    else:
        classification = "LOW"
        emoji = "😌"
    
    print(f"\n{label}")
    print(f"  → Probability: {probability:.1f}%")
    print(f"  → Classification: {classification} STRESS {emoji}")
    return probability, classification

print("="*70)
print("Testing Stress Detection Model with Example Data")
print("="*70)

# Example 1: LOW STRESS - Based on actual model training data
# Lower temperature, higher HR_MIN values predict lower stress
low_stress_features = [
    0.00,     # BVP mean (typical)
    5.0,      # BVP std
    -845.4,   # BVP min (typical from training)
    5.0,      # BVP max
    5.0,      # BVP range
    2.0,      # EDA mean
    3.5,      # EDA std (higher value → lower stress in this model)
    0.5,      # EDA min
    0.5,      # EDA max
    0.5,      # EDA range
    31.0,     # TEMP mean (lower → lower stress)
    0.3,      # TEMP std (lower)
    31.1,     # TEMP min (typical)
    33.4,     # TEMP max
    75,       # HR mean
    10,       # HR std
    65.0,     # HR min (higher → lower stress in this model)
    10,       # HR max
    0.0,      # ACC x mean
    0.0,      # ACC y mean
    42.4,     # ACC z mean (typical)
    0.0,      # ACC magnitude mean
    0.3       # ACC magnitude std
]

# Example 2: MODERATE STRESS - Mid-range values
moderate_stress_features = [
    0.00,     # BVP mean
    5.0,      # BVP std
    -845.4,   # BVP min (typical)
    5.0,      # BVP max
    5.0,      # BVP range
    2.0,      # EDA mean
    0.8,      # EDA std (lower → more stress)
    0.5,      # EDA min
    0.5,      # EDA max
    0.5,      # EDA range
    33.8,     # TEMP mean (elevated → more stress)
    0.65,     # TEMP std (elevated)
    31.1,     # TEMP min (typical)
    33.4,     # TEMP max
    75,       # HR mean
    10,       # HR std
    53.0,     # HR min (lower → more stress)
    10,       # HR max
    0.0,      # ACC x mean
    -2.0,     # ACC y mean
    40.0,     # ACC z mean (lower)
    0.0,      # ACC magnitude mean
    0.3       # ACC magnitude std
]

# Example 3: HIGH STRESS - Based on actual model behavior
# Higher temperature, lower HR_MIN, lower EDA_STD predict higher stress
high_stress_features = [
    0.00,     # BVP mean
    5.0,      # BVP std
    -845.4,   # BVP min (typical)
    5.0,      # BVP max
    5.0,      # BVP range
    2.0,      # EDA mean
    -0.5,     # EDA std (lower value → higher stress in this model)
    0.5,      # EDA min
    0.5,      # EDA max
    0.5,      # EDA range
    35.5,     # TEMP mean (higher → higher stress)
    0.8,      # TEMP std (higher)
    31.1,     # TEMP min (typical)
    33.4,     # TEMP max
    75,       # HR mean
    10,       # HR std
    48.0,     # HR min (lower → higher stress in this model)
    10,       # HR max
    0.0,      # ACC x mean
    -3.0,     # ACC y mean
    38.0,     # ACC z mean (lower)
    0.0,      # ACC magnitude mean
    0.3       # ACC magnitude std
]

# Run predictions
predict(low_stress_features, "Example 1: Low Stress (Relaxed)")
predict(moderate_stress_features, "Example 2: Moderate Stress")
predict(high_stress_features, "Example 3: High Stress")

print("\n" + "="*70)
print("Testing Complete!")
print("="*70)
