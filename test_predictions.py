"""
Test predictions with example data
"""
import numpy as np
import pickle
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Load model and preprocessors
model = keras.models.load_model('stress_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_selector.pkl', 'rb') as f:
    feature_selector = pickle.load(f)

def predict(features, label):
    """Make prediction and display result"""
    features_array = np.array(features).reshape(1, -1)
    features_selected = feature_selector.transform(features_array)
    features_scaled = scaler.transform(features_selected)
    prediction = model.predict(features_scaled, verbose=0)[0][0]
    probability = prediction * 100
    classification = "HIGH" if probability > 50 else "LOW"
    emoji = "😰" if probability > 50 else "😌"
    
    print(f"\n{label}")
    print(f"  → Probability: {probability:.1f}%")
    print(f"  → Classification: {classification} STRESS {emoji}")
    return probability, classification

print("="*70)
print("Testing Stress Detection Model with Example Data")
print("="*70)

# Example 1: Baseline/Relaxed state (low HR, normal temp, low EDA)
baseline_features = [
    0.5,   # BVP mean (low)
    2.0,   # BVP std
    -5.0,  # BVP min
    5.0,   # BVP max
    10.0,  # BVP range
    1.5,   # EDA mean (low - relaxed)
    0.2,   # EDA std (low variability)
    1.2,   # EDA min
    2.0,   # EDA max
    0.8,   # EDA range (small)
    31.5,  # TEMP mean (lower - relaxed)
    0.2,   # TEMP std
    31.0,  # TEMP min
    32.0,  # TEMP max
    68,    # HR mean (low - relaxed)
    5,     # HR std
    60,    # HR min (low)
    75,    # HR max
    -0.05, # ACC x
    0.02,  # ACC y
    -0.01, # ACC z
    0.2,   # ACC magnitude mean (low movement)
    0.1    # ACC magnitude std
]

# Example 2: Moderate stress (elevated HR, higher EDA)
moderate_features = [
    1.2,   # BVP mean (elevated)
    2.8,   # BVP std
    -4.0,  # BVP min
    6.0,   # BVP max
    10.0,  # BVP range
    2.5,   # EDA mean (elevated)
    0.5,   # EDA std (more variable)
    1.8,   # EDA min
    3.5,   # EDA max
    1.7,   # EDA range (larger)
    32.8,  # TEMP mean (elevated)
    0.4,   # TEMP std
    32.2,  # TEMP min
    33.5,  # TEMP max
    82,    # HR mean (elevated)
    9,     # HR std
    70,    # HR min
    95,    # HR max
    -0.1,  # ACC x
    0.05,  # ACC y
    0.03,  # ACC z
    0.4,   # ACC magnitude mean
    0.2    # ACC magnitude std
]

# Example 3: High stress (high HR, high EDA, elevated temp)
stress_features = [
    2.0,   # BVP mean (high)
    3.5,   # BVP std
    -2.0,  # BVP min
    8.0,   # BVP max
    10.0,  # BVP range
    3.5,   # EDA mean (high - stressed)
    0.8,   # EDA std (very variable)
    2.2,   # EDA min
    5.0,   # EDA max
    2.8,   # EDA range (very large)
    33.5,  # TEMP mean (high - stressed)
    0.5,   # TEMP std
    32.8,  # TEMP min
    34.5,  # TEMP max
    95,    # HR mean (high - stressed)
    12,    # HR std
    78,    # HR min
    110,   # HR max
    -0.15, # ACC x
    0.08,  # ACC y
    0.05,  # ACC z
    0.6,   # ACC magnitude mean (more movement)
    0.3    # ACC magnitude std
]

# Run predictions
predict(baseline_features, "Example 1: Baseline/Relaxed State")
predict(moderate_features, "Example 2: Moderate Stress")
predict(stress_features, "Example 3: High Stress")

print("\n" + "="*70)
print("Testing Complete!")
print("="*70)
