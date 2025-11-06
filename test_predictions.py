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
    0.00,     # bvp_mean - Blood Volume Pulse Mean
    5.0,      # bvp_std - Blood Volume Pulse Standard Deviation
    -845.4,   # bvp_min - Blood Volume Pulse Minimum (typical from training)
    5.0,      # bvp_max - Blood Volume Pulse Maximum
    5.0,      # bvp_range - Blood Volume Pulse Range
    2.0,      # eda_mean - Electrodermal Activity Mean
    3.5,      # eda_std - Electrodermal Activity Standard Deviation (higher value → lower stress in this model)
    0.5,      # eda_min - Electrodermal Activity Minimum
    0.5,      # eda_max - Electrodermal Activity Maximum
    0.5,      # eda_range - Electrodermal Activity Range
    31.0,     # temp_mean - Skin Temperature Mean (lower → lower stress)
    0.3,      # temp_std - Skin Temperature Standard Deviation (lower)
    31.1,     # temp_min - Skin Temperature Minimum (typical)
    33.4,     # temp_max - Skin Temperature Maximum
    75,       # hr_mean - Heart Rate Mean
    10,       # hr_std - Heart Rate Standard Deviation
    65.0,     # hr_min - Heart Rate Minimum (higher → lower stress in this model)
    10,       # hr_max - Heart Rate Maximum
    0.0,      # acc_x_mean - Accelerometer X-axis Mean
    0.0,      # acc_y_mean - Accelerometer Y-axis Mean
    42.4,     # acc_z_mean - Accelerometer Z-axis Mean (typical)
    0.0,      # acc_magnitude_mean - Accelerometer Magnitude Mean
    0.3       # acc_magnitude_std - Accelerometer Magnitude Standard Deviation
]

# Example 2: MODERATE STRESS - Mid-range values
moderate_stress_features = [
    0.00,     # bvp_mean - Blood Volume Pulse Mean
    5.0,      # bvp_std - Blood Volume Pulse Standard Deviation
    -845.4,   # bvp_min - Blood Volume Pulse Minimum (typical)
    5.0,      # bvp_max - Blood Volume Pulse Maximum
    5.0,      # bvp_range - Blood Volume Pulse Range
    2.0,      # eda_mean - Electrodermal Activity Mean
    0.8,      # eda_std - Electrodermal Activity Standard Deviation (lower → more stress)
    0.5,      # eda_min - Electrodermal Activity Minimum
    0.5,      # eda_max - Electrodermal Activity Maximum
    0.5,      # eda_range - Electrodermal Activity Range
    33.8,     # temp_mean - Skin Temperature Mean (elevated → more stress)
    0.65,     # temp_std - Skin Temperature Standard Deviation (elevated)
    31.1,     # temp_min - Skin Temperature Minimum (typical)
    33.4,     # temp_max - Skin Temperature Maximum
    75,       # hr_mean - Heart Rate Mean
    10,       # hr_std - Heart Rate Standard Deviation
    53.0,     # hr_min - Heart Rate Minimum (lower → more stress)
    10,       # hr_max - Heart Rate Maximum
    0.0,      # acc_x_mean - Accelerometer X-axis Mean
    -2.0,     # acc_y_mean - Accelerometer Y-axis Mean
    40.0,     # acc_z_mean - Accelerometer Z-axis Mean (lower)
    0.0,      # acc_magnitude_mean - Accelerometer Magnitude Mean
    0.3       # acc_magnitude_std - Accelerometer Magnitude Standard Deviation
]

# Example 3: HIGH STRESS - Based on actual model behavior
# Higher temperature, lower HR_MIN, lower EDA_STD predict higher stress
high_stress_features = [
    0.00,     # bvp_mean - Blood Volume Pulse Mean
    5.0,      # bvp_std - Blood Volume Pulse Standard Deviation
    -845.4,   # bvp_min - Blood Volume Pulse Minimum (typical)
    5.0,      # bvp_max - Blood Volume Pulse Maximum
    5.0,      # bvp_range - Blood Volume Pulse Range
    2.0,      # eda_mean - Electrodermal Activity Mean
    -0.5,     # eda_std - Electrodermal Activity Standard Deviation (lower value → higher stress in this model)
    0.5,      # eda_min - Electrodermal Activity Minimum
    0.5,      # eda_max - Electrodermal Activity Maximum
    0.5,      # eda_range - Electrodermal Activity Range
    35.5,     # temp_mean - Skin Temperature Mean (higher → higher stress)
    0.8,      # temp_std - Skin Temperature Standard Deviation (higher)
    31.1,     # temp_min - Skin Temperature Minimum (typical)
    33.4,     # temp_max - Skin Temperature Maximum
    75,       # hr_mean - Heart Rate Mean
    10,       # hr_std - Heart Rate Standard Deviation
    48.0,     # hr_min - Heart Rate Minimum (lower → higher stress in this model)
    10,       # hr_max - Heart Rate Maximum
    0.0,      # acc_x_mean - Accelerometer X-axis Mean
    -3.0,     # acc_y_mean - Accelerometer Y-axis Mean
    38.0,     # acc_z_mean - Accelerometer Z-axis Mean (lower)
    0.0,      # acc_magnitude_mean - Accelerometer Magnitude Mean
    0.3       # acc_magnitude_std - Accelerometer Magnitude Standard Deviation
]

# Run predictions
predict(low_stress_features, "Example 1: Low Stress (Relaxed)")
predict(moderate_stress_features, "Example 2: Moderate Stress")
predict(high_stress_features, "Example 3: High Stress")

print("\n" + "="*70)
print("Testing Complete!")
print("="*70)
