"""
Prediction script for stress detection using physiological signals
Based on PhysioNet Wearable Device Dataset trained model
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STRESS DETECTION SYSTEM - Wearable Device Model")
print("=" * 70)

# Load model and scaler
print("\nLoading trained model...")

try:
    model = keras.models.load_model('stress_model.h5')
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please run train_wearable.py to train the model first.")
    exit(1)

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    exit(1)

try:
    with open('feature_selector.pkl', 'rb') as f:
        feature_selector = pickle.load(f)
    print(f"✓ Feature selector loaded successfully")
    selected_indices = feature_selector.get_support()
    n_selected = sum(selected_indices)
    print(f"Model uses {n_selected} selected features")
except Exception as e:
    print(f"❌ Error loading feature selector: {e}")
    exit(1)

# Display feature information
print("\n" + "=" * 70)
print("Required Input Features (23 total):")
print("=" * 70)
print("""
Blood Volume Pulse (BVP):
  • Mean, Std Dev, Min, Max, Range

Electrodermal Activity (EDA):
  • Mean, Std Dev, Min, Max, Range

Skin Temperature (TEMP):
  • Mean, Std Dev, Min, Max

Heart Rate (HR):
  • Mean, Std Dev, Min, Max

Accelerometer (ACC):
  • X-axis Mean, Y-axis Mean, Z-axis Mean
  • Magnitude Mean, Magnitude Std Dev
""")
print("=" * 70)
print(f"\nNote: Model uses {n_selected} best features selected during training")
print("=" * 70)

def predict_stress(features):
    """
    Predict stress level from physiological features
    
    Parameters:
    -----------
    features : list or array
        23 physiological features in order:
        [bvp_mean, bvp_std, bvp_min, bvp_max, bvp_range,
         eda_mean, eda_std, eda_min, eda_max, eda_range,
         temp_mean, temp_std, temp_min, temp_max,
         hr_mean, hr_std, hr_min, hr_max,
         acc_x_mean, acc_y_mean, acc_z_mean,
         acc_magnitude_mean, acc_magnitude_std]
    
    Returns:
    --------
    probability : float
        Probability of high stress (0-100%)
    classification : str
        "LOW" or "HIGH" stress
    """
    # Reshape features
    features_array = np.array(features).reshape(1, -1)
    
    # Apply feature selection
    features_selected = feature_selector.transform(features_array)
    
    # Scale features
    features_scaled = scaler.transform(features_selected)
    
    # Predict
    prediction = model.predict(features_scaled, verbose=0)[0][0]
    probability = prediction * 100

    # Classify into three levels for user clarity
    if probability < 45:
        classification = "LOW"
        emoji = "😌"
        message = "You're in a relaxed state"
    elif probability <= 55:
        classification = "MODERATE"
        emoji = "😐"
        message = "Borderline stress — consider a short break"
    else:
        classification = "HIGH"
        emoji = "😰"
        message = "Elevated stress detected - take a break"
    
    return probability, classification, emoji, message

# Interactive mode
print("\n📊 Interactive Stress Prediction")
print("=" * 70)
print("\nExample values (from typical wearable data):")
print("  BVP: mean=0.5, std=2.5, min=-5, max=5, range=10")
print("  EDA: mean=2.0, std=0.5, min=1.0, max=3.0, range=2.0")
print("  TEMP: mean=32.5, std=0.3, min=32.0, max=33.0")
print("  HR: mean=75, std=8, min=65, max=90")
print("  ACC: x=0, y=0, z=0, mag_mean=0.5, mag_std=0.2")
print("\n" + "-" * 70)

while True:
    print("\nEnter 'q' to quit or press Enter to input features:")
    user_input = input().strip()
    
    if user_input.lower() == 'q':
        break
    
    try:
        print("\nEnter physiological features:")
        
        # BVP features
        bvp_mean = float(input("  BVP Mean: "))
        bvp_std = float(input("  BVP Std Dev: "))
        bvp_min = float(input("  BVP Min: "))
        bvp_max = float(input("  BVP Max: "))
        bvp_range = float(input("  BVP Range: "))
        
        # EDA features
        eda_mean = float(input("  EDA Mean: "))
        eda_std = float(input("  EDA Std Dev: "))
        eda_min = float(input("  EDA Min: "))
        eda_max = float(input("  EDA Max: "))
        eda_range = float(input("  EDA Range: "))
        
        # TEMP features
        temp_mean = float(input("  TEMP Mean: "))
        temp_std = float(input("  TEMP Std Dev: "))
        temp_min = float(input("  TEMP Min: "))
        temp_max = float(input("  TEMP Max: "))
        
        # HR features
        hr_mean = float(input("  HR Mean: "))
        hr_std = float(input("  HR Std Dev: "))
        hr_min = float(input("  HR Min: "))
        hr_max = float(input("  HR Max: "))
        
        # ACC features
        acc_x = float(input("  ACC X-axis Mean: "))
        acc_y = float(input("  ACC Y-axis Mean: "))
        acc_z = float(input("  ACC Z-axis Mean: "))
        acc_mag_mean = float(input("  ACC Magnitude Mean: "))
        acc_mag_std = float(input("  ACC Magnitude Std Dev: "))
        
        # Combine features
        features = [
            bvp_mean, bvp_std, bvp_min, bvp_max, bvp_range,
            eda_mean, eda_std, eda_min, eda_max, eda_range,
            temp_mean, temp_std, temp_min, temp_max,
            hr_mean, hr_std, hr_min, hr_max,
            acc_x, acc_y, acc_z, acc_mag_mean, acc_mag_std
        ]
        
        # Get prediction
        probability, classification, emoji, message = predict_stress(features)
        
        # Display results
        print("\n" + "=" * 70)
        print(f"{emoji} STRESS ASSESSMENT RESULTS")
        print("=" * 70)
        print(f"Stress Probability: {probability:.1f}%")
        print(f"Classification: {classification} STRESS")
        print(f"Assessment: {message}")
        print("=" * 70)
        
    except ValueError as e:
        print(f"\n❌ Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

print("\n" + "=" * 70)
print("Thank you for using the Stress Detection System!")
print("=" * 70)
