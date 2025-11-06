"""
Training script for stress detection using PhysioNet Wearable Device Dataset
Dataset: https://physionet.org/content/wearable-device-dataset/1.0.0/
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("WEARABLE DATASET STRESS DETECTION - Training")
print("=" * 70)

# Configuration
DATASET_PATH = "Dataset/Wearable_Dataset/STRESS"
STRESS_V1_PATH = "Dataset/Stress_Level_v1.csv"
STRESS_V2_PATH = "Dataset/Stress_Level_v2.csv"

def load_csv_signal(filepath):
    """Load Empatica E4 CSV format: first row is timestamp, second is sampling rate"""
    with open(filepath, 'r') as f:
        timestamp = f.readline().strip()
        sample_rate = float(f.readline().strip())
    data = pd.read_csv(filepath, skiprows=2, header=None).values.flatten()
    return data, sample_rate, timestamp

def extract_features_from_signals(subject_folder):
    """Extract statistical features from all physiological signals"""
    features = {}
    
    try:
        # BVP (Blood Volume Pulse) - 64 Hz
        bvp_data, bvp_rate, _ = load_csv_signal(os.path.join(subject_folder, 'BVP.csv'))
        features['bvp_mean'] = np.mean(bvp_data)
        features['bvp_std'] = np.std(bvp_data)
        features['bvp_min'] = np.min(bvp_data)
        features['bvp_max'] = np.max(bvp_data)
        features['bvp_range'] = features['bvp_max'] - features['bvp_min']
        
        # EDA (Electrodermal Activity) - 4 Hz
        eda_data, eda_rate, _ = load_csv_signal(os.path.join(subject_folder, 'EDA.csv'))
        features['eda_mean'] = np.mean(eda_data)
        features['eda_std'] = np.std(eda_data)
        features['eda_min'] = np.min(eda_data)
        features['eda_max'] = np.max(eda_data)
        features['eda_range'] = features['eda_max'] - features['eda_min']
        
        # TEMP (Skin Temperature) - 4 Hz
        temp_data, temp_rate, _ = load_csv_signal(os.path.join(subject_folder, 'TEMP.csv'))
        features['temp_mean'] = np.mean(temp_data)
        features['temp_std'] = np.std(temp_data)
        features['temp_min'] = np.min(temp_data)
        features['temp_max'] = np.max(temp_data)
        
        # HR (Heart Rate) - already computed by device
        hr_data, hr_rate, _ = load_csv_signal(os.path.join(subject_folder, 'HR.csv'))
        features['hr_mean'] = np.mean(hr_data)
        features['hr_std'] = np.std(hr_data)
        features['hr_min'] = np.min(hr_data)
        features['hr_max'] = np.max(hr_data)
        
        # ACC (3-axis Accelerometer) - 32 Hz
        acc_data = pd.read_csv(os.path.join(subject_folder, 'ACC.csv'), skiprows=2, header=None).values
        features['acc_x_mean'] = np.mean(acc_data[:, 0])
        features['acc_y_mean'] = np.mean(acc_data[:, 1])
        features['acc_z_mean'] = np.mean(acc_data[:, 2])
        features['acc_magnitude_mean'] = np.mean(np.sqrt(np.sum(acc_data**2, axis=1)))
        features['acc_magnitude_std'] = np.std(np.sqrt(np.sum(acc_data**2, axis=1)))
        
        return features
    except Exception as e:
        print(f"  Error extracting features: {e}")
        return None

def load_stress_labels():
    """Load self-reported stress levels from both protocol versions"""
    stress_v1 = pd.read_csv(STRESS_V1_PATH, index_col=0)
    stress_v2 = pd.read_csv(STRESS_V2_PATH, index_col=0)
    
    # Combine both versions
    all_stress = pd.concat([stress_v1, stress_v2])
    
    # Calculate average stress level across all tasks for each subject
    # (excluding baseline which is usually lower)
    avg_stress = all_stress.iloc[:, 1:].mean(axis=1)  # Skip baseline column
    
    return avg_stress

print("\n[1/5] Loading dataset...")
print(f"Dataset path: {DATASET_PATH}")

# Load stress labels
stress_labels = load_stress_labels()
print(f"Loaded stress labels for {len(stress_labels)} subjects")
print(f"Stress range: {stress_labels.min():.1f} - {stress_labels.max():.1f}")

# Get all subject folders
subject_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
subject_folders.sort()

print(f"Found {len(subject_folders)} subject folders")

# Extract features for all subjects
all_features = []
all_labels = []
all_subjects = []

print("\n[2/5] Extracting features...")
for subject_id in subject_folders:
    subject_folder = os.path.join(DATASET_PATH, subject_id)
    
    # Handle split data (e.g., f14_a, f14_b)
    subject_base = subject_id.split('_')[0]
    
    if subject_base not in stress_labels.index:
        print(f"  {subject_id} - No stress label, skipping")
        continue
    
    print(f"Processing {subject_id}...", end=" ")
    features = extract_features_from_signals(subject_folder)
    
    if features is not None:
        all_features.append(features)
        all_labels.append(stress_labels[subject_base])
        all_subjects.append(subject_id)
        print("✓")
    else:
        print("✗")

# Convert to DataFrame
df_features = pd.DataFrame(all_features)
y = np.array(all_labels)

print(f"\nDataset: {len(df_features)} subjects, {len(df_features.columns)} features")
print(f"Features: {list(df_features.columns)}")
print(f"\nStress level distribution:")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# Convert to binary classification: Low stress (0-5) vs High stress (5-10)
y_binary = (y > 5).astype(int)
print(f"\nBinary classification (threshold=5):")
print(f"  Low stress (0-5): {np.sum(y_binary == 0)} subjects")
print(f"  High stress (5-10): {np.sum(y_binary == 1)} subjects")

print("\n[3/5] Feature Selection...")

# Use all data for feature selection
X = df_features.values
y = y_binary

# Select best K features using ANOVA F-statistic
n_features = min(10, X.shape[1])  # Select top 10 features (or fewer if less available)
selector = SelectKBest(f_classif, k=n_features)
selector.fit(X, y)

# Get selected feature names
selected_features = df_features.columns[selector.get_support()].tolist()
print(f"Selected {n_features} best features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

# Apply feature selection
X_selected = selector.transform(X)

print("\n[4/5] Preparing data with Cross-Validation...")
print(f"Total samples: {len(X_selected)}")
print(f"Using 5-Fold Stratified Cross-Validation")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Save scaler and selector for prediction
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
print("Scaler saved: scaler.pkl")
print("Feature selector saved: feature_selector.pkl")

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

print("\n[5/5] Training with Cross-Validation...")

def create_model(n_features):
    """Create simplified model with aggressive regularization"""
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.05)),
        layers.Dropout(0.5),
        layers.Dense(4, activation='relu', kernel_regularizer=keras.regularizers.l2(0.05)),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Stratified K-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []
fold_losses = []
best_model = None
best_accuracy = 0

print("\nTraining models with 5-Fold Cross-Validation...")
print("-" * 70)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f"\n📊 Fold {fold}/{n_splits}")
    print("-" * 70)
    
    # Split data
    X_train_fold = X_scaled[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_scaled[val_idx]
    y_val_fold = y[val_idx]
    
    # Data augmentation for training fold only
    X_train_aug = [X_train_fold]
    y_train_aug = [y_train_fold]
    
    for _ in range(1):  # Create 1 additional copy (2x total)
        noise = np.random.normal(0, 0.03, X_train_fold.shape)
        X_train_aug.append(X_train_fold + noise)
        y_train_aug.append(y_train_fold)
    
    X_train_fold = np.vstack(X_train_aug)
    y_train_fold = np.hstack(y_train_aug)
    
    print(f"  Training samples: {len(X_train_fold)} (augmented)")
    print(f"  Validation samples: {len(X_val_fold)}")
    
    # Create and train model
    model = create_model(X_scaled.shape[1])
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=100,
        batch_size=8,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_accuracies.append(val_acc)
    fold_losses.append(val_loss)
    
    print(f"  Validation Accuracy: {val_acc*100:.2f}%")
    print(f"  Validation Loss: {val_loss:.4f}")
    
    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_model = model
        print(f"  ✓ New best model!")

# Final statistics
print("\n" + "=" * 70)
print("Cross-Validation Results:")
print("=" * 70)
print(f"Mean Validation Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
print(f"Mean Validation Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
print(f"Best Fold Accuracy: {best_accuracy*100:.2f}%")

# Train final model on all data
print("\n" + "=" * 70)
print("Training final model on all data...")
print("=" * 70)

# Augment full dataset
X_aug = [X_scaled]
y_aug = [y]

for _ in range(1):  # Create 1 additional copy (2x total)
    noise = np.random.normal(0, 0.03, X_scaled.shape)
    X_aug.append(X_scaled + noise)
    y_aug.append(y)

X_final = np.vstack(X_aug)
y_final = np.hstack(y_aug)

print(f"Training samples: {len(X_final)} (augmented from {len(X_scaled)} original)")

final_model = create_model(X_scaled.shape[1])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=20,
    restore_best_weights=True,
    verbose=0
)

final_model.fit(
    X_final, y_final,
    epochs=100,
    batch_size=8,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# Save final model
final_model.save('stress_model.h5')
print(f"\n✓ Model saved: stress_model.h5")

# Display model architecture
print("\n" + "=" * 70)
print("Final Model Architecture:")
print("=" * 70)
final_model.summary()

print("\n" + "=" * 70)
print("Training complete! Use predict_wearable.py to make predictions")
print("=" * 70)
