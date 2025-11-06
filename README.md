# Stress Detection System

A machine learning system that predicts stress levels from physiological signals using wearable device data.

## 📊 Dataset

This project uses the **PhysioNet Wearable Device Dataset** from Induced Stress and Structured Exercise Sessions.

**Download Link:** https://physionet.org/content/wearable-device-dataset/1.0.0/

**Setup:**
1. Download the dataset from the link above (69.7 MB ZIP file)
2. Extract it to the `Dataset/` folder in this project directory
3. The structure should be: `Dataset/Wearable_Dataset/STRESS/`, `Dataset/Wearable_Dataset/AEROBIC/`, etc.

**Dataset Details:**
- 36 subjects with stress recordings
- Empatica E4 wearable device signals: BVP, EDA, TEMP, ACC, HR, IBI
- Self-reported stress levels (1-10 scale)
- CSV format - easy to process

**Citation:**
```
Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). 
Wearable Device Dataset from Induced Stress and Structured Exercise Sessions (version 1.0.0). 
PhysioNet. https://doi.org/10.13026/zzf8-xv61
```

## 📁 Project Structure

```
Watch/
├── Dataset/              # Download dataset here (see above)
├── train_wearable.py     # Training script
├── predict_wearable.py   # Prediction interface
├── stress_model.h5       # Trained model (generated)
├── scaler.pkl            # Feature scaler (generated)
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy
```

### 2. Download Dataset

Download the PhysioNet Wearable Device Dataset from the link above and extract it to the `Dataset/` folder.

### 3. Train the Model

```bash
python train_wearable.py
```

This will:
- Load 37 stress recording sessions from Dataset/Wearable_Dataset/STRESS/
- Extract 23 physiological features from BVP, EDA, TEMP, HR, and ACC signals
- Train a neural network with data augmentation and class balancing
- Save the trained model and scaler

### 4. Make Predictions

```bash
python predict_wearable.py
```

The script will prompt you to enter 23 physiological features extracted from wearable device signals.

## 📊 Model Details

- **Input Features:** 10 selected features (from 23 extracted)
  - Selected via ANOVA F-statistic: bvp_mean, bvp_min, eda_std, temp_mean, temp_std, temp_min, temp_max, hr_min, acc_y_mean, acc_z_mean
- **Architecture:** Simplified dense network (8→4→1 neurons)
  - Only 129 trainable parameters
  - L2 regularization (lambda=0.05)
  - Dropout (50%)
- **Training:** 100 epochs with 5-fold stratified cross-validation
- **Dataset:** 37 subjects with self-reported stress levels (1-10 scale)
- **Classification:** Binary - Low stress (≤5) vs High stress (>5)

## 💡 Extracted Features

The model uses statistical features from Empatica E4 wearable device signals:

| Signal Type | Features | Description |
|-------------|----------|-------------|
| **BVP** | mean, std, min, max, range | Blood volume pulse (64 Hz) |
| **EDA** | mean, std, min, max, range | Electrodermal activity (4 Hz) |
| **TEMP** | mean, std, min, max | Skin temperature in °C (4 Hz) |
| **HR** | mean, std, min, max | Heart rate in bpm (derived from BVP) |
| **ACC** | x_mean, y_mean, z_mean, mag_mean, mag_std | 3-axis accelerometer (32 Hz) |

## 📈 Training Results

### Cross-Validation Performance
- **Mean Validation Accuracy:** 70.36% ± 14.97%
- **Mean Validation Loss:** 0.8314 ± 0.1243
- **Best Fold Accuracy:** 87.50%
- **Cross-Validation:** 5-fold stratified
- **Total Samples:** 37 subjects (28 low stress, 9 high stress)

### Per-Fold Results
| Fold | Validation Accuracy | Validation Loss | Status |
|------|-------------------|-----------------|---------|
| 1    | 87.50%            | 0.7719          | ✓ Best |
| 2    | 50.00%            | 0.7656          |         |
| 3    | 57.14%            | 0.7407          |         |
| 4    | 71.43%            | 0.8019          |         |
| 5    | 85.71%            | 1.0771          |         |

### Model Improvements Applied
✅ **Feature Selection:** Reduced from 23 → 10 features using ANOVA F-test  
✅ **Simpler Architecture:** 8→4→1 neurons (129 trainable params vs 1,441 before)  
✅ **Aggressive Regularization:** L2=0.05, Dropout=50%  
✅ **Cross-Validation:** Robust 5-fold evaluation instead of single split  
✅ **Conservative Augmentation:** 2x samples with ±3% noise

## 🔧 Technical Details

**Training Techniques:**
- **Feature Selection:** SelectKBest with ANOVA F-statistic (top 10 features)
- **Data Augmentation:** 2x samples with ±3% Gaussian noise
- **Class Weight Balancing:** Addresses 28:9 imbalance (weights: 0.66 stress, 2.06 baseline)
- **L2 Regularization:** Aggressive lambda=0.05 to prevent overfitting
- **Dropout:** 50% rate for strong regularization
- **Early Stopping:** Patience of 15-20 epochs
- **Cross-Validation:** Stratified 5-fold for robust evaluation
- **Optimizer:** Adam with learning rate 0.001
- **Batch Size:** 8 (small batches for small dataset)

**Selected Features (Top 10):**
1. `bvp_mean` - Blood volume pulse average
2. `bvp_min` - Minimum BVP value
3. `eda_std` - Electrodermal activity variability
4. `temp_mean` - Average skin temperature
5. `temp_std` - Temperature variability
6. `temp_min` - Minimum temperature
7. `temp_max` - Maximum temperature
8. `hr_min` - Minimum heart rate
9. `acc_y_mean` - Y-axis acceleration
10. `acc_z_mean` - Z-axis acceleration

**Stress Protocol:**
The dataset includes various stress-inducing tasks:
- Math challenges (TMCT - Trier Mental Challenge Test)
- Stroop Test (v1 protocol only)
- Opinion/debate tasks (expressing controversial views)
- Countdown subtraction tasks
- Rest periods with relaxing videos

## 💡 Example Predictions

The model classifies stress as LOW (≤50% probability) or HIGH (>50% probability):

```
Example Subject A (Baseline):
  Features: [bvp=0.5, bvp_min=-5, eda_std=0.3, temp=32.5, ...]
  → Prediction: 25% → LOW STRESS 😌

Example Subject B (Stressed):
  Features: [bvp=2.1, bvp_min=-2, eda_std=0.8, temp=33.2, ...]
  → Prediction: 85% → HIGH STRESS 😰
```

**Note:** Actual predictions depend on all 10 selected features. The model achieves best performance when input features are within the training data distribution.

## 📝 License

This project uses the PhysioNet Wearable Device Dataset under the Open Data Commons Attribution License v1.0.

## 🎯 Model Performance Summary

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 70.36% ± 14.97% |
| **Best Fold** | 87.50% |
| **Model Size** | 129 parameters (1.52 KB) |
| **Features** | 10 selected from 23 extracted |
| **Training Time** | ~2 minutes on CPU |
| **Subjects** | 37 (28 baseline, 9 stressed) |

**Key Strengths:**
- ✅ Robust cross-validation evaluation
- ✅ Very lightweight model (only 129 parameters)
- ✅ Automatic feature selection
- ✅ Class-balanced training
- ✅ Strong regularization prevents overfitting

**Limitations:**
- ⚠️ Small dataset (37 subjects)
- ⚠️ Class imbalance (76% baseline, 24% stressed)
- ⚠️ High variance across folds (50-87% accuracy range)
- ⚠️ Performance may vary with new subjects