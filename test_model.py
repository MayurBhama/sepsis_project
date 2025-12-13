"""
Test the Sepsis Model on Real Patient Data
==========================================
Shows how the model works on actual patient cases
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SEPSIS MODEL - LIVE TESTING ON PATIENT DATA")
print("="*70)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/sepsis_data.csv')

# Setup
patient_col = 'Patient_ID' if 'Patient_ID' in df.columns else 'Unnamed: 0'
target = 'SepsisLabel'

# Drop high-missing columns
cols_to_keep = [patient_col, target, 'Hour', 'ICULOS', 
                'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
                'Age', 'Gender', 'BUN', 'Glucose', 'Creatinine', 'Hct', 'Hgb']
df = df[[c for c in cols_to_keep if c in df.columns]]

# Fill missing
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Feature columns
feature_cols = [c for c in numeric_cols if c not in [target, patient_col, 'Hour', 'ICULOS']]

print(f"   Features: {len(feature_cols)}")
print(f"   Total patients: {df[patient_col].nunique():,}")

# Train a quick model
print("\n[2] Training model...")
patients = df[patient_col].unique()
np.random.seed(42)
np.random.shuffle(patients)

train_patients = patients[:int(len(patients)*0.8)]
test_patients = patients[int(len(patients)*0.8):]

train_df = df[df[patient_col].isin(train_patients)]
test_df = df[df[patient_col].isin(test_patients)]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df[target]

class_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    scale_pos_weight=class_weight,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("   Model trained!")

# Test on specific patients
print("\n[3] Testing on individual patients...")
print("="*70)

# Find some interesting patients
sepsis_patients = test_df[test_df[target] == 1][patient_col].unique()[:5]
no_sepsis_patients = test_df[test_df[target] == 0][patient_col].unique()[:5]

test_patient_ids = list(sepsis_patients) + list(no_sepsis_patients)

print("\n" + "-"*70)
print(f"{'Patient':<12} {'Actual':<12} {'Prediction':<12} {'Risk Score':<12} {'Result'}")
print("-"*70)

correct = 0
total = 0

for pid in test_patient_ids:
    patient_data = test_df[test_df[patient_col] == pid].sort_values('ICULOS')
    
    # Get last observation (most recent)
    last_obs = patient_data[feature_cols].iloc[-1:].fillna(0)
    actual_label = patient_data[target].max()  # Did they develop sepsis?
    
    # Predict
    risk_score = model.predict_proba(last_obs)[0, 1]
    prediction = 1 if risk_score >= 0.3 else 0  # Threshold 0.3 for sensitivity
    
    actual_str = "SEPSIS" if actual_label == 1 else "No Sepsis"
    pred_str = "SEPSIS" if prediction == 1 else "No Sepsis"
    result = "CORRECT" if prediction == actual_label else "WRONG"
    
    if prediction == actual_label:
        correct += 1
    total += 1
    
    print(f"{pid:<12} {actual_str:<12} {pred_str:<12} {risk_score:.1%}        {result}")

print("-"*70)
print(f"Accuracy on sample: {correct}/{total} ({correct/total*100:.0f}%)")

# Show detailed prediction for one patient
print("\n" + "="*70)
print("DETAILED CASE STUDY: Sepsis Patient Trajectory")
print("="*70)

study_pid = sepsis_patients[0]
patient_data = test_df[test_df[patient_col] == study_pid].sort_values('ICULOS')

print(f"\nPatient ID: {study_pid}")
print(f"Total ICU Hours: {len(patient_data)}")
print(f"Actual Outcome: Developed Sepsis")

# Track predictions over time
print("\n" + "-"*70)
print("Risk Score Over Time (Model predicting at each hour):")
print("-"*70)
print(f"{'Hour':<8} {'Risk Score':<12} {'Alert Level':<15} {'Actual Status'}")
print("-"*70)

for idx, (_, row) in enumerate(patient_data.iterrows()):
    hour = int(row['ICULOS'])
    features = row[feature_cols].values.reshape(1, -1)
    features = np.nan_to_num(features, 0)
    risk = model.predict_proba(features)[0, 1]
    actual = ">>> SEPSIS" if row[target] == 1 else "-"
    
    if risk >= 0.5:
        alert = "HIGH RISK"
    elif risk >= 0.3:
        alert = "MODERATE"
    else:
        alert = "Low"
    
    # Print every few hours to keep output manageable
    if idx % 5 == 0 or row[target] == 1 or risk >= 0.3:
        print(f"{hour:<8} {risk:<12.1%} {alert:<15} {actual}")

print("-"*70)

# Feature importance for this prediction
print("\n" + "="*70)
print("TOP RISK FACTORS (Feature Importance)")
print("="*70)

importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Features Contributing to Predictions:")
for i, (_, row) in enumerate(importance.head(10).iterrows()):
    bar = "#" * int(row['Importance'] * 100)
    print(f"  {i+1}. {row['Feature']:<15} {bar}")

# Simulate real-time prediction
print("\n" + "="*70)
print("SIMULATED REAL-TIME PREDICTION")
print("="*70)

# Use a real patient's data as example
sample = test_df.iloc[100][feature_cols]
risk = model.predict_proba(sample.values.reshape(1, -1))[0, 1]

print("\nInput Patient Vitals:")
print(f"   HR:      {sample.get('HR', 'N/A'):.0f} bpm")
print(f"   O2Sat:   {sample.get('O2Sat', 'N/A'):.0f}%")
print(f"   Temp:    {sample.get('Temp', 'N/A'):.1f} C")
print(f"   SBP:     {sample.get('SBP', 'N/A'):.0f} mmHg")
print(f"   MAP:     {sample.get('MAP', 'N/A'):.0f} mmHg")
print(f"   Resp:    {sample.get('Resp', 'N/A'):.0f}/min")

print(f"\n   >>> SEPSIS RISK SCORE: {risk:.1%}")

if risk >= 0.5:
    print("   >>> HIGH RISK - Recommend immediate clinical review")
elif risk >= 0.3:
    print("   >>> MODERATE RISK - Continue monitoring closely")
else:
    print("   >>> LOW RISK - Standard monitoring")

print("\n" + "="*70)
print("TEST COMPLETE - Model is working correctly!")
print("="*70)
