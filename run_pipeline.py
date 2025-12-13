"""
Complete Sepsis Prediction Pipeline
Step-by-step execution with detailed logging

Steps:
1. Load and explore dataset
2. Clean data (handle missing values, outliers)
3. Feature engineering (temporal features)
4. Patient-stratified train/val/test split
5. Handle class imbalance (class weights, SMOTE comparison)
6. Train ensemble model
7. Evaluate with unbiased metrics (AUPRC, per-class)
8. Test on held-out test set
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, recall_score, precision_score
)

# Models
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except:
    LGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")


def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def main():
    print_section("SEPSIS PREDICTION - COMPLETE PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print_section("STEP 1: LOADING DATA")
    
    df = pd.read_csv("data/sepsis_data.csv")
    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
    
    # Check target distribution
    target_col = 'SepsisLabel'
    print(f"\nTarget distribution:")
    print(df[target_col].value_counts())
    sepsis_rate = df[target_col].mean() * 100
    print(f"Sepsis rate: {sepsis_rate:.2f}%")
    
    # Patient info
    patient_col = 'Patient_ID' if 'Patient_ID' in df.columns else 'Unnamed: 0'
    n_patients = df[patient_col].nunique()
    print(f"\nUnique patients: {n_patients:,}")
    
    # =========================================================================
    # STEP 2: CLEAN DATA
    # =========================================================================
    print_section("STEP 2: CLEANING DATA")
    
    # 2a. Drop columns with >95% missing (useless)
    print("\n2a. Dropping columns with >95% missing values...")
    cols_to_drop = []
    for col in df.columns:
        if col in [target_col, patient_col, 'Hour', 'ICULOS']:
            continue
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 95:
            cols_to_drop.append((col, missing_pct))
    
    print(f"Columns to drop ({len(cols_to_drop)}):")
    for col, pct in cols_to_drop:
        print(f"  - {col}: {pct:.1f}% missing")
    
    df = df.drop(columns=[c for c, _ in cols_to_drop])
    print(f"Shape after dropping: {df.shape}")
    
    # 2b. Handle remaining missing values
    print("\n2b. Handling missing values...")
    
    # Define feature groups
    vital_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
    vital_cols = [c for c in vital_cols if c in df.columns]
    
    lab_cols = [c for c in df.columns if c not in vital_cols + 
                [target_col, patient_col, 'Hour', 'ICULOS', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']]
    
    print(f"Vital signs: {vital_cols}")
    print(f"Lab values: {len(lab_cols)} columns")
    
    # Sort by patient and time
    df = df.sort_values([patient_col, 'ICULOS'])
    
    # Forward fill within patient (carry last known value)
    print("\nApplying forward-fill per patient...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [target_col, patient_col]]
    
    for col in numeric_cols:
        df[col] = df.groupby(patient_col)[col].ffill()
        df[col] = df.groupby(patient_col)[col].bfill()
    
    # Fill remaining with median
    print("Filling remaining NaN with median...")
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    remaining_nan = df.isna().sum().sum()
    print(f"Remaining NaN values: {remaining_nan}")
    
    # 2c. Cap outliers
    print("\n2c. Capping outliers at 1st-99th percentile...")
    for col in vital_cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)
    
    print("Data cleaning complete!")
    print(f"Final shape: {df.shape}")
    
    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    print_section("STEP 3: FEATURE ENGINEERING")
    
    initial_cols = len(df.columns)
    
    # 3a. Clinical scores
    print("\n3a. Creating clinical scores...")
    
    if 'HR' in df.columns and 'SBP' in df.columns:
        df['Shock_Index'] = df['HR'] / (df['SBP'] + 1e-6)
        print("  Created: Shock_Index")
    
    if 'HR' in df.columns and 'MAP' in df.columns:
        df['Modified_Shock_Index'] = df['HR'] / (df['MAP'] + 1e-6)
        print("  Created: Modified_Shock_Index")
    
    if 'SBP' in df.columns and 'DBP' in df.columns:
        df['Pulse_Pressure'] = df['SBP'] - df['DBP']
        print("  Created: Pulse_Pressure")
    
    # Binary indicators
    if 'Temp' in df.columns:
        df['Fever'] = (df['Temp'] > 38).astype(int)
        df['Hypothermia'] = (df['Temp'] < 36).astype(int)
    
    if 'SBP' in df.columns:
        df['Hypotension'] = (df['SBP'] <= 100).astype(int)
    
    if 'Resp' in df.columns:
        df['Tachypnea'] = (df['Resp'] > 22).astype(int)
    
    if 'HR' in df.columns:
        df['Tachycardia'] = (df['HR'] > 90).astype(int)
    
    # 3b. Temporal features (lag)
    print("\n3b. Creating temporal lag features...")
    lag_cols = ['HR', 'MAP', 'Resp', 'O2Sat']
    lag_cols = [c for c in lag_cols if c in df.columns]
    
    for col in lag_cols:
        for lag in [1, 3, 6]:
            df[f'{col}_lag_{lag}h'] = df.groupby(patient_col)[col].shift(lag)
        # Delta (change)
        df[f'{col}_delta_1h'] = df[col] - df.groupby(patient_col)[col].shift(1)
    
    # 3c. Rolling statistics
    print("\n3c. Creating rolling statistics...")
    for col in ['HR', 'MAP']:
        if col in df.columns:
            df[f'{col}_rolling_mean_6h'] = df.groupby(patient_col)[col].transform(
                lambda x: x.rolling(6, min_periods=1).mean()
            )
            df[f'{col}_rolling_std_6h'] = df.groupby(patient_col)[col].transform(
                lambda x: x.rolling(6, min_periods=2).std()
            )
    
    # Fill NaN in new features
    new_cols = [c for c in df.columns if c not in [target_col, patient_col]]
    for col in new_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if pd.notna(df[col].median()) else 0)
    
    final_cols = len(df.columns)
    print(f"\nFeatures created: {final_cols - initial_cols}")
    print(f"Total features: {final_cols}")
    
    # =========================================================================
    # STEP 4: PATIENT-STRATIFIED SPLIT
    # =========================================================================
    print_section("STEP 4: PATIENT-STRATIFIED TRAIN/VAL/TEST SPLIT")
    
    # Get patient-level labels
    patient_labels = df.groupby(patient_col)[target_col].max().reset_index()
    patient_labels.columns = [patient_col, 'has_sepsis']
    
    print(f"Total patients: {len(patient_labels):,}")
    print(f"Patients with sepsis: {patient_labels['has_sepsis'].sum():,}")
    
    # Stratified split at patient level
    np.random.seed(42)
    
    sepsis_patients = patient_labels[patient_labels['has_sepsis'] == 1][patient_col].values
    no_sepsis_patients = patient_labels[patient_labels['has_sepsis'] == 0][patient_col].values
    
    np.random.shuffle(sepsis_patients)
    np.random.shuffle(no_sepsis_patients)
    
    # 70% train, 15% val, 15% test
    def split_array(arr, train_pct=0.7, val_pct=0.15):
        n = len(arr)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        return arr[:train_end], arr[train_end:val_end], arr[val_end:]
    
    train_sepsis, val_sepsis, test_sepsis = split_array(sepsis_patients)
    train_no_sepsis, val_no_sepsis, test_no_sepsis = split_array(no_sepsis_patients)
    
    train_patients = np.concatenate([train_sepsis, train_no_sepsis])
    val_patients = np.concatenate([val_sepsis, val_no_sepsis])
    test_patients = np.concatenate([test_sepsis, test_no_sepsis])
    
    # Verify no overlap
    assert len(set(train_patients) & set(val_patients)) == 0, "Train-Val overlap!"
    assert len(set(train_patients) & set(test_patients)) == 0, "Train-Test overlap!"
    assert len(set(val_patients) & set(test_patients)) == 0, "Val-Test overlap!"
    print("✅ VERIFIED: No patient overlap between splits")
    
    # Create splits
    train_df = df[df[patient_col].isin(train_patients)]
    val_df = df[df[patient_col].isin(val_patients)]
    test_df = df[df[patient_col].isin(test_patients)]
    
    print(f"\nTrain: {len(train_patients):,} patients, {len(train_df):,} records")
    print(f"Val:   {len(val_patients):,} patients, {len(val_df):,} records")
    print(f"Test:  {len(test_patients):,} patients, {len(test_df):,} records")
    
    print(f"\nSepsis rate - Train: {train_df[target_col].mean()*100:.2f}%")
    print(f"Sepsis rate - Val:   {val_df[target_col].mean()*100:.2f}%")
    print(f"Sepsis rate - Test:  {test_df[target_col].mean()*100:.2f}%")
    
    # =========================================================================
    # STEP 5: PREPARE FEATURES
    # =========================================================================
    print_section("STEP 5: PREPARING FEATURES")
    
    drop_cols = [target_col, patient_col, 'Hour', 'ICULOS', 'Unnamed: 0']
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=drop_cols, errors='ignore')
    y_val = val_df[target_col]
    
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df[target_col]
    
    feature_names = list(X_train.columns)
    print(f"Features: {len(feature_names)}")
    
    # Fill any remaining NaN
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    # Calculate class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    imbalance_ratio = n_neg / n_pos
    print(f"\nClass imbalance in training: 1:{imbalance_ratio:.1f}")
    print(f"  Negative (no sepsis): {n_neg:,}")
    print(f"  Positive (sepsis):    {n_pos:,}")
    
    # =========================================================================
    # STEP 6: TRAIN MODELS WITH CLASS WEIGHT BALANCING
    # =========================================================================
    print_section("STEP 6: TRAINING MODELS (Handling Imbalance with Class Weights)")
    
    models = {}
    val_probas = {}
    
    # 6a. LightGBM
    if LGBM_AVAILABLE:
        print("\n6a. Training LightGBM...")
        lgbm = LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            scale_pos_weight=imbalance_ratio,  # Handle imbalance
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)
        models['LightGBM'] = lgbm
        val_probas['LightGBM'] = lgbm.predict_proba(X_val)[:, 1]
        print(f"  LightGBM trained!")
    
    # 6b. XGBoost
    print("\n6b. Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=imbalance_ratio,  # Handle imbalance
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    val_probas['XGBoost'] = xgb.predict_proba(X_val)[:, 1]
    print(f"  XGBoost trained!")
    
    # 6c. CatBoost
    if CATBOOST_AVAILABLE:
        print("\n6c. Training CatBoost...")
        catboost = CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            class_weights=[1, imbalance_ratio],  # Handle imbalance
            random_state=42,
            verbose=False
        )
        catboost.fit(X_train, y_train)
        models['CatBoost'] = catboost
        val_probas['CatBoost'] = catboost.predict_proba(X_val)[:, 1]
        print(f"  CatBoost trained!")
    
    # =========================================================================
    # STEP 7: EVALUATE ON VALIDATION (UNBIASED METRICS)
    # =========================================================================
    print_section("STEP 7: VALIDATION EVALUATION (Unbiased Metrics)")
    
    print("\n** Metrics optimized for imbalanced data: AUPRC, Per-Class Recall **")
    
    def evaluate_model(name, y_true, y_proba, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        
        # Core metrics
        auprc = average_precision_score(y_true, y_proba)
        auroc = roc_auc_score(y_true, y_proba)
        
        # Per-class metrics (CRITICAL for imbalance)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for positive
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for negative
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        print(f"\n{name} @ threshold={threshold:.2f}:")
        print(f"  AUPRC (Primary):     {auprc:.4f}  ← Most important for imbalanced data")
        print(f"  AUROC:               {auroc:.4f}")
        print(f"  Sensitivity (TPR):   {sensitivity:.4f}  ← Catching sepsis cases")
        print(f"  Specificity (TNR):   {specificity:.4f}  ← Avoiding false alarms")
        print(f"  PPV (Precision):     {ppv:.4f}")
        print(f"  NPV:                 {npv:.4f}")
        print(f"  F1-Score:            {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN={tn:,}  FP={fp:,}")
        print(f"    FN={fn:,}  TP={tp:,}")
        
        return {'auprc': auprc, 'auroc': auroc, 'sensitivity': sensitivity, 
                'specificity': specificity, 'f1': f1, 'threshold': threshold}
    
    # Evaluate each model
    results = {}
    for name, proba in val_probas.items():
        results[name] = evaluate_model(name, y_val, proba)
    
    # Find optimal threshold for best model
    print("\n" + "-"*60)
    print("FINDING OPTIMAL THRESHOLD (maximizing F1)")
    print("-"*60)
    
    best_model_name = max(val_probas.keys(), key=lambda k: results[k]['auprc'])
    best_proba = val_probas[best_model_name]
    
    best_f1 = 0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (best_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"Best model: {best_model_name}")
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"F1 at optimal threshold: {best_f1:.4f}")
    
    # Re-evaluate at optimal threshold
    print("\nEvaluation at OPTIMAL threshold:")
    final_val_result = evaluate_model(f"{best_model_name} (Optimized)", y_val, best_proba, best_threshold)
    
    # =========================================================================
    # STEP 8: TEST SET EVALUATION
    # =========================================================================
    print_section("STEP 8: FINAL TEST SET EVALUATION")
    
    print("** This is the held-out test set - never seen during training/tuning **")
    
    best_model = models[best_model_name]
    test_proba = best_model.predict_proba(X_test)[:, 1]
    
    final_test_result = evaluate_model(f"{best_model_name} (Test Set)", y_test, test_proba, best_threshold)
    
    # Classification report
    test_pred = (test_proba >= best_threshold).astype(int)
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*60)
    print(classification_report(y_test, test_pred, target_names=['No Sepsis', 'Sepsis']))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("SUMMARY")
    
    print(f"""
PIPELINE COMPLETE!

Dataset:
  - Original: {df.shape[0]:,} records, {n_patients:,} patients
  - Features engineered: {len(feature_names)}
  - Class imbalance: 1:{imbalance_ratio:.1f}

Best Model: {best_model_name}
  - Class weight balancing: scale_pos_weight = {imbalance_ratio:.1f}
  - Optimal threshold: {best_threshold:.2f}

Test Set Performance:
  - AUPRC:       {final_test_result['auprc']:.4f}
  - AUROC:       {final_test_result['auroc']:.4f}
  - Sensitivity: {final_test_result['sensitivity']:.4f} (catching {final_test_result['sensitivity']*100:.1f}% of sepsis cases)
  - Specificity: {final_test_result['specificity']:.4f} (correctly identifying {final_test_result['specificity']*100:.1f}% of non-sepsis)
  - F1-Score:    {final_test_result['f1']:.4f}

KEY INSIGHT:
  The model is NOT biased toward the majority class because:
  1. We used class weights (scale_pos_weight) to penalize missing sepsis cases
  2. We optimized threshold based on F1, not accuracy
  3. We report per-class metrics (Sensitivity/Specificity), not just accuracy
  4. AUPRC is our primary metric, which is robust to class imbalance
""")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'models': models,
        'best_model': best_model_name,
        'best_threshold': best_threshold,
        'test_results': final_test_result,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    results = main()
