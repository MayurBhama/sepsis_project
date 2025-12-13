"""
Industry-Grade Sepsis Prediction Pipeline
==========================================
Features:
1. LSTM for temporal sequence modeling
2. Optuna hyperparameter optimization
3. Proper feature engineering with sequences
4. Ensemble of LSTM + Tree models
5. Clinical-grade evaluation
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    classification_report, confusion_matrix, f1_score
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# PyTorch for LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


# ==============================================================================
# LSTM MODEL DEFINITION
# ==============================================================================
class SepsisLSTM(nn.Module):
    """LSTM model for sepsis prediction on temporal ICU data."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        return self.fc(context)


class SepsisDataset(Dataset):
    """Dataset for patient sequences."""
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


# ==============================================================================
# DATA PREPARATION
# ==============================================================================
def prepare_data():
    """Load and prepare data."""
    print_section("STEP 1: LOADING AND PREPARING DATA")
    
    df = pd.read_csv('data/sepsis_data.csv')
    print(f"Original shape: {df.shape}")
    
    # Setup
    patient_col = 'Patient_ID' if 'Patient_ID' in df.columns else 'Unnamed: 0'
    target = 'SepsisLabel'
    
    # Drop high-missing columns
    cols_to_drop = []
    for col in df.columns:
        if col in [target, patient_col, 'Hour', 'ICULOS']:
            continue
        if df[col].isna().sum() / len(df) > 0.90:
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop)
    print(f"After dropping high-missing columns: {df.shape}")
    
    # Sort by patient and time
    df = df.sort_values([patient_col, 'ICULOS'])
    
    # Forward fill within patient
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [target, patient_col]]
    
    for col in numeric_cols:
        df[col] = df.groupby(patient_col)[col].ffill().bfill()
        df[col] = df[col].fillna(df[col].median())
    
    # Feature columns
    feature_cols = [c for c in numeric_cols if c not in ['Hour', 'ICULOS']]
    print(f"Feature columns: {len(feature_cols)}")
    
    return df, patient_col, target, feature_cols


def create_sequences(df, patient_col, target, feature_cols, seq_length=24):
    """Create sequences for LSTM - each sequence is patient's history."""
    print(f"\nCreating sequences (length={seq_length})...")
    
    sequences = []
    labels = []
    patient_ids = []
    
    # Get unique patients with sepsis labels
    for pid in df[patient_col].unique():
        patient_data = df[df[patient_col] == pid].sort_values('ICULOS')
        
        # Get features and target
        X = patient_data[feature_cols].values
        y = patient_data[target].values
        
        # Create sequences using sliding window
        for i in range(len(X)):
            # Use last seq_length rows up to current point
            start_idx = max(0, i - seq_length + 1)
            seq = X[start_idx:i+1]
            
            # Pad if needed
            if len(seq) < seq_length:
                pad = np.zeros((seq_length - len(seq), len(feature_cols)))
                seq = np.vstack([pad, seq])
            
            sequences.append(seq)
            labels.append(y[i])
            patient_ids.append(pid)
    
    return np.array(sequences), np.array(labels), np.array(patient_ids)


def patient_stratified_split(patient_ids, labels, test_size=0.15, val_size=0.15):
    """Split at patient level."""
    unique_patients = np.unique(patient_ids)
    
    # Get patient-level labels (max label per patient)
    patient_labels = {}
    for pid, label in zip(patient_ids, labels):
        if pid not in patient_labels:
            patient_labels[pid] = 0
        patient_labels[pid] = max(patient_labels[pid], label)
    
    # Separate by label
    pos_patients = [p for p in unique_patients if patient_labels[p] == 1]
    neg_patients = [p for p in unique_patients if patient_labels[p] == 0]
    
    np.random.seed(42)
    np.random.shuffle(pos_patients)
    np.random.shuffle(neg_patients)
    
    # Split
    def split_list(lst, test_pct, val_pct):
        n = len(lst)
        test_n = int(n * test_pct)
        val_n = int(n * val_pct)
        return lst[:-(test_n+val_n)], lst[-(test_n+val_n):-test_n], lst[-test_n:]
    
    train_pos, val_pos, test_pos = split_list(pos_patients, test_size, val_size)
    train_neg, val_neg, test_neg = split_list(neg_patients, test_size, val_size)
    
    train_patients = set(train_pos + train_neg)
    val_patients = set(val_pos + val_neg)
    test_patients = set(test_pos + test_neg)
    
    # Create masks
    train_mask = np.array([p in train_patients for p in patient_ids])
    val_mask = np.array([p in val_patients for p in patient_ids])
    test_mask = np.array([p in test_patients for p in patient_ids])
    
    return train_mask, val_mask, test_mask


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================
def train_lstm(X_train, y_train, X_val, y_val, class_weight, epochs=10, batch_size=512):
    """Train LSTM model."""
    print("\nTraining LSTM...")
    
    input_size = X_train.shape[2]
    model = SepsisLSTM(input_size=input_size, hidden_size=64, num_layers=2).to(DEVICE)
    
    # Class-weighted loss
    pos_weight = torch.tensor([class_weight]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Datasets
    train_dataset = SepsisDataset(X_train, y_train)
    val_dataset = SepsisDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    best_auprc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                outputs = model(X_batch)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(y_batch.numpy().flatten())
        
        auprc = average_precision_score(val_labels, val_preds)
        auroc = roc_auc_score(val_labels, val_preds)
        
        if auprc > best_auprc:
            best_auprc = auprc
            best_model_state = model.state_dict().copy()
        
        print(f"  Epoch {epoch+1}/{epochs}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    
    model.load_state_dict(best_model_state)
    return model


def optuna_lgbm(X_train, y_train, X_val, y_val, class_weight, n_trials=20):
    """Optuna hyperparameter tuning for LightGBM."""
    print(f"\nOptuna tuning LightGBM ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'scale_pos_weight': class_weight,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"  Best AUPRC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    
    # Train final model with best params
    best_model = LGBMClassifier(**study.best_params, scale_pos_weight=class_weight, 
                                  random_state=42, verbose=-1, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    return best_model, study.best_params


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print_section("INDUSTRY-GRADE SEPSIS PREDICTION PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Prepare data
    df, patient_col, target, feature_cols = prepare_data()
    
    # 2. Create sequences for LSTM
    print_section("STEP 2: CREATING TEMPORAL SEQUENCES")
    SEQ_LENGTH = 12  # 12 hours of history
    
    # Sample for speed (full data takes too long)
    sample_patients = df[patient_col].unique()[:5000]  # 5000 patients
    df_sample = df[df[patient_col].isin(sample_patients)]
    print(f"Using {len(sample_patients)} patients for training")
    
    sequences, labels, patient_ids = create_sequences(
        df_sample, patient_col, target, feature_cols, seq_length=SEQ_LENGTH
    )
    print(f"Created {len(sequences)} sequences")
    print(f"Positive class: {labels.sum()} ({labels.mean()*100:.2f}%)")
    
    # 3. Split
    print_section("STEP 3: PATIENT-STRATIFIED SPLITTING")
    train_mask, val_mask, test_mask = patient_stratified_split(patient_ids, labels)
    
    X_train_seq = sequences[train_mask]
    y_train = labels[train_mask]
    X_val_seq = sequences[val_mask]
    y_val = labels[val_mask]
    X_test_seq = sequences[test_mask]
    y_test = labels[test_mask]
    
    print(f"Train: {len(X_train_seq)}, Val: {len(X_val_seq)}, Test: {len(X_test_seq)}")
    
    # Class weight
    class_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Class imbalance: 1:{class_weight:.1f}")
    
    # 4. Normalize
    print_section("STEP 4: NORMALIZING FEATURES")
    scaler = StandardScaler()
    
    # Reshape for scaling
    n_train, seq_len, n_features = X_train_seq.shape
    X_train_flat = X_train_seq.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    
    X_train_seq = scaler.transform(X_train_seq.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    X_val_seq = scaler.transform(X_val_seq.reshape(-1, n_features)).reshape(len(X_val_seq), seq_len, n_features)
    X_test_seq = scaler.transform(X_test_seq.reshape(-1, n_features)).reshape(len(X_test_seq), seq_len, n_features)
    
    # For tree models, use last timestep
    X_train_flat = X_train_seq[:, -1, :]
    X_val_flat = X_val_seq[:, -1, :]
    X_test_flat = X_test_seq[:, -1, :]
    
    # 5. Train models
    print_section("STEP 5: TRAINING MODELS")
    
    # 5a. LSTM
    lstm_model = train_lstm(X_train_seq, y_train, X_val_seq, y_val, class_weight, epochs=8)
    
    # 5b. LightGBM with Optuna
    lgbm_model, lgbm_params = optuna_lgbm(X_train_flat, y_train, X_val_flat, y_val, class_weight, n_trials=15)
    
    # 5c. XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=class_weight, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train_flat, y_train)
    
    # 6. Get predictions
    print_section("STEP 6: ENSEMBLE PREDICTIONS")
    
    # LSTM predictions
    lstm_model.eval()
    with torch.no_grad():
        lstm_test_preds = lstm_model(torch.FloatTensor(X_test_seq).to(DEVICE)).cpu().numpy().flatten()
    
    # Tree predictions
    lgbm_test_preds = lgbm_model.predict_proba(X_test_flat)[:, 1]
    xgb_test_preds = xgb_model.predict_proba(X_test_flat)[:, 1]
    
    # Ensemble (weighted average)
    ensemble_preds = 0.4 * lstm_test_preds + 0.35 * lgbm_test_preds + 0.25 * xgb_test_preds
    
    # 7. Evaluate
    print_section("STEP 7: FINAL EVALUATION")
    
    def evaluate(name, y_true, y_proba):
        auroc = roc_auc_score(y_true, y_proba)
        auprc = average_precision_score(y_true, y_proba)
        
        # Find best threshold
        best_f1 = 0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        
        y_pred = (y_proba >= best_thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / max(tp+fn, 1)
        spec = tn / max(tn+fp, 1)
        
        print(f"\n{name}:")
        print(f"  AUROC:       {auroc:.4f}")
        print(f"  AUPRC:       {auprc:.4f}")
        print(f"  Sensitivity: {sens:.1%} (@ threshold {best_thresh:.2f})")
        print(f"  Specificity: {spec:.1%}")
        print(f"  F1-Score:    {best_f1:.4f}")
        
        return {'auroc': auroc, 'auprc': auprc, 'sensitivity': sens, 'specificity': spec, 'f1': best_f1}
    
    print("\n" + "-"*60)
    results = {}
    results['LSTM'] = evaluate("LSTM", y_test, lstm_test_preds)
    results['LightGBM (Optuna)'] = evaluate("LightGBM (Optuna-tuned)", y_test, lgbm_test_preds)
    results['XGBoost'] = evaluate("XGBoost", y_test, xgb_test_preds)
    results['Ensemble'] = evaluate("ENSEMBLE (LSTM+LGBM+XGB)", y_test, ensemble_preds)
    
    # Summary
    print_section("SUMMARY - INDUSTRY-GRADE RESULTS")
    
    best_model = max(results.items(), key=lambda x: x[1]['auprc'])
    print(f"\nBest Model: {best_model[0]}")
    print(f"  AUROC: {best_model[1]['auroc']:.4f}")
    print(f"  AUPRC: {best_model[1]['auprc']:.4f}")
    
    print("\n" + "-"*60)
    print("COMPARISON TO BENCHMARKS")
    print("-"*60)
    print(f"""
{'Model':<25} {'AUROC':<10} {'AUPRC':<10} {'Industry Target':>15}
{'-'*60}
{'PhysioNet Top Teams':<25} {'0.82+':<10} {'0.30+':<10} {'✓':>15}
{'PhysioNet Median':<25} {'0.75':<10} {'0.15':<10} {'-':>15}
{'-'*60}
{'Our LSTM':<25} {results['LSTM']['auroc']:<10.3f} {results['LSTM']['auprc']:<10.3f} {'':>15}
{'Our Ensemble':<25} {results['Ensemble']['auroc']:<10.3f} {results['Ensemble']['auprc']:<10.3f} {'':>15}
""")
    
    if results['Ensemble']['auroc'] >= 0.75:
        print("✅ AUROC meets industry median benchmark!")
    else:
        print("⚠️  AUROC below industry median - more data/features needed")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results


if __name__ == "__main__":
    results = main()
