"""
Full Training Pipeline for Industry-Grade Sepsis Prediction

Runs the complete ML pipeline:
1. Data preprocessing with intelligent imputation
2. Feature engineering with 30+ temporal features
3. Patient-stratified train/val/test splitting
4. Ensemble model training (LightGBM + XGBoost + CatBoost)
5. Clinical evaluation with AUPRC metrics
"""

import sys
import pandas as pd
from src.logger import logger
from src.exception import CustomException

# Pipeline components
from src.preprocessing.preprocessor import AdvancedPreprocessor
from src.feature_engineering.feature_engineer import AdvancedFeatureEngineer
from src.pipelines.split_data import PatientStratifiedSplitter
from src.model_training.trainer import EnsembleModelTrainer


def run_full_pipeline(config_path: str = "config/config.yaml"):
    """
    Run the complete training pipeline from raw data to trained model.
    """
    try:
        logger.info("=" * 80)
        logger.info("INDUSTRY-GRADE SEPSIS PREDICTION PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load raw data
        logger.info("\n[STEP 1/5] Loading raw data...")
        df = pd.read_csv("data/sepsis_data.csv")
        logger.info(f"Loaded raw data: {df.shape}")
        
        # Step 2: Preprocessing
        logger.info("\n[STEP 2/5] Running advanced preprocessing...")
        preprocessor = AdvancedPreprocessor(config_path)
        df_clean = preprocessor.run_preprocessing(df)
        preprocessor.save_preprocessed(df_clean)
        logger.info(f"Preprocessed data: {df_clean.shape}")
        
        # Step 3: Feature Engineering
        logger.info("\n[STEP 3/5] Applying advanced feature engineering...")
        feature_engineer = AdvancedFeatureEngineer(config_path)
        df_features = feature_engineer.apply_features(df_clean)
        logger.info(f"Featured data: {df_features.shape}")
        
        # Save featured data
        df_features.to_csv("data/processed/featured_sepsis_data.csv", index=False)
        
        # Step 4: Patient-stratified splitting
        logger.info("\n[STEP 4/5] Splitting with patient stratification...")
        # Note: Splitter loads from preprocessed path, so we need to save featured data first
        df_features.to_csv(preprocessor.output_path, index=False)
        
        splitter = PatientStratifiedSplitter(config_path)
        train_df, val_df, test_df = splitter.split()
        splitter.save_splits(train_df, val_df, test_df)
        
        # Step 5: Ensemble Training
        logger.info("\n[STEP 5/5] Training ensemble model...")
        trainer = EnsembleModelTrainer(config_path)
        model_package = trainer.train()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {trainer.config.best_model_path}")
        logger.info(f"Validation AUPRC: {model_package['performance']['validation']['auprc']:.4f}")
        logger.info(f"Test AUPRC: {model_package['performance']['test']['auprc']:.4f}")
        logger.info("=" * 80)
        
        return model_package
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise CustomException(e, sys)


def run_training_only(config_path: str = "config/config.yaml"):
    """
    Run only the training step (assumes data is already preprocessed and split).
    """
    try:
        logger.info("===== TRAINING PIPELINE STARTED =====")
        trainer = EnsembleModelTrainer(config_path)
        model_package = trainer.train()
        logger.info("===== TRAINING PIPELINE COMPLETED SUCCESSFULLY =====")
        return model_package
    except Exception as e:
        raise CustomException(e, sys)


# Legacy compatibility
def run_training_pipeline():
    """Legacy entry point."""
    return run_training_only()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sepsis Prediction Training Pipeline")
    parser.add_argument("--full", action="store_true", help="Run full pipeline from raw data")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10% data")
    args = parser.parse_args()
    
    if args.full:
        run_full_pipeline()
    else:
        run_training_only()