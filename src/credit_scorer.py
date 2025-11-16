"""
Credit Scoring Model
Main module for training and predicting credit default
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, make_scorer, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from data_loader import merge_all_data, load_data_for_prediction
from feature_engineering import prepare_training_data, prepare_prediction_data
from xgboost import XGBClassifier

XGBOOST_AVAILABLE = True
SMOTE_AVAILABLE = True

class CreditScorer:
    """Credit scoring model for predicting loan defaults"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize credit scorer
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic', 'ensemble', 'optimized', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'xgboost_smote' and XGBOOST_AVAILABLE and SMOTE_AVAILABLE:
            # XGBoost + SMOTE - BEST FOR 90%+ AUC
            self.model = xgb.XGBClassifier(
                n_estimators=800,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.2,
                reg_alpha=0.3,
                reg_lambda=1.5,
                scale_pos_weight=1,  # SMOTE handles imbalance
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=50,
                eval_metric='auc',
                tree_method='hist'
            )
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost - OPTIMIZED FOR 90%+ AUC with reduced overfitting
            self.model = xgb.XGBClassifier(
                n_estimators=500,          # Kamroq tree (overfitting kamaytirish)
                max_depth=4,               # Shallow trees (generalizatsiya yaxshilash)
                learning_rate=0.05,        # Ko'proq learning rate (tezroq konvergensiya)
                subsample=0.7,             # Ko'proq regularization
                colsample_bytree=0.7,      # Ko'proq feature randomness
                min_child_weight=10,       # Ko'proq cheklov (overfitting oldini olish)
                gamma=0.3,                 # Ko'proq pruning
                reg_alpha=0.5,             # L1 regularization
                reg_lambda=2.0,            # L2 regularization
                scale_pos_weight=18,       # Class imbalance
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=30,
                eval_metric='auc',
                tree_method='hist'         # Tezroq va efficient
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                max_features='sqrt',
                n_jobs=-1,
                bootstrap=True,
                oob_score=True
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=1000,          # Ko'proq tree'lar
                max_depth=6,                # Balanced depth
                learning_rate=0.015,        # Optimal learning rate
                subsample=0.8,              # Ko'proq generalizatsiya
                min_samples_split=15,       # Overfitting oldini olish
                min_samples_leaf=8,         # Overfitting oldini olish
                max_features='sqrt',
                random_state=42,
                validation_fraction=0.15,   # Validation data
                n_iter_no_change=100,       # Katta patience
                tol=0.0001
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'ensemble':
            # ULTIMATE ENSEMBLE FOR 90%+ AUC
            # Highly optimized diverse models
            gb_model1 = GradientBoostingClassifier(
                n_estimators=1200,
                max_depth=5,
                learning_rate=0.008,
                subsample=0.75,
                min_samples_split=25,
                min_samples_leaf=12,
                max_features='sqrt',
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=200
            )
            
            gb_model2 = GradientBoostingClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='log2',
                random_state=123,
                validation_fraction=0.1,
                n_iter_no_change=200
            )
            
            gb_model3 = GradientBoostingClassifier(
                n_estimators=800,
                max_depth=7,
                learning_rate=0.015,
                subsample=0.85,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                random_state=456,
                validation_fraction=0.1,
                n_iter_no_change=150
            )
            
            rf_model = RandomForestClassifier(
                n_estimators=600,
                max_depth=18,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                max_features='sqrt',
                n_jobs=-1,
                bootstrap=True,
                max_samples=0.8
            )
            
            # Voting ensemble with soft voting (averages probabilities)
            self.model = VotingClassifier(
                estimators=[
                    ('gb1', gb_model1),
                    ('gb2', gb_model2),
                    ('gb3', gb_model3),
                    ('rf', rf_model)
                ],
                voting='soft',
                weights=[4, 3, 2, 2],  # GB1 gets most weight (most conservative)
                n_jobs=-1
            )
        elif model_type == 'optimized':
            # Highly optimized gradient boosting with best parameters for 90%+ AUC
            self.model = GradientBoostingClassifier(
                n_estimators=1000,          # Maksimal tree'lar
                max_depth=6,                # Yaxshi generalizatsiya uchun
                learning_rate=0.01,         # Muvozanatlashtirilgan
                subsample=0.8,              # Regularization
                min_samples_split=20,       # Overfitting oldini olish
                min_samples_leaf=10,        # Overfitting oldini olish
                max_features='log2',        # Ko'proq feature diversity
                random_state=42,
                validation_fraction=0.15,   # Validation
                n_iter_no_change=150,       # Katta patience
                tol=0.0001,                 # Aniqlik
                verbose=1,                  # Progress ko'rsatish
                warm_start=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, data_dir: str, test_size: float = 0.2) -> dict:
        """
        Train the credit scoring model
        
        Args:
            data_dir: Directory containing training data
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Load and merge all data
        print("Loading training data...")
        df = merge_all_data(data_dir)
        
        # Check if default column exists
        if 'default' not in df.columns:
            raise ValueError("Training data must contain 'default' column")
        
        print(f"Loaded {len(df)} records")
        print(f"Default distribution: {df['default'].value_counts().to_dict()}")
        
        # Prepare features
        print("Preparing features...")
        X, y, customer_ids, feature_cols = prepare_training_data(df)
        self.feature_cols = feature_cols
        
        print(f"Features shape: {X.shape}")
        print(f"Number of features: {len(feature_cols)}")
        
        # Split data
        # Check if we have enough samples for stratified split
        min_class_count = min(np.bincount(y.astype(int)))
        if min_class_count < 2 or len(y) < 10:
            # For small datasets, don't use stratification
            print("Warning: Small dataset detected. Training without test split.")
            X_train, X_test, y_train, y_test = X, X, y, y
            test_size = 0
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE for xgboost_smote model
        if self.model_type == 'xgboost_smote' and SMOTE_AVAILABLE:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {len(y_train)} samples, Default: {(y_train==1).sum()}, Non-default: {(y_train==0).sum()}")
        
        # Calculate sample weights to handle class imbalance
        # OPTIMIZED FOR 90%+ AUC: Carefully balanced weighting
        if self.model_type == 'xgboost_smote' and XGBOOST_AVAILABLE:
            # XGBoost+SMOTE with early stopping for best AUC
            print(f"Training {self.model_type} model with early stopping...")
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost with early stopping for best AUC
            print(f"Training {self.model_type} model with early stopping...")
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
        elif self.model_type in ['gradient_boosting', 'ensemble', 'optimized']:
            # Calculate class weights - Optimal for maximum AUC
            class_counts = np.bincount(y_train.astype(int))
            weight_non_default = 1.0
            # Use moderate weighting: between sqrt and full ratio
            # This gives best discrimination without overfitting to minority class
            ratio = class_counts[0] / class_counts[1]
            weight_default = np.power(ratio, 0.6)  # Optimal exponent for AUC
            
            sample_weights = np.where(y_train == 1, weight_default, weight_non_default)
            print(f"Using optimized sample weights - Default: {weight_default:.1f}x, Non-default: {weight_non_default}x")
            
            # Train model with sample weights
            print(f"Training {self.model_type} model with class weighting...")
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            
            # Perform cross-validation to check model stability
            if self.model_type in ['ensemble', 'optimized']:
                print("\nPerforming cross-validation...")
                # Use 'roc_auc' string directly - works with all sklearn versions
                cv_scores = cross_val_score(
                    self.model, X_train_scaled, y_train, 
                    cv=5, scoring='roc_auc', n_jobs=-1
                )
                print(f"CV AUC Scores: {cv_scores}")
                print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Apply probability calibration only for single models (not VotingClassifier)
                # Calibration improves probability estimates for better AUC
                if cv_scores.mean() < 0.90 and self.model_type == 'optimized':
                    print("\nApplying probability calibration to improve AUC...")
                    self.model = CalibratedClassifierCV(
                        self.model, 
                        method='isotonic',  # Better for tree-based models
                        cv=3,
                        n_jobs=-1
                    )
                    self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                elif self.model_type == 'ensemble':
                    print("\n✓ Ensemble model trained (calibration skipped for VotingClassifier)")
        elif self.model_type == 'xgboost_smote':
            # xgboost_smote already trained above
            pass
        else:
            # Other models might have class_weight parameter
            print(f"Training {self.model_type} model...")
            self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Cross-validation AUC check BEFORE final evaluation
        # For XGBoost with early stopping, we need to create a model WITHOUT callbacks
        print("\n" + "="*60)
        print("CROSS-VALIDATION AUC CHECK")
        print("="*60)
        
        if self.model_type in ['xgboost', 'xgboost_smote']:
            # Create a temporary XGBoost model without early stopping for CV
            cv_model = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=10,
                scale_pos_weight=18,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
                # NO early_stopping_rounds or callbacks here
            )
        else:
            cv_model = self.model
        
        cv_scores = cross_val_score(
            cv_model, X_train_scaled, y_train,
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        print(f"CV AUC Scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("="*60)
        
        # Evaluate
        print("\nEvaluating model on test set...")
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_auc': roc_auc_score(y_train, train_proba),
            'test_auc': roc_auc_score(y_test, test_proba),
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        print("\nTraining Results:")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Train AUC: {metrics['train_auc']:.4f}")
        print(f"Test AUC: {metrics['test_auc']:.4f}")
        print(f"CV AUC (Mean): {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']*2:.4f})")
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, test_pred, zero_division=0))
        
        return metrics
    
    def predict(self, data_dir: str) -> pd.DataFrame:
        """
        Predict credit default for new data
        
        Args:
            data_dir: Directory containing data for prediction
            
        Returns:
            DataFrame with customer_id and predicted default
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Load data
        print("Loading prediction data...")
        df = load_data_for_prediction(data_dir)
        
        print(f"Loaded {len(df)} records for prediction")
        
        # Prepare features
        print("Preparing features...")
        X, customer_ids = prepare_prediction_data(df, self.feature_cols)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        print("Making predictions...")
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # OPTIMAL THRESHOLD CALCULATION
        # For maximum AUC, we need balanced precision and recall
        # Current issue: too many false positives (threshold too low)
        
        # Use probability distribution to find optimal threshold
        # Aiming for ~15-20% default rate (realistic for credit data)
        threshold = 0.45  # Balanced threshold for high AUC and accuracy
        predictions = (probabilities >= threshold).astype(int)
        
        # Create results DataFrame in COMPETITION FORMAT
        # Format: customer_id, prob, default
        # All columns must be FLOAT for proper ranking and correlation
        results = pd.DataFrame({
            'customer_id': customer_ids.astype(int),     # Integer ID
            'prob': probabilities.astype(float),         # Float probability
            'default': predictions.astype(float)         # Float 0.0/1.0
        })
        
        print(f"\nPrediction Summary:")
        print(f"Predicted defaults: {(predictions == 1).sum()}")
        print(f"Predicted non-defaults: {(predictions == 0).sum()}")
        print(f"Default rate: {(predictions == 1).mean():.2%}")
        print(f"\nFormat: customer_id (int), prob (float), default (float)")
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models)
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.model_type in ['random_forest', 'gradient_boosting', 'xgboost', 'optimized']:
            # Get feature importances
            if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                importances = self.model.feature_importances_
            else:
                importances = self.model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importances
            })
            feature_importance = feature_importance.sort_values(
                'importance', ascending=False
            ).head(top_n)
            
            return feature_importance
        else:
            print(f"Feature importance not available for {self.model_type}")
            return None
