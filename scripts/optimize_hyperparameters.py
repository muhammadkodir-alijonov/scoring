#!/usr/bin/env python3
"""
Hyperparameter Optimization for XGBoost Credit Scoring Model
Optimizes XGBoost parameters using RandomizedSearchCV with cross-validation
Goal: Maximize AUC and reduce overfitting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score
from scipy.stats import uniform, randint
import xgboost as xgb
from data_loader import merge_all_data
from feature_engineering import prepare_training_data
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("XGBoost Hyperparameter Optimization - AUC Maximization")
    print("="*70)
    
    # Load data
    print("\n1. Loading training data...")
    data_dir = "../data"
    df = merge_all_data(data_dir)
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Default distribution: {df['default'].value_counts().to_dict()}")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y, customer_ids, feature_cols = prepare_training_data(df)
    print(f"   ✓ Features shape: {X.shape}")
    print(f"   ✓ Number of features: {len(feature_cols)}")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   ✓ Features scaled")
    
    # Define parameter distributions for RandomizedSearch
    print("\n4. Setting up parameter search space...")
    
    # Calculate class imbalance ratio
    class_ratio = (y == 0).sum() / (y == 1).sum()
    print(f"   ✓ Class imbalance ratio: {class_ratio:.1f}:1")
    
    param_distributions = {
        # Number of boosting rounds
        'n_estimators': randint(300, 1000),
        
        # Tree depth - shallow for generalization
        'max_depth': randint(3, 7),
        
        # Learning rate - slower learning = better generalization
        'learning_rate': uniform(0.01, 0.15),
        
        # Subsample ratio - prevents overfitting
        'subsample': uniform(0.6, 0.3),  # 0.6 to 0.9
        
        # Feature sampling per tree
        'colsample_bytree': uniform(0.6, 0.3),  # 0.6 to 0.9
        
        # Minimum child weight - higher = more conservative
        'min_child_weight': randint(1, 20),
        
        # Minimum loss reduction for split
        'gamma': uniform(0, 0.5),
        
        # L1 regularization
        'reg_alpha': uniform(0, 1.0),
        
        # L2 regularization
        'reg_lambda': uniform(0.5, 3.0),
        
        # Handle class imbalance
        'scale_pos_weight': uniform(class_ratio * 0.5, class_ratio * 1.0)
    }
    
    print("   ✓ Parameter search space configured")
    print("\n   Search ranges:")
    print(f"      - n_estimators: 300-1000")
    print(f"      - max_depth: 3-7")
    print(f"      - learning_rate: 0.01-0.16")
    print(f"      - subsample: 0.6-0.9")
    print(f"      - colsample_bytree: 0.6-0.9")
    print(f"      - min_child_weight: 1-20")
    print(f"      - gamma: 0-0.5")
    print(f"      - reg_alpha: 0-1.0")
    print(f"      - reg_lambda: 0.5-3.5")
    print(f"      - scale_pos_weight: {class_ratio*0.5:.1f}-{class_ratio*1.5:.1f}")
    
    # Base XGBoost model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='auc'
    )
    
    # Setup cross-validation
    print("\n5. Setting up cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"   ✓ Using 5-fold Stratified CV")
    
    # Setup RandomizedSearchCV
    print("\n6. Starting RandomizedSearchCV...")
    print("   ⏳ This will take 15-30 minutes...")
    print("   ⏳ Testing 60 parameter combinations with 5-fold CV = 300 fits")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=60,  # Test 60 random combinations
        scoring='roc_auc',
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )
    
    random_search.fit(X_scaled, y)
    
    # Results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\n✓ Best CV AUC Score: {random_search.best_score_:.4f}")
    
    print("\n✓ Best Parameters:")
    for param, value in sorted(random_search.best_params_.items()):
        if isinstance(value, float):
            print(f"   {param:20s}: {value:.4f}")
        else:
            print(f"   {param:20s}: {value}")
    
    # Get detailed CV results
    cv_results = pd.DataFrame(random_search.cv_results_)
    
    # Top 5 parameter combinations
    print("\n📊 Top 5 Parameter Combinations:")
    print("-" * 70)
    top_5 = cv_results.nlargest(5, 'mean_test_score')[
        ['mean_test_score', 'std_test_score', 'mean_train_score', 'params']
    ]
    
    for idx, row in top_5.iterrows():
        print(f"\nRank {top_5.index.get_loc(idx) + 1}:")
        print(f"  CV AUC:    {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f})")
        print(f"  Train AUC: {row['mean_train_score']:.4f}")
        print(f"  Overfitting: {(row['mean_train_score'] - row['mean_test_score']):.4f}")
        print(f"  Params: {row['params']}")
    
    # Save results
    results_file = "../models/hyperparameter_optimization_results.csv"
    cv_results.to_csv(results_file, index=False)
    print(f"\n✓ Full results saved to: {results_file}")
    
    # Generate updated credit_scorer.py code snippet
    print("\n" + "="*70)
    print("UPDATED MODEL CONFIGURATION")
    print("="*70)
    print("\nCopy this configuration to credit_scorer.py (xgboost model):\n")
    
    best_params = random_search.best_params_
    print("self.model = xgb.XGBClassifier(")
    print(f"    n_estimators={best_params['n_estimators']},")
    print(f"    max_depth={best_params['max_depth']},")
    print(f"    learning_rate={best_params['learning_rate']:.4f},")
    print(f"    subsample={best_params['subsample']:.4f},")
    print(f"    colsample_bytree={best_params['colsample_bytree']:.4f},")
    print(f"    min_child_weight={best_params['min_child_weight']},")
    print(f"    gamma={best_params['gamma']:.4f},")
    print(f"    reg_alpha={best_params['reg_alpha']:.4f},")
    print(f"    reg_lambda={best_params['reg_lambda']:.4f},")
    print(f"    scale_pos_weight={best_params['scale_pos_weight']:.4f},")
    print("    random_state=42,")
    print("    n_jobs=-1,")
    print("    tree_method='hist',")
    print("    eval_metric='auc'")
    print(")")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*70)
    
    baseline_cv = 0.7952
    improvement = random_search.best_score_ - baseline_cv
    
    print(f"\n📈 Improvement Analysis:")
    print(f"   Baseline CV AUC:  {baseline_cv:.4f}")
    print(f"   Optimized CV AUC: {random_search.best_score_:.4f}")
    print(f"   Improvement:      {improvement:+.4f} ({improvement/baseline_cv*100:+.2f}%)")
    
    # Check overfitting
    best_idx = random_search.best_index_
    train_auc = cv_results.loc[best_idx, 'mean_train_score']
    test_auc = cv_results.loc[best_idx, 'mean_test_score']
    overfit = train_auc - test_auc
    
    print(f"\n🎯 Overfitting Analysis:")
    print(f"   Train AUC: {train_auc:.4f}")
    print(f"   CV AUC:    {test_auc:.4f}")
    print(f"   Gap:       {overfit:.4f}")
    
    if overfit < 0.03:
        print("   ✓ Excellent! Minimal overfitting")
    elif overfit < 0.05:
        print("   ✓ Good! Acceptable overfitting level")
    elif overfit < 0.08:
        print("   ⚠️  Moderate overfitting - consider more regularization")
    else:
        print("   ❌ High overfitting - need stronger regularization")
    
    print("\n💡 Next Steps:")
    if random_search.best_score_ < 0.85:
        print("   1. Try feature selection (remove low importance features)")
        print("   2. Create more interaction features")
        print("   3. Try ensemble stacking (XGBoost + LightGBM + CatBoost)")
        print("   4. Consider external data sources")
    elif random_search.best_score_ < 0.90:
        print("   1. Fine-tune around best parameters (narrow grid search)")
        print("   2. Try ensemble stacking")
        print("   3. Feature engineering refinement")
    else:
        print("   1. Test on held-out validation set")
        print("   2. Deploy to production!")
    
    print("\n" + "="*70)
    print("Optimization completed!")
    print("="*70)

if __name__ == "__main__":
    main()
