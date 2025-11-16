"""
Model training, evaluation, and prediction pipeline
Handles CatBoost model creation, cross-validation, and predictions
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def create_catboost_model(categorical_indices):
    """Create and configure CatBoost classifier"""
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.015,
        depth=8,
        l2_leaf_reg=6,
        auto_class_weights='SqrtBalanced',
        early_stopping_rounds=50,
        eval_metric='AUC',
        cat_features=categorical_indices,
        task_type='CPU',
        random_seed=42,
        verbose=False
    )


def perform_cross_validation(model, X, y, categorical_indices, n_splits=5):
    """Perform stratified k-fold cross-validation"""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores = []
    accuracy_scores = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=categorical_indices)
        val_pool = Pool(X_val, y_val, cat_features=categorical_indices)

        fitted_model = model.fit(train_pool, eval_set=val_pool, verbose=False)

        y_pred_proba = fitted_model.predict_proba(val_pool)[:, 1]
        y_pred = fitted_model.predict(val_pool)

        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = (y_pred == y_val).mean()

        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_pred_proba.extend(y_pred_proba)

    return auc_scores, accuracy_scores, all_y_true, all_y_pred, all_y_pred_proba


def train_final_model(model, X, y, categorical_indices):
    """Train final model on full dataset"""
    final_pool = Pool(X, y, cat_features=categorical_indices)
    return model.fit(final_pool, verbose=False)


# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

def display_results(auc_scores, accuracy_scores, all_y_true, all_y_pred, all_y_pred_proba):
    """Display model performance metrics"""
    overall_auc = roc_auc_score(all_y_true, all_y_pred_proba)
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)

    print(f"\n{'='*60}")
    print(" CATBOOST MODEL PERFORMANCE")
    print(f"{'='*60}")

    print(f"\nCross-Validation:")
    print(f"  AUC scores: {[f'{s:.4f}' for s in auc_scores]}")
    print(f"  Mean AUC: {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")
    print(f"\n  Accuracy: {[f'{s:.4f}' for s in accuracy_scores]}")
    print(f"  Mean Accuracy: {np.mean(accuracy_scores):.4f} +/- {np.std(accuracy_scores):.4f}")

    print(f"\nOverall AUC-ROC: {overall_auc:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {conf_matrix[0][0]:,}")
    print(f"  False Positives: {conf_matrix[0][1]:,}")
    print(f"  False Negatives: {conf_matrix[1][0]:,}")
    print(f"  True Positives:  {conf_matrix[1][1]:,}")

    print(f"\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=['No Default', 'Default']))


def display_feature_importance(model, X):
    """Display top feature importance"""
    print("\nTop 15 Most Important Features:")
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    for i, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:35s}: {row['importance']:.2f}")

    return importance_df


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def make_predictions_on_test(final_model, X_test, customer_ids, threshold=0.5):
    """Make predictions on test data and format results"""
    # Make predictions
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Create results dataframe
    results_df = pd.DataFrame({
        'customer_id': customer_ids,
        'prob': y_pred_proba,
        'default': y_pred
    })

    return results_df


def save_results(results_df, output_path='results.csv'):
    """Save prediction results to CSV"""
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total predictions: {len(results_df):,}")
    print(f"Predicted defaults: {results_df['default'].sum():,} ({results_df['default'].sum()/len(results_df)*100:.1f}%)")
