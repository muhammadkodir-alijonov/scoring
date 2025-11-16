"""
Credit Default Prediction Model using CatBoost
Refactored with pipeline architecture for better maintainability
"""

import time
import warnings
from pipeline.data import load_datasets, load_test_datasets, standardize_column_names, merge_datasets
from pipeline.features import preprocess_data
from pipeline.train import (
    create_catboost_model, perform_cross_validation, train_final_model,
    display_results, display_feature_importance,
    make_predictions_on_test, save_results
)

warnings.filterwarnings('ignore')


def main():
    """Main execution function"""
    print("="*60)
    print(" CREDIT DEFAULT PREDICTION - CATBOOST")
    print("="*60)

    start_time = time.time()

    # Load and prepare training data
    print("\n[1/5] Loading training data...")
    dfs = load_datasets()
    dfs = standardize_column_names(dfs)
    df_merged = merge_datasets(dfs)

    # Preprocess training data
    print("[2/5] Preprocessing training data...")
    df_merged = preprocess_data(df_merged)

    # Prepare features
    X = df_merged.drop('default', axis=1)
    y = df_merged['default']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

    print(f"\nDataset prepared:")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Target: 0={y.value_counts()[0]:,} ({y.value_counts()[0]/len(y)*100:.1f}%) | 1={y.value_counts()[1]:,} ({y.value_counts()[1]/len(y)*100:.1f}%)")

    # Model training
    print("\n[3/5] Training model with cross-validation...")
    model = create_catboost_model(categorical_indices)

    cv_start = time.time()
    auc_scores, accuracy_scores, all_y_true, all_y_pred, all_y_pred_proba = perform_cross_validation(
        model, X, y, categorical_indices
    )
    cv_time = time.time() - cv_start

    # Display results
    display_results(auc_scores, accuracy_scores, all_y_true, all_y_pred, all_y_pred_proba)

    # Train final model on full training data
    print("\n[4/5] Training final model on complete dataset...")
    final_model = train_final_model(model, X, y, categorical_indices)

    # Feature importance
    display_feature_importance(final_model, X)

    # Load and process test data
    print("\n[5/5] Making predictions on test data...")
    test_dfs = load_test_datasets()
    test_dfs = standardize_column_names(test_dfs)
    df_test_merged = merge_datasets(test_dfs)

    # Store customer IDs before preprocessing
    test_customer_ids = df_test_merged['customer_id'].copy()

    # Preprocess test data
    df_test_merged = preprocess_data(df_test_merged)

    # Ensure test data has same features as training data
    # Add missing columns with 0
    for col in X.columns:
        if col not in df_test_merged.columns:
            df_test_merged[col] = 0

    # Remove extra columns not in training
    df_test_merged = df_test_merged[X.columns]

    # Make predictions
    results_df = make_predictions_on_test(final_model, df_test_merged, test_customer_ids)

    # Save results
    save_results(results_df, 'results.csv')

    print(f"\nPerformance:")
    print(f"  Training time: {cv_time:.2f}s")
    print(f"  Total runtime: {time.time() - start_time:.2f}s")
    print(f"{'='*60}")

    print("\nCATBOOST MODEL READY!")


if __name__ == "__main__":
    main()
