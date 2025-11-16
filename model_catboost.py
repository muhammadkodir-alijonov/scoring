"""
Credit Default Prediction Model using CatBoost
Structured and modularized code for better maintainability
"""

import pandas as pd
import numpy as np
import time
from functools import reduce
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_datasets():
    """Load all required data files"""
    df_financial_ratios = pd.read_json('data/financial_ratios.jsonl', lines=True)
    df_geographic_data = pd.read_xml('data/geographic_data.xml')
    df_credit_history = pd.read_parquet('data/credit_history.parquet')
    df_demographics = pd.read_csv('data/demographics.csv')
    df_loan_details = pd.read_excel('data/loan_details.xlsx')
    df_application_metadata = pd.read_csv('data/application_metadata.csv')
    
    return (df_financial_ratios, df_geographic_data, df_credit_history,
            df_demographics, df_loan_details, df_application_metadata)


def standardize_column_names(dfs):
    """Standardize customer ID column names across all dataframes"""
    df_financial_ratios, df_geographic_data, df_credit_history, df_demographics, df_loan_details, df_application_metadata = dfs
    
    df_financial_ratios.rename(columns={'cust_num': 'customer_id'}, inplace=True)
    df_demographics.rename(columns={'cust_id': 'customer_id'}, inplace=True)
    df_credit_history.rename(columns={'customer_number': 'customer_id'}, inplace=True)
    df_application_metadata.rename(columns={'customer_ref': 'customer_id'}, inplace=True)
    df_geographic_data.rename(columns={'id': 'customer_id'}, inplace=True)
    
    return (df_financial_ratios, df_geographic_data, df_credit_history,
            df_demographics, df_loan_details, df_application_metadata)


def merge_datasets(dfs):
    """Merge all datasets on customer_id"""
    return reduce(lambda left, right: pd.merge(left, right, on='customer_id', how='outer'), dfs)



# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def clean_currency_columns(df):
    """Clean and convert currency columns to numeric"""
    currency_columns = [
       'monthly_income', 'existing_monthly_debt', 'monthly_payment',
       'revolving_balance', 'credit_usage_amount', 'available_credit',
       'total_monthly_debt_payment', 'total_debt_amount',
       'monthly_free_cash_flow', 'annual_income', 'loan_amount'
    ]
    
    for col in currency_columns:
       if col in df.columns:
           df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
           df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def remove_outliers(df):
    """Remove outliers using 3*IQR method"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != 'default':
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 3 * IQR, Q3 + 3 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def drop_unnecessary_columns(df):
    """Drop columns that don't contribute to prediction"""
    columns_to_drop = [
        'revolving_balance', 'oldest_credit_line_age', 'recent_inquiry_count',
        'cost_of_living_index', 'regional_unemployment_rate', 'housing_price_index',
        'random_noise_1', 'regional_median_income', 'regional_median_rent',
        'loan_officer_id', 'application_id', 'num_customer_service_calls',
        'num_inquiries_6mo', 'application_hour', 'account_open_year',
        'num_collections', 'previous_zip_code', 'customer_id', 'application_day_of_week'
    ]
    return df.drop(columns=columns_to_drop, axis=1, errors='ignore')


def fill_missing_values(df):
    """Fill missing values with appropriate strategies"""
    df['num_delinquencies_2yrs'] = df['num_delinquencies_2yrs'].fillna(0)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df


def standardize_categorical_columns(df):
    """Standardize categorical column values"""
    if 'loan_type' in df.columns:
        df['loan_type'] = (df['loan_type'].astype(str).str.lower().str.strip()
            .str.replace(r'personal.*', 'personal', regex=True)
            .str.replace(r'(mortgage|home loan)', 'mortgage', regex=True)
            .str.replace(r'(credit ?card|cc)', 'credit_card', regex=True))
    
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].str.upper().replace({
            'FULL-TIME': 'FULLTIME', 'FULL_TIME': 'FULLTIME', 'FULL TIME': 'FULLTIME', 'FULLTIME': 'FULLTIME', 'FT': 'FULLTIME',
            'PART TIME': 'PARTTIME', 'PART_TIME': 'PARTTIME', 'PT': 'PARTTIME', 'PART-TIME': 'PARTTIME',
            'SELF EMPLOYED': 'SELFEMPLOYED', 'SELF EMP': 'SELFEMPLOYED', 'SELF-EMPLOYED': 'SELFEMPLOYED', 'SELF_EMPLOYED': 'SELFEMPLOYED',
        })
    
    if 'education' in df.columns:
        df['education'] = df['education'].str.strip().str.title()
    
    if 'marital_status' in df.columns:
        df['marital_status'] = df['marital_status'].str.strip().str.title()
    
    return df



# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def remove_highly_correlated_features(df, threshold=0.90):
    """Remove features with correlation > threshold"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    columns_to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                if col1 != 'default' and col2 != 'default':
                    # Only use target correlation if 'default' column exists
                    if 'default' in numeric_df.columns:
                        corr_target = numeric_df.corr()['default'].abs()
                        if corr_target[col1] < corr_target[col2]:
                            columns_to_drop.add(col1)
                        else:
                            columns_to_drop.add(col2)
                    else:
                        # If no target, just drop the first column
                        columns_to_drop.add(col1)

    if columns_to_drop:
        df = df.drop(columns=list(columns_to_drop), axis=1)

    return df


def remove_low_correlation_features(df, threshold=0.05):
    """Remove features with low correlation to target"""
    # Only remove low correlation features if 'default' column exists
    if 'default' in df.columns:
        correlation = df.select_dtypes(include=[np.number]).corr()['default'].abs()
        low_corr_features = [col for col in correlation.index if correlation[col] < threshold and col != 'default']

        if low_corr_features:
            df = df.drop(columns=low_corr_features, axis=1, errors='ignore')

    return df


def create_interaction_features(df):
    """Create interaction features based on domain knowledge"""
    if 'existing_monthly_debt' in df.columns and 'monthly_payment' in df.columns:
        df['debt_burden'] = (df['existing_monthly_debt'] + df['monthly_payment']) / (df['annual_income'] / 12 + 1)
    
    if 'credit_utilization' in df.columns and 'debt_to_income_ratio' in df.columns:
        df['credit_pressure'] = df['credit_utilization'] * df['debt_to_income_ratio']
    
    if 'monthly_payment' in df.columns and 'monthly_free_cash_flow' in df.columns:
        df['payment_stress'] = df['monthly_payment'] / (df['monthly_free_cash_flow'] + 1)
    
    if 'credit_score' in df.columns and 'credit_utilization' in df.columns:
        df['credit_efficiency'] = df['credit_score'] / (df['credit_utilization'] + 0.01)
    
    if 'num_delinquencies_2yrs' in df.columns and 'debt_to_income_ratio' in df.columns:
        df['delinquency_severity'] = df['num_delinquencies_2yrs'] * df['debt_to_income_ratio']
    
    return df


def drop_low_value_categorical_features(df):
    """Drop categorical features with low predictive value"""
    categorical_features_to_drop = [
        'referral_code',
        'marketing_campaign',
        'employment_type',
        'account_status_code',
        'preferred_contact',
    ]
    
    for col in categorical_features_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df


def handle_infinite_values(df):
    """Replace infinite values with NaN and fill with median"""
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    return df


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
# TEST DATA PROCESSING FUNCTIONS
# ============================================================================

def load_test_datasets():
    """Load all test data files"""
    df_financial_ratios = pd.read_json('data/tests/financial_ratios.jsonl', lines=True)
    df_geographic_data = pd.read_xml('data/tests/geographic_data.xml')
    df_credit_history = pd.read_parquet('data/tests/credit_history.parquet')
    df_demographics = pd.read_csv('data/tests/demographics.csv')
    df_loan_details = pd.read_excel('data/tests/loan_details.xlsx')
    df_application_metadata = pd.read_csv('data/tests/application_metadata.csv')

    return (df_financial_ratios, df_geographic_data, df_credit_history,
            df_demographics, df_loan_details, df_application_metadata)


def preprocess_data(df_merged):
    """Apply all preprocessing steps to data"""
    df_merged = clean_currency_columns(df_merged)
    df_merged = remove_outliers(df_merged)
    df_merged = drop_unnecessary_columns(df_merged)
    df_merged = fill_missing_values(df_merged)
    df_merged = standardize_categorical_columns(df_merged)
    df_merged = remove_highly_correlated_features(df_merged)
    df_merged = remove_low_correlation_features(df_merged)
    df_merged = create_interaction_features(df_merged)
    df_merged = drop_low_value_categorical_features(df_merged)
    df_merged = handle_infinite_values(df_merged)
    return df_merged


def make_predictions_on_test(final_model, X_test, customer_ids, threshold=0.5):
    """Make predictions on test data and format results"""
    # Make predictions
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Create results dataframe
    results_df = pd.DataFrame({
        'customer_id': customer_ids,
        'predicted_probability': y_pred_proba,
        'verdict': y_pred
    })

    return results_df


def save_results(results_df, output_path='results.csv'):
    """Save prediction results to CSV"""
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total predictions: {len(results_df):,}")
    print(f"Predicted defaults: {results_df['verdict'].sum():,} ({results_df['verdict'].sum()/len(results_df)*100:.1f}%)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

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
