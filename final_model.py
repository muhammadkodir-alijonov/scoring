import pandas as pd
import numpy as np
import time
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print(" CREDIT DEFAULT PREDICTION - CATBOOST")
print("=" * 60)

# Loading data
print("\n[1/7] Loading data files...")
start_time = time.time()

df_financial_ratios = pd.read_json('./data/financial_ratios.jsonl', lines=True)
df_geographic_data = pd.read_xml('./data/geographic_data.xml')
df_credit_history = pd.read_parquet('./data/credit_history.parquet')
df_demographics = pd.read_csv('./data/demographics.csv')
df_loan_details = pd.read_excel('./data/loan_details.xlsx')
df_application_metadata = pd.read_csv('./data/application_metadata.csv')

print(f"Loaded 6 files in {time.time() - start_time:.2f}s")

# Standardize column names
print("\n[2/7] Standardizing columns...")
df_financial_ratios.rename(columns={'cust_num': 'customer_id'}, inplace=True)
df_demographics.rename(columns={'cust_id': 'customer_id'}, inplace=True)
df_credit_history.rename(columns={'customer_number': 'customer_id'}, inplace=True)
df_application_metadata.rename(columns={'customer_ref': 'customer_id'}, inplace=True)
df_geographic_data.rename(columns={'id': 'customer_id'}, inplace=True)

# Merge datasets
print("\n[3/7] Merging datasets...")
df_merged = pd.merge(df_financial_ratios, df_demographics, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_credit_history, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_loan_details, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_application_metadata, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_geographic_data, on='customer_id', how='outer')
print(f"Merged shape: {df_merged.shape}")

# Data cleaning
print("\n[4/7] Cleaning data...")

# Clean currency columns
currency_columns = [
    'monthly_income', 'existing_monthly_debt', 'monthly_payment',
    'revolving_balance', 'credit_usage_amount', 'available_credit',
    'total_monthly_debt_payment', 'total_debt_amount',
    'monthly_free_cash_flow', 'annual_income', 'loan_amount'
]

for col in currency_columns:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

print(f"Cleaned {len([c for c in currency_columns if c in df_merged.columns])} currency columns")

# Remove outliers
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
outliers_removed = 0
for col in numeric_cols:
    if col != 'default':
        Q1, Q3 = df_merged[col].quantile(0.25), df_merged[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 3 * IQR, Q3 + 3 * IQR
        outliers_removed += ((df_merged[col] < lower_bound) | (df_merged[col] > upper_bound)).sum()
        df_merged[col] = df_merged[col].clip(lower=lower_bound, upper=upper_bound)

print(f"Clipped {outliers_removed:,} outliers")

# Drop unnecessary columns
columns_to_drop = [
    'revolving_balance', 'oldest_credit_line_age', 'recent_inquiry_count',
    'cost_of_living_index', 'regional_unemployment_rate', 'housing_price_index',
    'random_noise_1', 'regional_median_income', 'regional_median_rent',
    'loan_officer_id', 'application_id', 'num_customer_service_calls',
    'num_inquiries_6mo', 'application_hour', 'account_open_year',
    'num_collections', 'previous_zip_code', 'customer_id', 'application_day_of_week'
]
df_merged = df_merged.drop(columns=columns_to_drop, axis=1, errors='ignore')

# Fill missing values
df_merged['num_delinquencies_2yrs'] = df_merged['num_delinquencies_2yrs'].fillna(0)
for col in df_merged.select_dtypes(include=[np.number]).columns:
    if df_merged[col].isnull().sum() > 0:
        df_merged[col].fillna(df_merged[col].median(), inplace=True)

# Standardize categorical columns
df_merged['loan_type'] = (df_merged['loan_type'].astype(str).str.lower().str.strip()
                          .str.replace(r'personal.*', 'personal', regex=True)
                          .str.replace(r'(mortgage|home loan)', 'mortgage', regex=True)
                          .str.replace(r'(credit ?card|cc)', 'credit_card', regex=True))

df_merged['employment_type'] = df_merged['employment_type'].str.upper().replace({
    'FULL-TIME': 'FULLTIME', 'FULL_TIME': 'FULLTIME', 'FULL TIME': 'FULLTIME', 'FULLTIME': 'FULLTIME', 'FT': 'FULLTIME',
    'PART TIME': 'PARTTIME', 'PART_TIME': 'PARTTIME', 'PT': 'PARTTIME', 'PART-TIME': 'PARTTIME',
    'SELF EMPLOYED': 'SELFEMPLOYED', 'SELF EMP': 'SELFEMPLOYED', 'SELF-EMPLOYED': 'SELFEMPLOYED',
    'SELF_EMPLOYED': 'SELFEMPLOYED',
})

df_merged['education'] = df_merged['education'].str.strip().str.title()
df_merged['marital_status'] = df_merged['marital_status'].str.strip().str.title()

print("Data cleaned")

# Feature selection
print("\n[5/7] Feature selection...")

# Remove highly correlated features (>0.90)
numeric_df = df_merged.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

columns_to_drop_corr = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.90:
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            if col1 != 'default' and col2 != 'default':
                corr_target = numeric_df.corr()['default'].abs()
                if corr_target[col1] < corr_target[col2]:
                    columns_to_drop_corr.add(col1)
                else:
                    columns_to_drop_corr.add(col2)

if columns_to_drop_corr:
    print(f"Dropping {len(columns_to_drop_corr)} highly correlated features")
    df_merged = df_merged.drop(columns=list(columns_to_drop_corr), axis=1)

# Drop low correlation features (<0.05)
correlation = df_merged.select_dtypes(include=[np.number]).corr()['default'].abs()
low_corr_features = [col for col in correlation.index if correlation[col] < 0.05 and col != 'default']
if low_corr_features:
    print(f"Dropping {len(low_corr_features)} low-correlation features")
    df_merged = df_merged.drop(columns=low_corr_features, axis=1, errors='ignore')

# Feature engineering
print("\n[6/7] Creating interaction features...")

if 'existing_monthly_debt' in df_merged.columns and 'monthly_payment' in df_merged.columns:
    print("Creating feature: debt_burden")
    df_merged['debt_burden'] = (df_merged['existing_monthly_debt'] + df_merged['monthly_payment']) / (
                df_merged['annual_income'] / 12 + 1)
if 'credit_utilization' in df_merged.columns and 'debt_to_income_ratio' in df_merged.columns:
    print("Creating feature: credit_pressure")
    df_merged['credit_pressure'] = df_merged['credit_utilization'] * df_merged['debt_to_income_ratio']
if 'monthly_payment' in df_merged.columns and 'monthly_free_cash_flow' in df_merged.columns:
    print("Creating feature: payment_stress")
    df_merged['payment_stress'] = df_merged['monthly_payment'] / (df_merged['monthly_free_cash_flow'] + 1)
if 'credit_score' in df_merged.columns and 'credit_utilization' in df_merged.columns:
    print("Creating feature: credit_efficiency")
    df_merged['credit_efficiency'] = df_merged['credit_score'] / (df_merged['credit_utilization'] + 0.01)
if 'num_delinquencies_2yrs' in df_merged.columns and 'debt_to_income_ratio' in df_merged.columns:
    print("Creating feature: delinquency_severity")
    df_merged['delinquency_severity'] = df_merged['num_delinquencies_2yrs'] * df_merged['debt_to_income_ratio']

categorical_features_to_drop = [
    'referral_code',
    'marketing_campaign',
    'employment_type',
    'account_status_code',
    'preferred_contact',
]

for col in categorical_features_to_drop:
    if col in df_merged.columns:
        df_merged.drop(columns=[col], inplace=True)

# Replace inf values
for col in df_merged.select_dtypes(include=[np.number]).columns:
    df_merged[col] = df_merged[col].replace([np.inf, -np.inf], np.nan)
    df_merged[col] = df_merged[col].fillna(df_merged[col].median())

print("Created 6 interaction features")

# Prepare features
X = df_merged.drop('default', axis=1)
print(X.info())
y = df_merged['default']

# Identify categorical features (CatBoost handles them natively)
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

print(f"\nDataset prepared:")
print(f"  Total features: {X.shape[1]}")
print(f"  Categorical: {len(categorical_features)}")
print(
    f"  Target: 0={y.value_counts()[0]:,} ({y.value_counts()[0] / len(y) * 100:.1f}%) | 1={y.value_counts()[1]:,} ({y.value_counts()[1] / len(y) * 100:.1f}%)")

# Build CatBoost model
print("\n[7/7] Training CatBoost model...")
print("Configuration:")
print("  Iterations: 500")
print("  Learning rate: 0.05")
print("  Depth: 8")
print("  Class weights: Balanced (auto)")

catboost_model = CatBoostClassifier(
    iterations=700,
    learning_rate=0.015,
    depth=8,
    l2_leaf_reg=6,
    auto_class_weights='SqrtBalanced',  # HANDLE IMBALANCED DATA
    early_stopping_rounds=50,
    eval_metric='AUC',
    cat_features=categorical_indices,
    task_type='CPU',
    random_seed=42,
    verbose=False
)

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_start = time.time()

auc_scores = []
accuracy_scores = []
all_y_true = []
all_y_pred = []
all_y_pred_proba = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"  Fold {fold}/3...", end=' ', flush=True)

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=categorical_indices)
    val_pool = Pool(X_val, y_val, cat_features=categorical_indices)

    model = catboost_model.fit(train_pool, eval_set=val_pool, verbose=False)

    y_pred_proba = model.predict_proba(val_pool)[:, 1]
    y_pred = model.predict(val_pool)

    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = (y_pred == y_val).mean()

    auc_scores.append(auc)
    accuracy_scores.append(accuracy)
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_pred_proba.extend(y_pred_proba)

    print(f"AUC: {auc:.4f}")

cv_time = time.time() - cv_start

# Calculate metrics
overall_auc = roc_auc_score(all_y_true, all_y_pred_proba)
conf_matrix = confusion_matrix(all_y_true, all_y_pred)

# Find optimal F1 threshold
print("\\nFinding optimal F1 threshold...")
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []
for threshold in thresholds:
    preds = (np.array(all_y_pred_proba) >= threshold).astype(int)
    f1_scores.append(f1_score(all_y_true, preds))

optimal_threshold = thresholds[np.argmax(f1_scores)]
optimal_f1 = max(f1_scores)
print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.4f})")

# Display results
print(f"\n{'=' * 60}")
print(" CATBOOST MODEL PERFORMANCE")
print(f"{'=' * 60}")

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

# Train final model
print("\nTraining final model on full dataset...")
final_pool = Pool(X, y, cat_features=categorical_indices)
final_model = catboost_model.fit(final_pool, verbose=False)

# Feature importance
print("\nTop 15 Most Important Features:")
feature_importance = final_model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:35s}: {row['importance']:.2f}")

print(f"\nPerformance:")
print(f"  Training time: {cv_time:.2f}s")
print(f"  Total runtime: {time.time() - start_time:.2f}s")
print(f"{'=' * 60}")

print("\nCATBOOST MODEL READY!")

# ============================================================
# LOAD AND PREDICT ON TEST DATA
# ============================================================
print("\n" + "=" * 60)
print(" GENERATING PREDICTIONS ON TEST DATA")
print("=" * 60)

print("\nLoading test data from ./data/tests/...")
test_start_time = time.time()

df_test_financial_ratios = pd.read_json('./data/tests/financial_ratios.jsonl', lines=True)
df_test_geographic_data = pd.read_xml('./data/tests/geographic_data.xml')
df_test_credit_history = pd.read_parquet('./data/tests/credit_history.parquet')
df_test_demographics = pd.read_csv('./data/tests/demographics.csv')
df_test_loan_details = pd.read_excel('./data/tests/loan_details.xlsx')
df_test_application_metadata = pd.read_csv('./data/tests/application_metadata.csv')

print(f"Loaded test data in {time.time() - test_start_time:.2f}s")
print(f"  financial_ratios: {len(df_test_financial_ratios)} rows")
print(f"  geographic_data: {len(df_test_geographic_data)} rows")
print(f"  credit_history: {len(df_test_credit_history)} rows")
print(f"  demographics: {len(df_test_demographics)} rows")
print(f"  loan_details: {len(df_test_loan_details)} rows")
print(f"  application_metadata: {len(df_test_application_metadata)} rows")

# Standardize column names
df_test_financial_ratios.rename(columns={'cust_num': 'customer_id'}, inplace=True)
df_test_demographics.rename(columns={'cust_id': 'customer_id'}, inplace=True)
df_test_credit_history.rename(columns={'customer_number': 'customer_id'}, inplace=True)
df_test_application_metadata.rename(columns={'customer_ref': 'customer_id'}, inplace=True)
df_test_geographic_data.rename(columns={'id': 'customer_id'}, inplace=True)

# Merge test datasets
print("Merging test datasets...")
df_test_merged = pd.merge(df_test_financial_ratios, df_test_demographics, on='customer_id', how='outer')
df_test_merged = pd.merge(df_test_merged, df_test_credit_history, on='customer_id', how='outer')
df_test_merged = pd.merge(df_test_merged, df_test_loan_details, on='customer_id', how='outer')
df_test_merged = pd.merge(df_test_merged, df_test_application_metadata, on='customer_id', how='outer')
df_test_merged = pd.merge(df_test_merged, df_test_geographic_data, on='customer_id', how='outer')

# Preserve customer_id for output
customer_ids = df_test_merged['customer_id'].copy()

# Apply same transformations as training data
print("Applying transformations...")

# Clean currency columns
for col in currency_columns:
    if col in df_test_merged.columns:
        df_test_merged[col] = df_test_merged[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df_test_merged[col] = pd.to_numeric(df_test_merged[col], errors='coerce')

# Remove outliers (same bounds as training)
for col in numeric_cols:
    if col != 'default' and col in df_test_merged.columns and col in df_merged.columns:
        Q1, Q3 = df_merged[col].quantile(0.25), df_merged[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 3 * IQR, Q3 + 3 * IQR
        df_test_merged[col] = df_test_merged[col].clip(lower=lower_bound, upper=upper_bound)

# Drop same columns as training
df_test_merged = df_test_merged.drop(columns=columns_to_drop, axis=1, errors='ignore')

# Fill missing values
if 'num_delinquencies_2yrs' in df_test_merged.columns:
    df_test_merged['num_delinquencies_2yrs'] = df_test_merged['num_delinquencies_2yrs'].fillna(0)
for col in df_test_merged.select_dtypes(include=[np.number]).columns:
    if df_test_merged[col].isnull().sum() > 0:
        df_test_merged[col].fillna(df_test_merged[col].median(), inplace=True)

# Standardize categorical columns
if 'loan_type' in df_test_merged.columns:
    df_test_merged['loan_type'] = (df_test_merged['loan_type'].astype(str).str.lower().str.strip()
                              .str.replace(r'personal.*', 'personal', regex=True)
                              .str.replace(r'(mortgage|home loan)', 'mortgage', regex=True)
                              .str.replace(r'(credit ?card|cc)', 'credit_card', regex=True))

if 'employment_type' in df_test_merged.columns:
    df_test_merged['employment_type'] = df_test_merged['employment_type'].str.upper().replace({
        'FULL-TIME': 'FULLTIME', 'FULL_TIME': 'FULLTIME', 'FULL TIME': 'FULLTIME', 'FULLTIME': 'FULLTIME', 'FT': 'FULLTIME',
        'PART TIME': 'PARTTIME', 'PART_TIME': 'PARTTIME', 'PT': 'PARTTIME', 'PART-TIME': 'PARTTIME',
        'SELF EMPLOYED': 'SELFEMPLOYED', 'SELF EMP': 'SELFEMPLOYED', 'SELF-EMPLOYED': 'SELFEMPLOYED',
        'SELF_EMPLOYED': 'SELFEMPLOYED',
    })

if 'education' in df_test_merged.columns:
    df_test_merged['education'] = df_test_merged['education'].str.strip().str.title()
if 'marital_status' in df_test_merged.columns:
    df_test_merged['marital_status'] = df_test_merged['marital_status'].str.strip().str.title()

# Drop highly correlated and low correlation features (same as training)
df_test_merged = df_test_merged.drop(columns=list(columns_to_drop_corr), axis=1, errors='ignore')
df_test_merged = df_test_merged.drop(columns=low_corr_features, axis=1, errors='ignore')

# Feature engineering (same as training)
if 'existing_monthly_debt' in df_test_merged.columns and 'monthly_payment' in df_test_merged.columns:
    df_test_merged['debt_burden'] = (df_test_merged['existing_monthly_debt'] + df_test_merged['monthly_payment']) / (
                df_test_merged['annual_income'] / 12 + 1)
if 'credit_utilization' in df_test_merged.columns and 'debt_to_income_ratio' in df_test_merged.columns:
    df_test_merged['credit_pressure'] = df_test_merged['credit_utilization'] * df_test_merged['debt_to_income_ratio']
if 'monthly_payment' in df_test_merged.columns and 'monthly_free_cash_flow' in df_test_merged.columns:
    df_test_merged['payment_stress'] = df_test_merged['monthly_payment'] / (df_test_merged['monthly_free_cash_flow'] + 1)
if 'credit_score' in df_test_merged.columns and 'credit_utilization' in df_test_merged.columns:
    df_test_merged['credit_efficiency'] = df_test_merged['credit_score'] / (df_test_merged['credit_utilization'] + 0.01)
if 'num_delinquencies_2yrs' in df_test_merged.columns and 'debt_to_income_ratio' in df_test_merged.columns:
    df_test_merged['delinquency_severity'] = df_test_merged['num_delinquencies_2yrs'] * df_test_merged['debt_to_income_ratio']

# Drop categorical features
for col in categorical_features_to_drop:
    if col in df_test_merged.columns:
        df_test_merged.drop(columns=[col], inplace=True)

# Replace inf values
for col in df_test_merged.select_dtypes(include=[np.number]).columns:
    df_test_merged[col] = df_test_merged[col].replace([np.inf, -np.inf], np.nan)
    df_test_merged[col] = df_test_merged[col].fillna(df_test_merged[col].median())

# Drop target if exists in test data
if 'default' in df_test_merged.columns:
    df_test_merged = df_test_merged.drop('default', axis=1)

# Align test data features with training data
X_test = df_test_merged[X.columns]

print(f"Test data shape: {X_test.shape}")

# Make predictions
print("\nGenerating predictions...")
test_pool = Pool(X_test, cat_features=categorical_indices)
pred_probabilities = final_model.predict_proba(test_pool)[:, 1]
pred_verdicts = (pred_probabilities >= optimal_threshold).astype(int)

# Create results dataframe
results_df = pd.DataFrame({
    'customer_id': customer_ids,
    'predicted_probability': pred_probabilities,
    'verdict': pred_verdicts
})

# Save results
output_path = './results.csv'
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to {output_path}")
print(f"Total predictions: {len(results_df):,}")
print(f"Predicted defaults: {pred_verdicts.sum():,} ({pred_verdicts.sum()/len(pred_verdicts)*100:.1f}%)")
print(f"Predicted non-defaults: {(1-pred_verdicts).sum():,} ({(1-pred_verdicts).sum()/len(pred_verdicts)*100:.1f}%)")
print(f"\nPrediction time: {time.time() - test_start_time:.2f}s")
print("=" * 60)
