import pandas as pd
import numpy as np
import time
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print(" CREDIT DEFAULT PREDICTION - CATBOOST")
print("="*60)

# Loading data
start_time = time.time()

df_financial_ratios = pd.read_json('financial_ratios.jsonl', lines=True)
df_geographic_data = pd.read_xml('geographic_data.xml')
df_credit_history = pd.read_parquet('credit_history.parquet')
df_demographics = pd.read_csv('demographics.csv')
df_loan_details = pd.read_excel('loan_details.xlsx')
df_application_metadata = pd.read_csv('application_metadata.csv')

# Standardize column names
df_financial_ratios.rename(columns={'cust_num': 'customer_id'}, inplace=True)
df_demographics.rename(columns={'cust_id': 'customer_id'}, inplace=True)
df_credit_history.rename(columns={'customer_number': 'customer_id'}, inplace=True)
df_application_metadata.rename(columns={'customer_ref': 'customer_id'}, inplace=True)
df_geographic_data.rename(columns={'id': 'customer_id'}, inplace=True)

# Merge datasets
df_merged = pd.merge(df_financial_ratios, df_demographics, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_credit_history, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_loan_details, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_application_metadata, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_geographic_data, on='customer_id', how='outer')

# Data cleaning
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
    'SELF EMPLOYED': 'SELFEMPLOYED', 'SELF EMP': 'SELFEMPLOYED', 'SELF-EMPLOYED': 'SELFEMPLOYED', 'SELF_EMPLOYED': 'SELFEMPLOYED',
})

df_merged['education'] = df_merged['education'].str.strip().str.title()
df_merged['marital_status'] = df_merged['marital_status'].str.strip().str.title()

# Feature selection
# Remove highly correlated features (>0.90)
numeric_df = df_merged.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

columns_to_drop_corr = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.90:
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            if col1 != 'default' and col2 != 'default':
                corr_target = numeric_df.corr()['default'].abs()
                if corr_target[col1] < corr_target[col2]:
                    columns_to_drop_corr.add(col1)
                else:
                    columns_to_drop_corr.add(col2)

if columns_to_drop_corr:
    df_merged = df_merged.drop(columns=list(columns_to_drop_corr), axis=1)

# Drop low correlation features (<0.05)
correlation = df_merged.select_dtypes(include=[np.number]).corr()['default'].abs()
low_corr_features = [col for col in correlation.index if correlation[col] < 0.05 and col != 'default']
if low_corr_features:
    df_merged = df_merged.drop(columns=low_corr_features, axis=1, errors='ignore')

# Feature engineering
if 'existing_monthly_debt' in df_merged.columns and 'monthly_payment' in df_merged.columns:
    df_merged['debt_burden'] = (df_merged['existing_monthly_debt'] + df_merged['monthly_payment']) / (df_merged['annual_income'] / 12 + 1)
if 'credit_utilization' in df_merged.columns and 'debt_to_income_ratio' in df_merged.columns:
    df_merged['credit_pressure'] = df_merged['credit_utilization'] * df_merged['debt_to_income_ratio']
if 'monthly_payment' in df_merged.columns and 'monthly_free_cash_flow' in df_merged.columns:
    df_merged['payment_stress'] = df_merged['monthly_payment'] / (df_merged['monthly_free_cash_flow'] + 1)
if 'credit_score' in df_merged.columns and 'credit_utilization' in df_merged.columns:
    df_merged['credit_efficiency'] = df_merged['credit_score'] / (df_merged['credit_utilization'] + 0.01)
if 'num_delinquencies_2yrs' in df_merged.columns and 'debt_to_income_ratio' in df_merged.columns:
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

# Prepare features
X = df_merged.drop('default', axis=1)
y = df_merged['default']

# Identify categorical features (CatBoost handles them natively)
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
categorical_indices = [X.columns.get_loc(col) for col in categorical_features]


print(f"\nDataset prepared:")
print(f"  Total features: {X.shape[1]}")
print(f"  Categorical: {len(categorical_features)}")
print(f"  Target: 0={y.value_counts()[0]:,} ({y.value_counts()[0]/len(y)*100:.1f}%) | 1={y.value_counts()[1]:,} ({y.value_counts()[1]/len(y)*100:.1f}%)")

# Build CatBoost model
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
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_start = time.time()

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

cv_time = time.time() - cv_start

# Calculate metrics
overall_auc = roc_auc_score(all_y_true, all_y_pred_proba)
conf_matrix = confusion_matrix(all_y_true, all_y_pred)

# Display results
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

# Train final model
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
print(f"{'='*60}")

print("\nCATBOOST MODEL READY!")
