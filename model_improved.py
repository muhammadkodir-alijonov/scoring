import pandas as pd
import numpy as np
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print(" CREDIT DEFAULT PREDICTION MODEL - OPTIMIZED VERSION")
print("="*60)

# Loading data files
print("\n[1/7] Loading data files...")
start_time = time.time()

df_financial_ratios = pd.read_json('financial_ratios.jsonl', lines=True)
df_geographic_data = pd.read_xml('geographic_data.xml')
df_credit_history = pd.read_parquet('credit_history.parquet')
df_demographics = pd.read_csv('demographics.csv')
df_loan_details = pd.read_excel('loan_details.xlsx')
df_application_metadata = pd.read_csv('application_metadata.csv')

print(f"Loaded 6 files in {time.time() - start_time:.2f}s")
print(f"  - financial_ratios: {df_financial_ratios.shape}")
print(f"  - demographics: {df_demographics.shape}")
print(f"  - credit_history: {df_credit_history.shape}")

# Standardizing column names
print("\n[2/7] Standardizing customer ID columns...")
df_financial_ratios.rename(columns={'cust_num': 'customer_id'}, inplace=True)
df_demographics.rename(columns={'cust_id': 'customer_id'}, inplace=True)
df_credit_history.rename(columns={'customer_number': 'customer_id'}, inplace=True)
df_application_metadata.rename(columns={'customer_ref': 'customer_id'}, inplace=True)
df_geographic_data.rename(columns={'id': 'customer_id'}, inplace=True)
print("Customer ID columns standardized")

# Merging datasets
print("\n[3/7] Merging datasets...")
df_merged = pd.merge(df_financial_ratios, df_demographics, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_credit_history, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_loan_details, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_application_metadata, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_geographic_data, on='customer_id', how='outer')
print(f"Merged shape: {df_merged.shape}")
print(f"Missing values: {df_merged.isnull().sum().sum():,}")

# Data cleaning
print("\n[4/7] Cleaning data...")

# 4.1 Clean currency columns
columns_to_clean = [
   'monthly_income', 'existing_monthly_debt', 'monthly_payment',
   'revolving_balance', 'credit_usage_amount', 'available_credit',
   'total_monthly_debt_payment', 'total_debt_amount',
   'monthly_free_cash_flow', 'annual_income', 'loan_amount'
]

for col in columns_to_clean:
   df_merged[col] = df_merged[col].astype(str)
   df_merged[col] = df_merged[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
   df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
print(f"Cleaned {len(columns_to_clean)} currency columns")

# Remove outliers using IQR method
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
outliers_removed = 0
for col in numeric_cols:
    if col != 'default':
        Q1 = df_merged[col].quantile(0.25)
        Q3 = df_merged[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers_before = ((df_merged[col] < lower_bound) | (df_merged[col] > upper_bound)).sum()
        df_merged[col] = df_merged[col].clip(lower=lower_bound, upper=upper_bound)
        outliers_removed += outliers_before
print(f"Clipped {outliers_removed:,} outlier values")

# Drop redundant columns
columns_to_drop = [
    'revolving_balance',
    'oldest_credit_line_age',
    'recent_inquiry_count',
]
df_merged.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(columns_to_drop)} redundant columns")

# Fill missing values
df_merged['num_delinquencies_2yrs'] = df_merged['num_delinquencies_2yrs'].fillna(0)
df_merged['employment_length'] = df_merged.groupby(
    ['education', 'employment_type', 'marital_status']
)['employment_length'].transform(lambda x: x.fillna(x.median()))

for col in df_merged.select_dtypes(include=[np.number]).columns:
    if df_merged[col].isnull().sum() > 0:
        df_merged[col].fillna(df_merged[col].median(), inplace=True)
print(f"Missing values filled")

# 4.5 Standardize categorical columns
df_merged['loan_type'] = (
   df_merged['loan_type']
       .astype(str).str.lower().str.strip()
       .str.replace(r'personal.*', 'personal', regex=True)
       .str.replace(r'(mortgage|home loan)', 'mortgage', regex=True)
       .str.replace(r'(credit ?card|cc)', 'credit_card', regex=True)
)

df_merged['employment_type'] = df_merged['employment_type'].str.upper().replace({
    'FULL-TIME': 'FULLTIME', 'FULL_TIME': 'FULLTIME', 'FULL TIME': 'FULLTIME',
    'FULLTIME': 'FULLTIME', 'FT': 'FULLTIME',
    'PART TIME': 'PARTTIME', 'PART_TIME': 'PARTTIME', 'PT': 'PARTTIME', 'PART-TIME': 'PARTTIME',
    'SELF EMPLOYED': 'SELFEMPLOYED', 'SELF EMP': 'SELFEMPLOYED', 
    'SELF-EMPLOYED': 'SELFEMPLOYED', 'SELF_EMPLOYED': 'SELFEMPLOYED',
    'CONTRACTOR': 'CONTRACT', 'CONTRACT': 'CONTRACT'
})

df_merged['education'] = df_merged['education'].str.strip().str.title()
df_merged['marital_status'] = df_merged['marital_status'].str.strip().str.title()
print(f"Categorical columns standardized")

# Feature selection
print("\n[5/7] Selecting features...")

columns_to_drop_final = [
    'cost_of_living_index', 'regional_unemployment_rate', 'housing_price_index',
    'random_noise_1', 'regional_median_income', 'regional_median_rent',
    'loan_officer_id', 'application_id', 'num_customer_service_calls',
    'num_inquiries_6mo', 'application_hour', 'account_open_year',
    'num_collections', 'previous_zip_code', 'customer_id', 'application_day_of_week'
]
df_merged = df_merged.drop(columns=columns_to_drop_final, axis=1, errors='ignore')
print(f"Dropped {len(columns_to_drop_final)} low-value columns")

# Analyze multicollinearity (high correlation between features)
print("\nAnalyzing feature correlations...")
numeric_df = df_merged.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

# Find features with correlation > 0.90 (stricter)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.90:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            high_corr_pairs.append((col1, col2, corr_val))

if high_corr_pairs:
    print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (>0.90):")
    columns_to_drop_corr = set()
    for col1, col2, corr_val in high_corr_pairs:
        print(f"  {col1} <-> {col2}: {corr_val:.4f}")
        # Keep the one with higher correlation to target
        if col1 != 'default' and col2 != 'default':
            corr_target = numeric_df.corr()['default'].abs()
            if corr_target[col1] < corr_target[col2]:
                columns_to_drop_corr.add(col1)
            else:
                columns_to_drop_corr.add(col2)
    
    if columns_to_drop_corr:
        print(f"\nDropping {len(columns_to_drop_corr)} redundant features:")
        for col in columns_to_drop_corr:
            print(f"  - {col}")
        df_merged = df_merged.drop(columns=list(columns_to_drop_corr), axis=1)
else:
    print("No highly correlated feature pairs found (>0.90)")

# Analyze correlation with target
print("\nFeature importance based on correlation with target:")
correlation = df_merged.select_dtypes(include=[np.number]).corr()['default'].abs().sort_values(ascending=False)

# Select only features with correlation > 0.07 (meaningful correlation)
important_features = correlation[correlation > 0.07].index.tolist()
important_features.remove('default')

print(f"\nTop features (correlation > 0.07):")
for i, col in enumerate(important_features[:20], 1):
    print(f"  {i:2d}. {col:35s}: {correlation[col]:.4f}")

# Drop low correlation features
low_corr_features = [col for col in correlation.index if correlation[col] < 0.07 and col != 'default']
if low_corr_features:
    print(f"\nDropping {len(low_corr_features)} low-correlation features (<0.07):")
    for col in low_corr_features[:10]:  # Show first 10
        print(f"  - {col}")
    if len(low_corr_features) > 10:
        print(f"  ... and {len(low_corr_features) - 10} more")
    df_merged = df_merged.drop(columns=low_corr_features, axis=1, errors='ignore')

# Prepare X and y
X = df_merged.drop('default', axis=1)
y = df_merged['default']

numerical_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Feature engineering: create powerful interaction features
print("\nCreating advanced interaction features...")

# Check available columns
available_cols = df_merged.columns.tolist()

# Financial stress indicators
if 'existing_monthly_debt' in available_cols and 'monthly_payment' in available_cols:
    X['debt_burden'] = (df_merged['existing_monthly_debt'] + df_merged['monthly_payment']) / (df_merged['annual_income'] / 12 + 1)
if 'credit_utilization' in available_cols and 'debt_to_income_ratio' in available_cols:
    X['credit_pressure'] = df_merged['credit_utilization'] * df_merged['debt_to_income_ratio']
if 'monthly_payment' in available_cols and 'monthly_free_cash_flow' in available_cols:
    X['payment_stress'] = df_merged['monthly_payment'] / (df_merged['monthly_free_cash_flow'] + 1)

# Credit behavior
if 'credit_score' in available_cols and 'credit_utilization' in available_cols:
    X['credit_efficiency'] = df_merged['credit_score'] / (df_merged['credit_utilization'] + 0.01)
if 'oldest_account_age_months' in available_cols and 'num_credit_accounts' in available_cols:
    X['account_maturity'] = df_merged['oldest_account_age_months'] / (df_merged['num_credit_accounts'] + 1)
if 'num_delinquencies_2yrs' in available_cols and 'debt_to_income_ratio' in available_cols:
    X['delinquency_severity'] = df_merged['num_delinquencies_2yrs'] * df_merged['debt_to_income_ratio']

# Income & employment stability
if 'annual_income' in available_cols and 'employment_length' in available_cols and 'age' in available_cols:
    X['income_stability'] = df_merged['annual_income'] * df_merged['employment_length'] / (df_merged['age'] + 1)
if 'annual_income' in available_cols and 'age' in available_cols:
    X['income_age_interaction'] = df_merged['annual_income'] / (df_merged['age'] + 1)

# Credit usage
if 'credit_usage_amount' in available_cols and 'total_credit_limit' in available_cols:
    X['credit_usage_ratio'] = df_merged['credit_usage_amount'] / (df_merged['total_credit_limit'] + 1)
if 'available_credit' in available_cols and 'total_credit_limit' in available_cols:
    X['available_credit_ratio'] = df_merged['available_credit'] / (df_merged['total_credit_limit'] + 1)

# Replace inf and very large values
interaction_cols = [col for col in X.columns if col not in numerical_features and col not in categorical_features]

for col in interaction_cols:
    if col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        X[col] = X[col].fillna(X[col].median())
        # Clip extreme values
        X[col] = X[col].clip(lower=X[col].quantile(0.01), upper=X[col].quantile(0.99))

numerical_features.extend(interaction_cols)
print(f"Created {len(interaction_cols)} advanced interaction features")

print(f"\nFinal dataset: {df_merged.shape}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Target distribution: 0={y.value_counts()[0]:,} | 1={y.value_counts()[1]:,}")

# Build model pipeline
print("\n[6/7] Building model pipeline...")

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10), categorical_features)
    ])

# Build optimized model
print("\nBuilding optimized HistGradientBoosting model...")

# HistGradient Boosting - highly optimized parameters for AUC
hgb_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_depth=10,
    max_leaf_nodes=50,
    min_samples_leaf=15,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=15,
    scoring='roc_auc',
    random_state=42,
    verbose=0
)

# Use StandardScaler (often better than RobustScaler for tree models)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20), categorical_features)
    ])

# Create pipeline WITHOUT SMOTE but with class_weight in model
# Adjust decision threshold instead
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', hgb_model)
])

print("Created optimized pipeline (no SMOTE, using early_stopping)")

# Train and evaluate
print("\n[7/7] Training model with cross-validation...")
print("This may take 1-2 minutes...")

# Use Stratified K-Fold for imbalanced data (3 splits for faster training)
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_start = time.time()

print("\nTraining HistGradientBoosting + SMOTE...")

# Get predictions
y_pred_proba = cross_val_predict(model_pipeline, X, y, cv=kf, method='predict_proba', n_jobs=-1)[:, 1]
y_pred = cross_val_predict(model_pipeline, X, y, cv=kf, n_jobs=-1)

# Calculate metrics
auc_score = roc_auc_score(y, y_pred_proba)
cv_scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
conf_matrix = confusion_matrix(y, y_pred)

cv_time = time.time() - cv_start

# Display results
print(f"\n{'='*60}")
print(" MODEL PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"\nModel: HistGradientBoosting + SMOTE")
print(f"\nCross-Validation Accuracy:")
print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean:  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

print(f"\nAUC-ROC Score: {auc_score:.4f}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {conf_matrix[0][0]:,}")
print(f"  False Positives: {conf_matrix[0][1]:,}")
print(f"  False Negatives: {conf_matrix[1][0]:,}")
print(f"  True Positives:  {conf_matrix[1][1]:,}")

print(f"\nClassification Report:")
report = classification_report(y, y_pred, target_names=['No Default (0)', 'Default (1)'])
print(report)

print(f"\nPerformance:")
print(f"  Training time: {cv_time:.2f} seconds")
print(f"  Total runtime: {time.time() - start_time:.2f} seconds")
print(f"{'='*60}")

# Final model training
print("\nTraining final model on full dataset...")
final_model = model_pipeline.fit(X, y)
print("Model trained successfully!")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
print(f"\nROC Curve Data:")
print(f"  False Positive Rate range: {fpr.min():.4f} - {fpr.max():.4f}")
print(f"  True Positive Rate range:  {tpr.min():.4f} - {tpr.max():.4f}")

print("\nMODEL READY FOR PREDICTIONS!")
