import pandas as pd
import numpy as np
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print(" CREDIT DEFAULT PREDICTION MODEL")
print("="*60)

# ==================== DATA LOADING ====================
print("\n[1/7] Loading data files...")
start_time = time.time()

df_financial_ratios = pd.read_json('financial_ratios.jsonl', lines=True)
df_geographic_data = pd.read_xml('geographic_data.xml')
df_credit_history = pd.read_parquet('credit_history.parquet')
df_demographics = pd.read_csv('demographics.csv')
df_loan_details = pd.read_excel('loan_details.xlsx')
df_application_metadata = pd.read_csv('application_metadata.csv')

print(f"Loaded 6 files in {time.time() - start_time:.2f}s")

# ==================== STANDARDIZE COLUMN NAMES ====================
print("\n[2/7] Standardizing column names...")

df_financial_ratios.rename(columns={'cust_num': 'customer_id'}, inplace=True)
df_demographics.rename(columns={'cust_id': 'customer_id'}, inplace=True)
df_credit_history.rename(columns={'customer_number': 'customer_id'}, inplace=True)
df_application_metadata.rename(columns={'customer_ref': 'customer_id'}, inplace=True)
df_geographic_data.rename(columns={'id': 'customer_id'}, inplace=True)

print("Column names standardized")

# ==================== MERGE DATASETS ====================
print("\n[3/7] Merging datasets...")

df_merged = pd.merge(df_financial_ratios, df_demographics, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_credit_history, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_loan_details, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_application_metadata, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_geographic_data, on='customer_id', how='outer')

print(f"Merged shape: {df_merged.shape}")

# ==================== DATA CLEANING ====================
print("\n[4/7] Cleaning data...")

# Clean currency columns
currency_columns = [
   'monthly_income', 'existing_monthly_debt', 'monthly_payment',
   'revolving_balance', 'credit_usage_amount', 'available_credit',
   'total_monthly_debt_payment', 'total_debt_amount',
   'monthly_free_cash_flow', 'annual_income', 'loan_amount'
]

for col in currency_columns:
   df_merged[col] = df_merged[col].astype(str)
   df_merged[col] = df_merged[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
   df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

print(f"Cleaned {len(currency_columns)} currency columns")

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

print(f"Clipped {outliers_removed:,} outliers")

# Drop redundant columns
df_merged.drop(columns=['revolving_balance', 'oldest_credit_line_age', 'recent_inquiry_count'], 
               inplace=True, errors='ignore')

# Fill missing values
df_merged['num_delinquencies_2yrs'] = df_merged['num_delinquencies_2yrs'].fillna(0)
df_merged['employment_length'] = df_merged.groupby(
    ['education', 'employment_type', 'marital_status']
)['employment_length'].transform(lambda x: x.fillna(x.median()))

for col in df_merged.select_dtypes(include=[np.number]).columns:
    if df_merged[col].isnull().sum() > 0:
        df_merged[col].fillna(df_merged[col].median(), inplace=True)

# Standardize categorical columns
df_merged['loan_type'] = (df_merged['loan_type'].astype(str).str.lower().str.strip()
    .str.replace(r'personal.*', 'personal', regex=True)
    .str.replace(r'(mortgage|home loan)', 'mortgage', regex=True)
    .str.replace(r'(credit ?card|cc)', 'credit_card', regex=True))

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

print("Data cleaning completed")

# ==================== FEATURE SELECTION ====================
print("\n[5/7] Selecting features...")

# Drop unnecessary columns
columns_to_drop = [
    'cost_of_living_index', 'regional_unemployment_rate', 'housing_price_index',
    'random_noise_1', 'regional_median_income', 'regional_median_rent',
    'loan_officer_id', 'application_id', 'num_customer_service_calls',
    'num_inquiries_6mo', 'application_hour', 'account_open_year',
    'num_collections', 'previous_zip_code', 'customer_id', 'application_day_of_week'
]
df_merged = df_merged.drop(columns=columns_to_drop, axis=1, errors='ignore')

# Remove highly correlated features (>0.90)
numeric_df = df_merged.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.90:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            if col1 != 'default' and col2 != 'default':
                corr_target = numeric_df.corr()['default'].abs()
                if corr_target[col1] < corr_target[col2]:
                    high_corr_features.add(col1)
                else:
                    high_corr_features.add(col2)

if high_corr_features:
    print(f"Dropping {len(high_corr_features)} highly correlated features")
    df_merged = df_merged.drop(columns=list(high_corr_features), axis=1)

# Drop low correlation features (<0.02)
correlation = df_merged.select_dtypes(include=[np.number]).corr()['default'].abs()
low_corr_features = [col for col in correlation.index if correlation[col] < 0.02 and col != 'default']
if low_corr_features:
    print(f"Dropping {len(low_corr_features)} low-correlation features")
    df_merged = df_merged.drop(columns=low_corr_features, axis=1, errors='ignore')

print(f"Features after selection: {df_merged.shape[1] - 1}")

# ==================== FEATURE ENGINEERING ====================
print("\n[6/7] Creating interaction features...")

X = df_merged.drop('default', axis=1)
y = df_merged['default']

# Financial stress indicators
X['debt_burden'] = (df_merged['existing_monthly_debt'] + df_merged['monthly_payment']) / (df_merged['annual_income'] / 12 + 1)
X['credit_pressure'] = df_merged['credit_utilization'] * df_merged['debt_to_income_ratio']
X['payment_stress'] = df_merged['monthly_payment'] / (df_merged['monthly_free_cash_flow'] + 1)

# Credit behavior
X['credit_efficiency'] = df_merged['credit_score'] / (df_merged['credit_utilization'] + 0.01)
X['account_maturity'] = df_merged['oldest_account_age_months'] / (df_merged['num_credit_accounts'] + 1)
X['delinquency_severity'] = df_merged['num_delinquencies_2yrs'] * df_merged['debt_to_income_ratio']

# Income stability
X['income_stability'] = df_merged['annual_income'] * df_merged['employment_length'] / (df_merged['age'] + 1)
X['income_age_ratio'] = df_merged['annual_income'] / (df_merged['age'] + 1)

# Credit usage
X['credit_usage_ratio'] = df_merged['credit_usage_amount'] / (df_merged['total_credit_limit'] + 1)
X['available_credit_ratio'] = df_merged['available_credit'] / (df_merged['total_credit_limit'] + 1)

# Clean interaction features
interaction_cols = ['debt_burden', 'credit_pressure', 'payment_stress', 'credit_efficiency', 
                    'account_maturity', 'delinquency_severity', 'income_stability', 
                    'income_age_ratio', 'credit_usage_ratio', 'available_credit_ratio']

for col in interaction_cols:
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    X[col] = X[col].fillna(X[col].median())
    X[col] = X[col].clip(lower=X[col].quantile(0.01), upper=X[col].quantile(0.99))

print(f"Created {len(interaction_cols)} interaction features")

# Separate features
numerical_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Final features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
print(f"Dataset: {len(X)} samples, Target: {y.value_counts()[0]:,} No Default, {y.value_counts()[1]:,} Default")

# ==================== MODEL TRAINING ====================
print("\n[7/7] Training model...")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20), categorical_features)
    ])

# Model
model = HistGradientBoostingClassifier(
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

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Cross-validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_start = time.time()

y_pred_proba = cross_val_predict(pipeline, X, y, cv=kf, method='predict_proba', n_jobs=-1)[:, 1]
y_pred = cross_val_predict(pipeline, X, y, cv=kf, n_jobs=-1)
cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy', n_jobs=-1)

auc_score = roc_auc_score(y, y_pred_proba)
conf_matrix = confusion_matrix(y, y_pred)
cv_time = time.time() - cv_start

# ==================== RESULTS ====================
print(f"\n{'='*60}")
print(" MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"AUC-ROC Score: {auc_score:.4f}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {conf_matrix[0][0]:,}")
print(f"  False Positives: {conf_matrix[0][1]:,}")
print(f"  False Negatives: {conf_matrix[1][0]:,}")
print(f"  True Positives:  {conf_matrix[1][1]:,}")

print(f"\nClassification Report:")
print(classification_report(y, y_pred, target_names=['No Default', 'Default']))

print(f"\nTraining time: {cv_time:.2f}s")
print(f"Total runtime: {time.time() - start_time:.2f}s")
print(f"{'='*60}")

# Train final model
print("\nTraining final model on full dataset...")
final_model = pipeline.fit(X, y)
print("Model trained successfully!")

print("\nMODEL READY FOR PREDICTIONS!")
