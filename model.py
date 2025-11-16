import pandas as pd
import numpy as np

# Load the 'financial_ratios.jsonl' file into a DataFrame
print("Loading data files...")
df_financial_ratios = pd.read_json('financial_ratios.jsonl', lines=True)
print(f"✓ financial_ratios.jsonl loaded: {df_financial_ratios.shape}")
df_geographic_data = pd.read_xml('geographic_data.xml')
print(f"✓ geographic_data.xml loaded: {df_geographic_data.shape}")
df_credit_history = pd.read_parquet('credit_history.parquet')
print(f"✓ credit_history.parquet loaded: {df_credit_history.shape}")
df_demographics = pd.read_csv('demographics.csv')
print(f"✓ demographics.csv loaded: {df_demographics.shape}")
df_loan_details = pd.read_excel('loan_details.xlsx')
print(f"✓ loan_details.xlsx loaded: {df_loan_details.shape}")
df_application_metadata = pd.read_csv('application_metadata.csv')
print(f"✓ application_metadata.csv loaded: {df_application_metadata.shape}")
df_financial_ratios.rename(columns={'cust_num': 'customer_id'}, inplace=True)
df_demographics.rename(columns={'cust_id': 'customer_id'}, inplace=True)
df_credit_history.rename(columns={'customer_number': 'customer_id'}, inplace=True)
df_application_metadata.rename(columns={'customer_ref': 'customer_id'}, inplace=True)
df_geographic_data.rename(columns={'id': 'customer_id'}, inplace=True)

print("✓ Customer ID columns renamed successfully")
# Merge all dataframes
print("\n=== MERGING DATASETS ===")
df_merged = pd.merge(df_financial_ratios, df_demographics, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_credit_history, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_loan_details, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_application_metadata, on='customer_id', how='outer')
df_merged = pd.merge(df_merged, df_geographic_data, on='customer_id', how='outer')
print(f"✓ All datasets merged: {df_merged.shape}")
columns_to_clean = [
   'monthly_income',
   'existing_monthly_debt',
   'monthly_payment',
   'revolving_balance',
   'credit_usage_amount',
   'available_credit',
   'total_monthly_debt_payment',
   'total_debt_amount',
   'monthly_free_cash_flow',
   'annual_income',
   'loan_amount',
]


print("\n=== CLEANING DATA ===")
for col in columns_to_clean:
   df_merged[col] = df_merged[col].astype(str)
   df_merged[col] = df_merged[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
   df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
print(f"✓ Cleaned {len(columns_to_clean)} numeric columns")
columns_to_drop = [
    'revolving_balance',
    'oldest_credit_line_age',
    'recent_inquiry_count',
]
df_merged.drop(columns=columns_to_drop, inplace=True)
print(f"✓ Dropped {len(columns_to_drop)} columns")
df_merged['num_delinquencies_2yrs'] = df_merged['num_delinquencies_2yrs'].fillna(0)
df_merged['loan_type'] = (
   df_merged['loan_type']
       .astype(str) # check qilish kerak
       .str.lower()
       .str.strip()
       .str.replace(r'personal.*', 'personal', regex=True)
       .str.replace(r'(mortgage|home loan)', 'mortgage', regex=True)
       .str.replace(r'(credit ?card|cc)', 'credit card', regex=True)
)
df_merged['employment_type'] = df_merged['employment_type'].str.upper().replace({
    'FULL-TIME': 'FULLTIME',
    'FULL_TIME': 'FULLTIME',
    'FULL TIME': 'FULLTIME',
    'FULLTIME': 'FULLTIME',
    'FT': 'FULLTIME',
    'PART TIME': 'PARTTIME',
    'PART_TIME': 'PARTTIME',
    'PT': 'PARTTIME',
    'PART-TIME': 'PARTTIME',
    'SELF EMPLOYED': 'SELFEMPLOYED',
    'SELF EMP': 'SELFEMPLOYED',
    'SELF-EMPLOYED': 'SELFEMPLOYED',
    'SELF_EMPLOYED': 'SELFEMPLOYED',
    'CONTRACTOR': 'CONTRACT',
    'CONTRACT': 'CONTRACT'
})
# employment_length null qiymatlari 3 ta  calonka bo’yicha guruhlanim medianasi olindi.
df_merged['employment_length'] = df_merged.groupby(['education', 'employment_type', 'marital_status'])['employment_length'].transform(lambda x: x.fillna(x.median()))
print("\n=== DATA ANALYSIS ===")
print(f"Target distribution:\n{df_merged['default'].value_counts()}")
non_numeric_cols = df_merged.select_dtypes(include=['object', 'category']).columns.tolist()
exclude_cols = ['customer_id', 'application_id']
categorical_cols = [col for col in non_numeric_cols if col not in exclude_cols]
print(f"✓ Identified {len(categorical_cols)} categorical columns")
columns_to_drop_final = [
    'cost_of_living_index', 'regional_unemployment_rate', 'housing_price_index',
    'random_noise_1', 'regional_median_income', 'regional_median_rent',
    'loan_officer_id', 'application_id', 'num_customer_service_calls',
    'num_inquiries_6mo', 'application_hour', 'account_open_year',
    'num_collections', 'previous_zip_code', 'customer_id', 'application_day_of_week'
]
df_merged = df_merged.drop(columns=columns_to_drop_final, axis=1)
print(f"✓ Dropped {len(columns_to_drop_final)} unnecessary columns")
print(f"✓ Final dataset shape: {df_merged.shape}")

print("\n=== PREPARING FEATURES ===")
X = df_merged.drop('default', axis=1)
y = df_merged['default']

numerical_features = df_merged.select_dtypes(include=['number']).columns.tolist()
categorical_features = df_merged.select_dtypes(include=['object', 'category']).columns.tolist()

if 'default' in numerical_features:
    numerical_features.remove('default')

print(f"✓ Numerical features: {len(numerical_features)}")
print(f"✓ Categorical features: {len(categorical_features)}")
print(f"✓ Total samples: {len(X)}" )
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import time

print("\n=== BUILDING MODEL ===")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Optimized Logistic Regression
logistic_model = LogisticRegression(
    random_state=42, 
    max_iter=2000,
    solver='saga',     # Faster for large datasets
    n_jobs=-1,         # Use all CPU cores
    warm_start=True    # Reuse solution from previous call
)

# Complete pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logistic_model)
])

print("✓ Pipeline created successfully")

# Cross-validation with 3 splits (faster than 5)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

print("\n=== TRAINING MODEL ===")
print("This may take 1-2 minutes...")
start_time = time.time()

cv_scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)

end_time = time.time()
print(f"\n✓ Training completed in {end_time - start_time:.2f} seconds")
print(f"✓ Cross-validation scores: {cv_scores}")
print(f"✓ Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print("\n=== MODEL TRAINING COMPLETE ===")