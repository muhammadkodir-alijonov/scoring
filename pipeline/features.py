"""
Feature engineering and preprocessing pipeline
Handles data cleaning, feature creation, and transformation
"""

import pandas as pd
import numpy as np


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
# PIPELINE ORCHESTRATION
# ============================================================================

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
