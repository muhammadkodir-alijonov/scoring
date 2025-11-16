"""
Feature Engineering Module
Prepares features for credit scoring model
"""
import pandas as pd
import numpy as np
from typing import List


def create_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Create features for credit scoring
    
    Args:
        df: Input DataFrame
        is_training: Whether this is training data (includes target)
        
    Returns:
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying original
    features = df.copy()
    
    # Extract customer_id and default (if training)
    customer_id = features['customer_id'].copy()
    target = None
    if is_training and 'default' in features.columns:
        target = features['default'].copy()
    
    # ========== YANGI FEATURE'LAR (IMPROVED) ==========
    
    # 1. Loan-to-Income Ratio (ENG MUHIM!)
    if 'loan_amount' in features.columns and 'monthly_income' in features.columns:
        features['loan_to_income_ratio'] = features['loan_amount'] / (features['monthly_income'] * 12 + 1)
        features['monthly_payment_burden'] = (features['loan_amount'] * 0.05) / (features['monthly_income'] + 1)
        features['income_loan_product'] = features['monthly_income'] * features['loan_amount'] / 1000000
    
    # 2. Debt-to-Income Ratio
    if 'total_debt' in features.columns and 'monthly_income' in features.columns:
        features['debt_to_income_ratio'] = features['total_debt'] / (features['monthly_income'] * 12 + 1)
        features['total_debt_log'] = np.log1p(features['total_debt'])
        features['debt_burden_score'] = features['total_debt'] / (features['monthly_income'] + 1)
    
    # 3. Credit Utilization (normalized)
    if 'credit_score' in features.columns:
        features['credit_score_normalized'] = features['credit_score'] / 850.0
        features['credit_risk_level'] = pd.cut(features['credit_score'], 
                                               bins=[0, 580, 670, 740, 800, 850],
                                               labels=[4, 3, 2, 1, 0]).astype(float)
        features['credit_score_squared'] = features['credit_score'] ** 2
        features['low_credit_score'] = (features['credit_score'] < 650).astype(int)
    
    # 4. Age-based risk factors
    if 'age' in features.columns:
        features['is_young_borrower'] = (features['age'] < 25).astype(int)
        features['is_prime_age'] = ((features['age'] >= 30) & (features['age'] <= 50)).astype(int)
        features['is_senior'] = (features['age'] > 60).astype(int)
        features['age_squared'] = features['age'] ** 2
        features['age_log'] = np.log1p(features['age'])
    
    # 5. Income stability indicators
    if 'employment_length' in features.columns:
        features['employment_stability'] = features['employment_length'].apply(
            lambda x: 1 if x >= 3 else 0
        )
        features['employment_length_squared'] = features['employment_length'] ** 2
        features['short_employment'] = (features['employment_length'] < 2).astype(int)
    
    # 6. Loan size categories (relative to income)
    if 'loan_amount' in features.columns and 'annual_income' in features.columns:
        features['high_loan_amount'] = (features['loan_amount'] > features['annual_income'] * 2).astype(int)
        features['loan_amount_log'] = np.log1p(features['loan_amount'])
        features['loan_to_annual_income'] = features['loan_amount'] / (features['annual_income'] + 1)
    
    # 7. Interest rate risk
    if 'interest_rate' in features.columns:
        features['high_interest_rate'] = (features['interest_rate'] > 12).astype(int)
        features['very_high_interest'] = (features['interest_rate'] > 15).astype(int)
        features['interest_rate_squared'] = features['interest_rate'] ** 2
        features['interest_rate_log'] = np.log1p(features['interest_rate'])
    
    # 8. Combined risk scores
    if 'credit_score' in features.columns and 'monthly_income' in features.columns:
        features['income_credit_interaction'] = features['monthly_income'] * features['credit_score'] / 1000000
        features['credit_income_ratio'] = features['credit_score'] / (features['monthly_income'] + 1)
    
    # 9. Financial health indicators
    if 'monthly_income' in features.columns:
        features['income_log'] = np.log1p(features['monthly_income'])
        features['low_income'] = (features['monthly_income'] < 3000).astype(int)
        features['high_income'] = (features['monthly_income'] > 8000).astype(int)
    
    # 10. Loan term analysis
    if 'loan_term' in features.columns and 'loan_amount' in features.columns:
        features['monthly_payment_estimate'] = features['loan_amount'] / (features['loan_term'] + 1)
        features['long_term_loan'] = (features['loan_term'] > 48).astype(int)
    
    # Remove non-feature columns
    cols_to_remove = ['customer_id', 'application_id', 'referral_code', 
                     'account_status_code', 'random_noise_1', 'state',
                     'previous_zip_code', 'loan_officer_id', 'marketing_campaign']
    if 'default' in features.columns:
        cols_to_remove.append('default')
    
    for col in cols_to_remove:
        if col in features.columns:
            features = features.drop(col, axis=1)
    
    # Encode categorical variables
    categorical_cols = features.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        # One-hot encode categorical variables
        dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
        features = pd.concat([features, dummies], axis=1)
        features = features.drop(col, axis=1)
    
    # Fill missing values
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        features[col] = features[col].fillna(features[col].median())
    
    # Add customer_id back
    features.insert(0, 'customer_id', customer_id)
    
    # Add target back if training
    if target is not None:
        features['default'] = target
    
    return features


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding customer_id and default)
    
    Args:
        df: DataFrame with features
        
    Returns:
        List of feature column names
    """
    exclude_cols = ['customer_id', 'default']
    return [col for col in df.columns if col not in exclude_cols]


def prepare_training_data(df: pd.DataFrame):
    """
    Prepare data for training
    
    Args:
        df: Input DataFrame with default column
        
    Returns:
        X (features), y (target), customer_ids
    """
    features_df = create_features(df, is_training=True)
    customer_ids = features_df['customer_id'].values
    
    feature_cols = get_feature_columns(features_df)
    X = features_df[feature_cols].values
    y = features_df['default'].values
    
    return X, y, customer_ids, feature_cols


def prepare_prediction_data(df: pd.DataFrame, feature_cols: List[str]):
    """
    Prepare data for prediction
    
    Args:
        df: Input DataFrame without default column
        feature_cols: List of feature column names from training
        
    Returns:
        X (features), customer_ids
    """
    features_df = create_features(df, is_training=False)
    customer_ids = features_df['customer_id'].values
    
    # Ensure all training features are present
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Select only the features used in training
    X = features_df[feature_cols].values
    
    return X, customer_ids
