"""
Data Loader Module
Loads and merges data from multiple file formats
"""
import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict


def clean_currency(value):
    """Convert currency strings to float"""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Remove $, commas, and convert to float
    value_str = str(value).replace('$', '').replace(',', '').strip()
    try:
        return float(value_str)
    except ValueError:
        return None


def load_loan_details(file_path: str) -> pd.DataFrame:
    """Load loan details from Excel"""
    df = pd.read_excel(file_path)
    # Clean loan_amount
    if 'loan_amount' in df.columns:
        df['loan_amount'] = df['loan_amount'].apply(clean_currency)
    # Standardize loan_type
    if 'loan_type' in df.columns:
        df['loan_type'] = df['loan_type'].str.upper().str.strip()
    return df


def load_demographics(file_path: str) -> pd.DataFrame:
    """Load demographics from CSV"""
    df = pd.read_csv(file_path)
    # Clean annual_income
    if 'annual_income' in df.columns:
        df['annual_income'] = df['annual_income'].apply(clean_currency)
    # Handle missing employment_length
    if 'employment_length' in df.columns:
        df['employment_length'] = pd.to_numeric(df['employment_length'], errors='coerce')
        df['employment_length'] = df['employment_length'].fillna(0)
    # Standardize employment_type
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].str.upper().str.strip()
        df['employment_type'] = df['employment_type'].replace({
            'FULL_TIME': 'FULL-TIME',
            'FULLTIME': 'FULL-TIME',
            'FULL TIME': 'FULL-TIME'
        })
    return df


def load_application_metadata(file_path: str) -> pd.DataFrame:
    """Load application metadata from CSV"""
    return pd.read_csv(file_path)


def load_financial_ratios(file_path: str) -> pd.DataFrame:
    """Load financial ratios from JSONL"""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    
    # Clean currency fields
    currency_fields = [
        'monthly_income', 'existing_monthly_debt', 'monthly_payment',
        'revolving_balance', 'credit_usage_amount', 'available_credit',
        'total_monthly_debt_payment', 'total_debt_amount', 'monthly_free_cash_flow'
    ]
    for field in currency_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_currency)
    
    return df


def load_credit_history(file_path: str) -> pd.DataFrame:
    """Load credit history from Parquet"""
    return pd.read_parquet(file_path)


def load_geographic_data(file_path: str) -> pd.DataFrame:
    """Load geographic data from XML"""
    return pd.read_xml(file_path)


def merge_all_data(data_dir: str) -> pd.DataFrame:
    """
    Load and merge all data sources
    
    Args:
        data_dir: Directory containing all data files
        
    Returns:
        Merged DataFrame with all customer information
    """
    data_path = Path(data_dir)
    
    # Load all datasets
    loan_df = load_loan_details(str(data_path / 'loan_details.xlsx'))
    demographics_df = load_demographics(str(data_path / 'demographics.csv'))
    app_meta_df = load_application_metadata(str(data_path / 'application_metadata.csv'))
    financial_df = load_financial_ratios(str(data_path / 'financial_ratios.jsonl'))
    credit_df = load_credit_history(str(data_path / 'credit_history.parquet'))
    geo_df = load_geographic_data(str(data_path / 'geographic_data.xml'))
    
    # Start with application metadata as base (has the target variable)
    merged = app_meta_df.copy()
    merged = merged.rename(columns={'customer_ref': 'customer_id'})
    
    # Merge loan details
    loan_df = loan_df.rename(columns={'customer_id': 'customer_id'})
    merged = merged.merge(loan_df, on='customer_id', how='left')
    
    # Merge demographics
    demographics_df = demographics_df.rename(columns={'cust_id': 'customer_id'})
    merged = merged.merge(demographics_df, on='customer_id', how='left')
    
    # Merge financial ratios
    financial_df = financial_df.rename(columns={'cust_num': 'customer_id'})
    merged = merged.merge(financial_df, on='customer_id', how='left')
    
    # Merge credit history
    credit_df = credit_df.rename(columns={'customer_number': 'customer_id'})
    merged = merged.merge(credit_df, on='customer_id', how='left')
    
    # Merge geographic data
    geo_df = geo_df.rename(columns={'id': 'customer_id'})
    merged = merged.merge(geo_df, on='customer_id', how='left')
    
    return merged


def load_data_for_prediction(data_dir: str) -> pd.DataFrame:
    """
    Load data for prediction (without default column)
    
    Args:
        data_dir: Directory containing all data files
        
    Returns:
        DataFrame ready for prediction
    """
    df = merge_all_data(data_dir)
    # Remove default column if it exists for prediction
    if 'default' in df.columns:
        df = df.drop('default', axis=1)
    return df
