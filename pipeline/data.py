"""
Data loading and merging pipeline
Handles loading data from various formats and merging them
"""

import pandas as pd
from functools import reduce


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
