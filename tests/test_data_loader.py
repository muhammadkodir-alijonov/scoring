"""
Tests for data loader module
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import (
    clean_currency,
    load_loan_details,
    load_demographics,
    load_financial_ratios,
    load_credit_history,
    merge_all_data
)


def test_clean_currency():
    """Test currency cleaning function"""
    test_cases = [
        ('$1,234.56', 1234.56),
        ('$100', 100.0),
        ('5000.50', 5000.50),
        (1000, 1000.0),
        (None, None),
        ('$0', 0.0),
        ('', None),
        ('$999,999.99', 999999.99),
    ]
    
    for input_val, expected in test_cases:
        result = clean_currency(input_val)
        assert result == expected, f"Failed for {input_val}: expected {expected}, got {result}"
    
    print(f"✓ Currency cleaning tests passed ({len(test_cases)} cases)")


def test_load_data():
    """Test loading all data sources"""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Test loan details (XLSX format)
    loan_df = load_loan_details(str(data_dir / 'loan_details.xlsx'))
    assert len(loan_df) > 0, "Loan details should not be empty"
    assert 'customer_id' in loan_df.columns, "customer_id column missing"
    assert 'loan_amount' in loan_df.columns, "loan_amount column missing"
    print(f"✓ Loaded {len(loan_df)} loan records")
    
    # Test demographics
    demo_df = load_demographics(str(data_dir / 'demographics.csv'))
    assert len(demo_df) > 0, "Demographics should not be empty"
    assert 'cust_id' in demo_df.columns, "cust_id column missing"
    print(f"✓ Loaded {len(demo_df)} demographic records")
    
    # Test financial ratios (JSONL format)
    fin_df = load_financial_ratios(str(data_dir / 'financial_ratios.jsonl'))
    assert len(fin_df) > 0, "Financial ratios should not be empty"
    assert 'cust_num' in fin_df.columns, "cust_num column missing"
    print(f"✓ Loaded {len(fin_df)} financial ratio records")
    
    # Test credit history (PARQUET format)
    credit_df = load_credit_history(str(data_dir / 'credit_history.parquet'))
    assert len(credit_df) > 0, "Credit history should not be empty"
    assert 'customer_number' in credit_df.columns, "customer_number column missing"
    print(f"✓ Loaded {len(credit_df)} credit history records")


def test_merge_data():
    """Test merging all data sources"""
    data_dir = Path(__file__).parent.parent / 'data'
    merged = merge_all_data(str(data_dir))
    
    assert len(merged) > 0, "Merged data should not be empty"
    assert 'customer_id' in merged.columns, "customer_id column missing"
    assert 'default' in merged.columns, "default column missing"
    
    # Check that data from all sources is present
    required_columns = {
        'loan_amount': 'loan_details',
        'age': 'demographics',
        'debt_to_income_ratio': 'financial_ratios',
        'credit_score': 'credit_history',
        'regional_unemployment_rate': 'geographic_data'
    }
    
    for col, source in required_columns.items():
        assert col in merged.columns, f"{col} missing (from {source})"
    
    # Validate data quality
    assert merged['customer_id'].nunique() == len(merged), "Duplicate customer IDs found"
    assert not merged['customer_id'].isna().any(), "Missing customer IDs found"
    assert merged['default'].isin([0, 1]).all(), "Invalid default values"
    
    # Check for reasonable data ranges
    assert (merged['age'] >= 18).all() and (merged['age'] <= 100).all(), "Age out of range"
    assert (merged['loan_amount'] > 0).all(), "Negative loan amounts found"
    assert (merged['credit_score'] >= 300).all() and (merged['credit_score'] <= 850).all(), "Credit score out of range"
    
    print(f"✓ Merged data has {len(merged)} records with {len(merged.columns)} columns")
    print(f"✓ All {len(required_columns)} data sources validated")
    print(f"✓ Data quality checks passed")


def run_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running Data Loader Tests")
    print("="*60 + "\n")
    
    try:
        test_clean_currency()
        test_load_data()
        test_merge_data()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
