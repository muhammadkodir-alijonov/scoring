"""
Data Quality Analysis and Feature Importance
Analyzes data quality, missing values, outliers, and feature correlations
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import merge_all_data


def analyze_data_quality(data_dir: str):
    """Analyze data quality and feature importance"""
    print("=" * 80)
    print("DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    df = merge_all_data(data_dir)
    print(f"   Total records: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")
    
    if 'default' not in df.columns:
        print("❌ Error: 'default' column not found")
        return
    
    # Basic statistics
    print("\n" + "=" * 80)
    print("📊 TARGET DISTRIBUTION")
    print("=" * 80)
    default_counts = df['default'].value_counts()
    print(f"Non-default (0): {default_counts[0]} ({default_counts[0]/len(df)*100:.2f}%)")
    print(f"Default (1):     {default_counts[1]} ({default_counts[1]/len(df)*100:.2f}%)")
    print(f"Imbalance ratio: {default_counts[0]/default_counts[1]:.1f}:1")
    
    # Missing values analysis
    print("\n" + "=" * 80)
    print("🔍 MISSING VALUES ANALYSIS")
    print("=" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if len(missing_df) > 0:
        print("\n⚠️  Columns with missing values:")
        print(missing_df.to_string(index=False))
    else:
        print("✅ No missing values found!")
    
    # Numeric columns analysis
    print("\n" + "=" * 80)
    print("📈 NUMERIC FEATURES CORRELATION WITH DEFAULT")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['customer_id', 'default']]
    
    correlations = []
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            corr = df[col].corr(df['default'])
            correlations.append({
                'Feature': col,
                'Correlation': abs(corr),
                'Direction': 'Positive' if corr > 0 else 'Negative',
                'Raw_Corr': corr
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    
    print("\n🔝 TOP 20 MOST IMPORTANT FEATURES:")
    print("-" * 80)
    print(f"{'Feature':<30} {'Correlation':<15} {'Direction':<15} {'Raw Value':<15}")
    print("-" * 80)
    for idx, row in corr_df.head(20).iterrows():
        print(f"{row['Feature']:<30} {row['Correlation']:<15.4f} {row['Direction']:<15} {row['Raw_Corr']:<15.4f}")
    
    print("\n⚠️  WEAK FEATURES (Correlation < 0.01):")
    weak_features = corr_df[corr_df['Correlation'] < 0.01]
    if len(weak_features) > 0:
        print(f"Found {len(weak_features)} weak features that can be removed:")
        for feat in weak_features['Feature'].head(10):
            print(f"  - {feat}")
    
    # Outliers detection
    print("\n" + "=" * 80)
    print("🚨 OUTLIERS DETECTION (IQR Method)")
    print("=" * 80)
    
    outlier_summary = []
    for col in numeric_cols[:15]:  # Check top 15 numeric columns
        if df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = outliers / len(df) * 100
            
            if outlier_pct > 1:
                outlier_summary.append({
                    'Feature': col,
                    'Outliers': outliers,
                    'Percentage': outlier_pct,
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Mean': df[col].mean()
                })
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary).sort_values('Percentage', ascending=False)
        print("\n⚠️  Features with significant outliers (>1%):")
        print(outlier_df.to_string(index=False))
    else:
        print("✅ No significant outliers detected!")
    
    # Categorical features analysis
    print("\n" + "=" * 80)
    print("📋 CATEGORICAL FEATURES ANALYSIS")
    print("=" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'customer_id']
    
    if len(categorical_cols) > 0:
        print(f"\nFound {len(categorical_cols)} categorical features:")
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            print(f"\n  {col}:")
            print(f"    Unique values: {unique_vals}")
            if unique_vals <= 10:
                value_counts = df[col].value_counts()
                for val, count in value_counts.head(5).items():
                    print(f"      {val}: {count} ({count/len(df)*100:.1f}%)")
    
    # Feature importance by default rate
    print("\n" + "=" * 80)
    print("💡 FEATURE INSIGHTS - DEFAULT RATE BY CATEGORY")
    print("=" * 80)
    
    # Analyze key numeric features
    key_features = ['credit_score', 'monthly_income', 'loan_amount', 'interest_rate', 'age']
    
    for feat in key_features:
        if feat in df.columns:
            print(f"\n📊 {feat.upper()}:")
            
            # Create bins
            if feat == 'credit_score':
                bins = [0, 580, 650, 700, 750, 850]
                labels = ['Very Low', 'Low', 'Medium', 'Good', 'Excellent']
            elif feat == 'age':
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
            elif feat == 'monthly_income':
                bins = [0, 2000, 3000, 4000, 6000, 100000]
                labels = ['<2K', '2K-3K', '3K-4K', '4K-6K', '6K+']
            elif feat == 'interest_rate':
                bins = [0, 8, 10, 12, 15, 30]
                labels = ['<8%', '8-10%', '10-12%', '12-15%', '15%+']
            elif feat == 'loan_amount':
                bins = [0, 50000, 100000, 150000, 200000, 1000000]
                labels = ['<50K', '50K-100K', '100K-150K', '150K-200K', '200K+']
            else:
                continue
            
            df['temp_bin'] = pd.cut(df[feat], bins=bins, labels=labels)
            default_rate = df.groupby('temp_bin')['default'].agg(['mean', 'count'])
            default_rate['default_rate'] = (default_rate['mean'] * 100).round(2)
            
            print(f"{'Category':<15} {'Count':<10} {'Default Rate':<15}")
            print("-" * 40)
            for idx, row in default_rate.iterrows():
                print(f"{str(idx):<15} {int(row['count']):<10} {row['default_rate']:.2f}%")
            
            df.drop('temp_bin', axis=1, inplace=True)
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n✅ ACTIONS TO IMPROVE AUC:")
    print("1. Remove weak features (correlation < 0.01)")
    print("2. Handle outliers in key features using capping/winsorization")
    print("3. Create interaction features between highly correlated features")
    print("4. Bin numeric features based on default rate patterns")
    print("5. Apply feature scaling (StandardScaler or RobustScaler)")
    print("6. Use polynomial features for top 5-10 most important features")
    
    if len(missing_df) > 0:
        print("\n⚠️  Handle missing values:")
        print("   - Consider removing features with >30% missing")
        print("   - Use median/mode imputation for others")
        print("   - Create 'is_missing' indicator features")
    
    print("\n" + "=" * 80)
    
    return corr_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze data quality')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    args = parser.parse_args()
    
    analyze_data_quality(args.data_dir)
