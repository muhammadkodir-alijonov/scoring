"""
Feature Selection - Keep only top important features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import merge_all_data
from feature_engineering import create_features, get_feature_columns


def get_top_features(n=40):
    """Get top N most important features"""
    try:
        importance_df = pd.read_csv('./models/credit_model_feature_importance.csv')
        top_features = importance_df.head(n)['feature'].tolist()
        print(f"✓ Selected top {n} features")
        return top_features
    except:
        print("⚠️  Feature importance file not found. Using all features.")
        return None


def select_features(df, feature_cols, top_features=None):
    """Select only important features"""
    if top_features is None:
        return df[feature_cols]
    
    # Keep only top features that exist in dataframe
    selected_cols = [col for col in top_features if col in feature_cols]
    
    print(f"\nFeature selection:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Selected: {len(selected_cols)}")
    print(f"  Removed: {len(feature_cols) - len(selected_cols)}")
    
    return df[selected_cols], selected_cols


if __name__ == '__main__':
    # Test feature selection
    data_dir = './data'
    df = merge_all_data(data_dir)
    features_df = create_features(df, is_training=True)
    feature_cols = get_feature_columns(features_df)
    
    top_features = get_top_features(n=40)
    
    if top_features:
        selected_df, selected_cols = select_features(features_df, feature_cols, top_features)
        print(f"\nSelected features:")
        for i, feat in enumerate(selected_cols[:20], 1):
            print(f"  {i}. {feat}")
        if len(selected_cols) > 20:
            print(f"  ... and {len(selected_cols) - 20} more")
