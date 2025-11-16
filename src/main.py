"""
Main script for credit scoring module
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from credit_scorer import CreditScorer


def train_model(data_dir: str, output_model: str, model_type: str = 'random_forest'):
    """Train credit scoring model"""
    print("=" * 60)
    print("Credit Scoring Model - Training")
    print("=" * 60)
    
    scorer = CreditScorer(model_type=model_type)
    metrics = scorer.train(data_dir)
    
    # Save model
    scorer.save_model(output_model)
    
    # Save feature importance
    importance_df = scorer.get_feature_importance(top_n=20)
    if importance_df is not None:
        importance_file = output_model.replace('.pkl', '_feature_importance.csv')
        importance_df.to_csv(importance_file, index=False)
        print(f"\nFeature importance saved to {importance_file}")
    
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


def predict_defaults(data_dir: str, model_path: str, output_file: str):
    """Predict credit defaults for new data"""
    print("=" * 60)
    print("Credit Scoring Model - Prediction")
    print("=" * 60)
    
    # Load model
    scorer = CreditScorer()
    scorer.load_model(model_path)
    
    # Make predictions
    results = scorer.predict(data_dir)
    
    # Save predictions in COMPETITION FORMAT
    # Format: customer_id, prob, default
    # All 3 columns for evaluation criteria:
    # 1. AUC (Area Under Curve)
    # 2. Linguistic ranking (3 column correlation)
    # 3. Spearman correlation
    
    # Ensure proper data types
    results['customer_id'] = results['customer_id'].astype(int)
    results['prob'] = results['prob'].astype(float)
    results['default'] = results['default'].astype(int)
    
    # Save with tab separator to match results.csv format
    results[['customer_id', 'prob', 'default']].to_csv(
        output_file, 
        index=False, 
        sep='\t',
        float_format='%.5f'
    )
    print(f"\nPredictions saved to {output_file}")
    print(f"Format: customer_id (int), prob (float), default (int)")
    print(f"Total predictions: {len(results)}")
    
    print("=" * 60)
    print("Prediction completed successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Credit Scoring Module - Predict loan defaults'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing training data files'
    )
    train_parser.add_argument(
        '--output-model',
        default='credit_model.pkl',
        help='Path to save trained model (default: credit_model.pkl)'
    )
    train_parser.add_argument(
        '--model-type',
        choices=['random_forest', 'gradient_boosting', 'logistic', 'ensemble', 'optimized', 'xgboost', 'xgboost_smote'],
        default='random_forest',
        help='Type of model to train (default: random_forest)'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing data files for prediction'
    )
    predict_parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model file'
    )
    predict_parser.add_argument(
        '--output',
        default='predictions.csv',
        help='Path to save predictions (default: predictions.csv)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.data_dir, args.output_model, args.model_type)
    elif args.command == 'predict':
        predict_defaults(args.data_dir, args.model, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
