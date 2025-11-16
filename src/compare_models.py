"""
Quick Model Comparison - Test all model types to find best AUC
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from credit_scorer import CreditScorer
from data_loader import merge_all_data
from feature_engineering import prepare_training_data


def test_model(model_type: str, data_dir: str):
    """Test a single model and return metrics"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_type.upper()}")
    print(f"{'='*80}\n")
    
    try:
        scorer = CreditScorer(model_type=model_type)
        metrics = scorer.train(data_dir)
        
        print(f"\n✓ {model_type}: Test AUC = {metrics['test_auc']:.4f}")
        
        return {
            'model_type': model_type,
            'test_auc': metrics['test_auc'],
            'test_accuracy': metrics['test_accuracy'],
            'train_auc': metrics['train_auc'],
            'train_accuracy': metrics['train_accuracy']
        }
    except Exception as e:
        print(f"\n✗ {model_type} FAILED: {str(e)}")
        return {
            'model_type': model_type,
            'test_auc': 0.0,
            'test_accuracy': 0.0,
            'train_auc': 0.0,
            'train_accuracy': 0.0
        }


def main():
    print("="*80)
    print("MODEL COMPARISON TEST - Finding Best Model for 90%+ AUC")
    print("="*80)
    
    data_dir = './data'
    
    # Models to test (ordered from fastest to slowest)
    models_to_test = [
        'logistic',           # Fastest
        'random_forest',      # Fast
        'gradient_boosting',  # Medium
        'optimized',          # Slow
        'ensemble'            # Slowest
    ]
    
    results = []
    
    for model_type in models_to_test:
        result = test_model(model_type, data_dir)
        results.append(result)
        
        # Early stop if we find 90%+ AUC
        if result['test_auc'] >= 0.90:
            print(f"\n🎉 FOUND 90%+ AUC! Stopping early.")
            break
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_auc', ascending=False)
    
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    print("\n")
    print(results_df.to_string(index=False))
    
    # Find best model
    best_model = results_df.iloc[0]
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST MODEL: {best_model['model_type'].upper()}")
    print(f"{'='*80}")
    print(f"Test AUC:      {best_model['test_auc']:.4f}")
    print(f"Test Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"Train AUC:     {best_model['train_auc']:.4f}")
    print(f"Train Accuracy:{best_model['train_accuracy']:.4f}")
    print("="*80)
    
    # Save results
    results_df.to_csv('./models/model_comparison_results.csv', index=False)
    print("\n✓ Results saved to: ./models/model_comparison_results.csv")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if best_model['test_auc'] >= 0.90:
        print(f"✅ Target achieved! Use {best_model['model_type']} model.")
    elif best_model['test_auc'] >= 0.85:
        print(f"⚠️  Close to target. Consider:")
        print(f"   1. Hyperparameter tuning for {best_model['model_type']}")
        print(f"   2. More feature engineering")
        print(f"   3. Try XGBoost (if available)")
    else:
        print(f"❌ Target not met. Recommendations:")
        print(f"   1. Improve data quality and feature engineering")
        print(f"   2. Collect more data")
        print(f"   3. Try advanced models (XGBoost, LightGBM, CatBoost)")
        print(f"   4. Hyperparameter optimization (GridSearchCV/RandomizedSearchCV)")


if __name__ == '__main__':
    main()
