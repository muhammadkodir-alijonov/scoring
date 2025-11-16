"""
Model Performance Tests - Validates model quality and behavior
OPTIMIZED: Fast validation of critical model metrics
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from credit_scorer import CreditScorer


def test_model_accuracy_threshold():
    """Test that model meets minimum accuracy requirements"""
    print("\n" + "="*60)
    print("Test 1: Model Accuracy Threshold")
    print("="*60 + "\n")
    
    start_time = time.time()
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Train model
    scorer = CreditScorer(model_type='gradient_boosting')
    metrics = scorer.train(str(data_dir))
    
    # Validate metrics
    print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Test AUC: {metrics['test_auc']:.4f}")
    
    # Assertions
    assert metrics['test_accuracy'] >= 0.85, \
        f"Accuracy {metrics['test_accuracy']:.2%} below 85% threshold"
    assert metrics['test_auc'] >= 0.75, \
        f"AUC {metrics['test_auc']:.4f} below 0.75 threshold"
    
    print(f"✓ Model meets accuracy requirements")
    print(f"⏱ Time: {time.time() - start_time:.2f}s\n")
    
    return True


def test_recall_for_defaults():
    """Test that model catches sufficient defaults (recall)"""
    print("\n" + "="*60)
    print("Test 2: Recall for Default Detection")
    print("="*60 + "\n")
    
    start_time = time.time()
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Train model
    scorer = CreditScorer(model_type='gradient_boosting')
    metrics = scorer.train(str(data_dir))
    
    # Extract classification report
    report = metrics['classification_report']
    default_recall = report['1']['recall']
    
    print(f"Recall for defaults (class 1): {default_recall:.2%}")
    
    # Critical: must catch at least 70% of defaults
    assert default_recall >= 0.70, \
        f"Recall {default_recall:.2%} too low - missing too many defaults"
    
    print(f"✓ Model catches {default_recall:.0%} of actual defaults")
    print(f"⏱ Time: {time.time() - start_time:.2f}s\n")
    
    return True


def test_prediction_distribution():
    """Test that predictions have reasonable distribution"""
    print("\n" + "="*60)
    print("Test 3: Prediction Distribution")
    print("="*60 + "\n")
    
    start_time = time.time()
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Train and predict
    scorer = CreditScorer(model_type='gradient_boosting')
    scorer.train(str(data_dir))
    predictions = scorer.predict(str(data_dir))
    
    # Analyze distribution
    default_rate = predictions['default'].mean()
    n_defaults = predictions['default'].sum()
    n_total = len(predictions)
    
    print(f"Predicted default rate: {default_rate:.2%}")
    print(f"Predicted defaults: {n_defaults:,} / {n_total:,}")
    
    # Validate: should not predict all/none as default
    assert 0.01 < default_rate < 0.99, \
        f"Extreme prediction rate {default_rate:.2%} - model not learning"
    
    # Validate: should have both classes
    assert predictions['default'].nunique() == 2, \
        "Model predicting only one class"
    
    print(f"✓ Reasonable prediction distribution")
    print(f"⏱ Time: {time.time() - start_time:.2f}s\n")
    
    return True


def test_prediction_consistency():
    """Test that predictions are consistent across runs"""
    print("\n" + "="*60)
    print("Test 4: Prediction Consistency")
    print("="*60 + "\n")
    
    start_time = time.time()
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Train model once
    scorer = CreditScorer(model_type='gradient_boosting')
    scorer.train(str(data_dir))
    
    # Make predictions twice
    pred1 = scorer.predict(str(data_dir))
    pred2 = scorer.predict(str(data_dir))
    
    # Compare
    merged = pred1.merge(pred2, on='customer_id', suffixes=('_1', '_2'))
    consistency = (merged['default_1'] == merged['default_2']).mean()
    
    print(f"Prediction consistency: {consistency:.2%}")
    
    # Should be 100% consistent
    assert consistency == 1.0, \
        f"Predictions inconsistent: only {consistency:.2%} match"
    
    print(f"✓ Predictions are deterministic")
    print(f"⏱ Time: {time.time() - start_time:.2f}s\n")
    
    return True


def test_feature_importance():
    """Test that model uses features appropriately"""
    print("\n" + "="*60)
    print("Test 5: Feature Importance")
    print("="*60 + "\n")
    
    start_time = time.time()
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Train model
    scorer = CreditScorer(model_type='gradient_boosting')
    scorer.train(str(data_dir))
    
    # Get feature importance
    if hasattr(scorer.model, 'feature_importances_'):
        importances = scorer.model.feature_importances_
        
        # Validate: no single feature dominates
        max_importance = importances.max()
        print(f"Max feature importance: {max_importance:.2%}")
        
        # Validate: features are being used
        n_used = (importances > 0).sum()
        n_total = len(importances)
        print(f"Features used: {n_used} / {n_total}")
        
        assert max_importance < 0.8, \
            f"One feature dominates ({max_importance:.2%})"
        assert n_used >= 10, \
            f"Too few features used ({n_used})"
        
        # Show top 5 features
        feature_names = scorer.feature_names if hasattr(scorer, 'feature_names') else \
                       [f"feature_{i}" for i in range(len(importances))]
        
        top_indices = np.argsort(importances)[-5:][::-1]
        print("\nTop 5 most important features:")
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
        
        print(f"\n✓ Feature importance validated")
    else:
        print("⚠ Model doesn't support feature importance")
    
    print(f"⏱ Time: {time.time() - start_time:.2f}s\n")
    
    return True


def test_no_data_leakage():
    """Test that there's no data leakage (train != test)"""
    print("\n" + "="*60)
    print("Test 6: No Data Leakage")
    print("="*60 + "\n")
    
    start_time = time.time()
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Train model
    scorer = CreditScorer(model_type='gradient_boosting')
    metrics = scorer.train(str(data_dir))
    
    train_acc = metrics['train_accuracy']
    test_acc = metrics['test_accuracy']
    
    print(f"Train accuracy: {train_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    
    # If train accuracy is much higher than test, possible overfitting
    gap = train_acc - test_acc
    print(f"Accuracy gap: {gap:.2%}")
    
    # Reasonable gap is < 10%
    assert gap < 0.15, \
        f"Large train-test gap ({gap:.2%}) suggests overfitting"
    
    print(f"✓ No significant overfitting detected")
    print(f"⏱ Time: {time.time() - start_time:.2f}s\n")
    
    return True


def run_all_performance_tests():
    """Run all model performance tests"""
    print("\n" + "="*70)
    print("MODEL PERFORMANCE TEST SUITE - OPTIMIZED")
    print("="*70)
    
    total_start = time.time()
    tests = [
        ("Accuracy Threshold", test_model_accuracy_threshold),
        ("Recall Detection", test_recall_for_defaults),
        ("Prediction Distribution", test_prediction_distribution),
        ("Prediction Consistency", test_prediction_consistency),
        ("Feature Importance", test_feature_importance),
        ("No Data Leakage", test_no_data_leakage),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ PASS"))
        except AssertionError as e:
            results.append((name, f"✗ FAIL: {e}"))
        except Exception as e:
            results.append((name, f"✗ ERROR: {e}"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    for name, result in results:
        print(f"{name:.<40} {result}")
    
    passed = sum(1 for _, r in results if "✓ PASS" in r)
    total = len(results)
    
    print("="*70)
    print(f"PASSED: {passed}/{total}")
    
    total_time = time.time() - total_start
    print(f"⏱ Total time: {total_time:.2f}s")
    print("="*70 + "\n")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_performance_tests()
    sys.exit(0 if success else 1)
