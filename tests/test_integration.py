"""
Integration test - Simulates hackathon evaluation
OPTIMIZED VERSION: Faster execution with better validation
"""
import sys
from pathlib import Path
import pandas as pd
import shutil
import os
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from credit_scorer import CreditScorer


def test_full_workflow():
    """
    Test the complete workflow:
    1. Train model on data with 'default' column
    2. Remove 'default' column
    3. Predict on data without 'default'
    4. Verify predictions match expected format
    """
    print("\n" + "="*60)
    print("Integration Test - Simulating Hackathon Evaluation")
    print("="*60 + "\n")
    
    start_time = time.time()
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    test_dir = base_dir / 'test_temp'
    
    try:
        # Step 1: Train model (OPTIMIZED: use gradient_boosting instead of random_forest)
        print("Step 1: Training model on historical data...")
        step_start = time.time()
        scorer = CreditScorer(model_type='gradient_boosting')  # Faster than random_forest
        metrics = scorer.train(str(data_dir))
        print(f"  ⏱ Training time: {time.time() - step_start:.2f}s")
        
        assert scorer.is_trained, "Model should be trained"
        print(f"  ✓ Model trained successfully")
        print(f"  ✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        # Save ground truth
        app_meta = pd.read_csv(data_dir / 'application_metadata.csv')
        ground_truth = app_meta[['customer_ref', 'default']].copy()
        ground_truth = ground_truth.rename(columns={'customer_ref': 'customer_id'})
        
        # Step 2: Create test data (simulate hackathon test data)
        print("\nStep 2: Creating test data (removing 'default' column)...")
        test_dir.mkdir(exist_ok=True)
        
        # Copy all files
        for file in data_dir.glob('*'):
            if file.is_file():
                shutil.copy(file, test_dir / file.name)
        
        # Remove default column from application_metadata
        test_app_meta = pd.read_csv(test_dir / 'application_metadata.csv')
        test_app_meta = test_app_meta.drop('default', axis=1)
        test_app_meta.to_csv(test_dir / 'application_metadata.csv', index=False)
        print("  ✓ Test data created (default column removed)")
        
        # Step 3: Make predictions
        print("\nStep 3: Making predictions on test data...")
        step_start = time.time()
        predictions = scorer.predict(str(test_dir))
        print(f"  ⏱ Prediction time: {time.time() - step_start:.2f}s")
        
        assert 'customer_id' in predictions.columns, "Predictions must have customer_id"
        assert 'default' in predictions.columns, "Predictions must have default"
        assert len(predictions) > 0, "Predictions should not be empty"
        assert len(predictions) == len(ground_truth), "Prediction count must match ground truth"
        print(f"  ✓ Generated {len(predictions)} predictions")
        
        # Step 4: Verify format
        print("\nStep 4: Verifying output format...")
        
        # Check values are 0 or 1
        assert predictions['default'].isin([0, 1]).all(), "Default must be 0 or 1"
        print("  ✓ All predictions are 0 or 1")
        
        # Check no missing values
        assert not predictions['customer_id'].isna().any(), "No missing customer IDs"
        assert not predictions['default'].isna().any(), "No missing predictions"
        print("  ✓ No missing values")
        
        # Step 5: Calculate accuracy against ground truth
        print("\nStep 5: Evaluating predictions against ground truth...")
        merged = ground_truth.merge(
            predictions[['customer_id', 'default']], 
            on='customer_id', 
            suffixes=('_true', '_pred')
        )
        
        accuracy = (merged['default_true'] == merged['default_pred']).mean()
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(merged['default_true'], merged['default_pred'])
        recall = recall_score(merged['default_true'], merged['default_pred'])
        f1 = f1_score(merged['default_true'], merged['default_pred'])
        
        print(f"  ✓ Prediction accuracy: {accuracy:.2%}")
        print(f"  ✓ Precision: {precision:.2%}")
        print(f"  ✓ Recall: {recall:.2%}")
        print(f"  ✓ F1 Score: {f1:.4f}")
        
        # Validate minimum performance thresholds
        assert accuracy > 0.85, f"Accuracy {accuracy:.2%} is below 85% threshold"
        assert recall > 0.70, f"Recall {recall:.2%} is below 70% threshold"
        print("  ✓ Performance thresholds met")
        
        # Step 6: Save in required format
        print("\nStep 6: Saving in competition format...")
        output_file = test_dir / 'submission.csv'
        predictions[['customer_id', 'default']].to_csv(output_file, index=False)
        
        # Verify file format
        submission = pd.read_csv(output_file)
        assert list(submission.columns) == ['customer_id', 'default'], \
            "Output must have columns: customer_id, default"
        assert len(submission) == len(predictions), "All customers must be in output"
        print(f"  ✓ Submission file saved: {output_file}")
        print(f"  ✓ Format verified: {len(submission)} rows, 2 columns")
        
        # Display sample and statistics
        print("\nSample predictions:")
        print(submission.head(5).to_string(index=False))
        
        default_rate = submission['default'].mean()
        print(f"\nPrediction Statistics:")
        print(f"  Default rate: {default_rate:.2%}")
        print(f"  Total customers: {len(submission)}")
        print(f"  Predicted defaults: {submission['default'].sum()}")
        print(f"  Predicted non-defaults: {(submission['default'] == 0).sum()}")
        
        total_time = time.time() - start_time
        print(f"\n⏱ Total test time: {total_time:.2f}s")
        
        print("\n" + "="*60)
        print("Integration Test PASSED ✓")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60 + "\n")
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    # Test 1: Model prediction before training
    print("Test 1: Prediction before training...")
    try:
        scorer = CreditScorer()
        scorer.predict(str(data_dir))
        print("  ✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print("  ✓ Correctly raised ValueError")
    
    # Test 2: Training without default column
    print("\nTest 2: Training without default column...")
    test_dir = base_dir / 'test_temp_edge'
    try:
        test_dir.mkdir(exist_ok=True)
        for file in data_dir.glob('*'):
            if file.is_file():
                shutil.copy(file, test_dir / file.name)
        
        # Remove default column
        test_app_meta = pd.read_csv(test_dir / 'application_metadata.csv')
        test_app_meta = test_app_meta.drop('default', axis=1)
        test_app_meta.to_csv(test_dir / 'application_metadata.csv', index=False)
        
        scorer = CreditScorer()
        scorer.train(str(test_dir))
        print("  ✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print("  ✓ Correctly raised ValueError")
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)
    
    print("\n" + "="*60)
    print("Edge Case Tests PASSED ✓")
    print("="*60 + "\n")
    
    return True


def run_all_tests():
    """Run all integration tests"""
    success = True
    
    success = test_full_workflow() and success
    success = test_edge_cases() and success
    
    if success:
        print("\n" + "="*60)
        print("ALL INTEGRATION TESTS PASSED ✓✓✓")
        print("Ready for hackathon evaluation!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED ✗")
        print("="*60 + "\n")
    
    return success


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
