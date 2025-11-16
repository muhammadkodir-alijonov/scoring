"""
Model Comparison Tool
Tests different thresholds to find optimal configuration for production
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, 'src')

# Load the trained model
print("=" * 80)
print("QUICK MODEL COMPARISON - Different Thresholds")
print("=" * 80)
print()

from credit_scorer import CreditScorer

scorer = CreditScorer()
scorer.load_model('credit_model.pkl')

# Load actual data
actual_df = pd.read_csv('data/application_metadata.csv')
if 'customer_ref' in actual_df.columns:
    actual_df = actual_df.rename(columns={'customer_ref': 'customer_id'})

# Load test data and get probabilities
from data_loader import merge_all_data
from feature_engineering import prepare_prediction_data

df = merge_all_data('data_test')
X, customer_ids = prepare_prediction_data(df, scorer.feature_cols)
X_scaled = scorer.scaler.transform(X)
probabilities = scorer.model.predict_proba(X_scaled)[:, 1]

# Test different thresholds
thresholds = [0.15, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22]

results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'customer_id': customer_ids,
        'default': predictions
    })
    
    # Merge with actual
    eval_df = actual_df[['customer_id', 'default']].merge(
        pred_df, 
        on='customer_id', 
        suffixes=('_actual', '_predicted')
    )
    
    # Calculate metrics
    accuracy = accuracy_score(eval_df['default_actual'], eval_df['default_predicted'])
    cm = confusion_matrix(eval_df['default_actual'], eval_df['default_predicted'])
    
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate mismatches
    mismatches = fn + fp
    
    # Financial impact (average loan $105,978)
    avg_loan = 105978
    fn_risk = fn * avg_loan * 0.7  # 70% loss on defaults
    fp_risk = fp * avg_loan * 0.05  # 5% profit loss on rejected good customers
    total_risk = fn_risk + fp_risk
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mismatches': mismatches,
        'fn': fn,
        'fp': fp,
        'fn_risk': fn_risk,
        'fp_risk': fp_risk,
        'total_risk': total_risk
    })

# Display results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print()

# Print header
print(f"{'Threshold':<12} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1':<10} {'Total MM':<10} {'FN':<8} {'FP':<8} {'Total Risk':<15}")
print("-" * 120)

# Print results
for r in results:
    print(f"{r['threshold']:<12.2f} {r['accuracy']:<10.4f} {r['recall']:<10.4f} {r['precision']:<10.4f} {r['f1']:<10.4f} {r['mismatches']:<10} {r['fn']:<8} {r['fp']:<8} ${r['total_risk']:>13,.0f}")

print()

# Find best models
best_recall = max(results, key=lambda x: x['recall'])
best_risk = min(results, key=lambda x: x['total_risk'])
best_balanced = min(results, key=lambda x: abs(x['recall'] - 0.75) + abs(x['precision'] - 0.50))

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()

print("🏆 BEST FOR RECALL (catch most defaults):")
print(f"   Threshold: {best_recall['threshold']:.2f}")
print(f"   Recall: {best_recall['recall']:.1%}, FN: {best_recall['fn']}, Total Risk: ${best_recall['total_risk']:,.0f}")
print()

print("💰 BEST FOR FINANCIAL RISK (lowest total cost):")
print(f"   Threshold: {best_risk['threshold']:.2f}")
print(f"   Recall: {best_risk['recall']:.1%}, FN: {best_risk['fn']}, Total Risk: ${best_risk['total_risk']:,.0f}")
print()

print("⚖️  BEST BALANCED (recall + precision):")
print(f"   Threshold: {best_balanced['threshold']:.2f}")
print(f"   Recall: {best_balanced['recall']:.1%}, Precision: {best_balanced['precision']:.1%}")
print(f"   FN: {best_balanced['fn']}, FP: {best_balanced['fp']}, Total Risk: ${best_balanced['total_risk']:,.0f}")
print()

print("=" * 80)
print("RECOMMENDATION FOR PRODUCTION:")
print("=" * 80)
print()

# Recommend the one with best risk
print(f"✅ Use threshold: {best_risk['threshold']:.2f}")
print(f"   This minimizes total financial risk while maintaining good recall.")
print()
print(f"   Expected Performance:")
print(f"   - Recall: {best_risk['recall']:.1%} (will catch {best_risk['recall']:.1%} of defaults)")
print(f"   - Precision: {best_risk['precision']:.1%} (of flagged risks, {best_risk['precision']:.1%} are actual defaults)")
print(f"   - Total Mismatches: {best_risk['mismatches']:,}")
print(f"   - False Negatives: {best_risk['fn']:,} (bad loans approved)")
print(f"   - False Positives: {best_risk['fp']:,} (good customers rejected)")
print(f"   - Total Financial Risk: ${best_risk['total_risk']:,.0f}")
print()
