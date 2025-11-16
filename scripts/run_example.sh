#!/bin/bash
# Credit Scoring System - Training & Prediction Example
# Updated for new project structure

set -e  # Exit on error

# Navigate to project root
cd "$(dirname "$0")/.."

echo "============================================================"
echo "Credit Scoring System - Production Pipeline"
echo "============================================================"
echo ""

# Step 1: Train the model
echo "Step 1: Training ensemble model for maximum AUC..."
python src/main.py train \
  --data-dir ./data \
  --output-model ./models/credit_model.pkl \
  --model-type ensemble

echo ""
echo "Step 2: Creating test data (removing default column)..."
mkdir -p data_test
cp -r data/* data_test/ 2>/dev/null || true
python -c "
import pandas as pd
df = pd.read_csv('data_test/application_metadata.csv')
df = df.drop('default', axis=1)
df.to_csv('data_test/application_metadata.csv', index=False)
print('✓ Test data created')
"

echo ""
echo "Step 3: Making predictions on test data..."
python src/main.py predict \
  --data-dir ./data_test \
  --model ./models/credit_model.pkl \
  --output ./outputs/predictions.csv

echo ""
echo "Step 4: Evaluating prediction accuracy..."
python src/evaluate.py \
  --actual ./data/application_metadata.csv \
  --predictions ./outputs/predictions.csv

echo ""
echo "Step 5: Analyzing prediction errors..."
python src/analyze_errors.py

echo ""
echo "Step 6: Sample predictions (first 20 rows)..."
head -20 ./outputs/predictions.csv

echo ""
echo "============================================================"
echo "Pipeline completed successfully! ✓"
echo "============================================================"
echo ""
echo "Output files:"
echo "  ✓ models/credit_model.pkl           - Trained model"
echo "  ✓ outputs/predictions.csv           - Full predictions"
echo "  ✓ outputs/predictions_simple.csv    - Submission format"
echo "  ✓ outputs/prediction_mismatches.csv - Error analysis"
echo ""
echo "Performance Summary:"
grep -E "Accuracy|Recall|Precision|F1-Score" ./outputs/predictions.csv | head -5 || echo "  Run analysis for detailed metrics"
echo ""
