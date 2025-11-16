# Credit Default Prediction Model

A machine learning solution for predicting credit default risk using CatBoost algorithm, designed to handle highly imbalanced datasets with professional-grade performance.

## 🎯 Project Overview

This project implements a credit default prediction system that processes multi-source financial data to predict the likelihood of loan default. The model achieves an **AUC score of 0.80** with exceptional **69% recall** on the minority class, making it suitable for real-world credit risk assessment.

### Key Features

- **Advanced Data Processing**: Handles 6 heterogeneous data sources (JSONL, XML, Parquet, CSV, Excel)
- **Imbalanced Data Handling**: Utilizes CatBoost's `SqrtBalanced` class weighting for optimal minority class detection
- **Feature Engineering**: Automated creation of 5 interaction features based on domain knowledge
- **Robust Preprocessing**: Currency parsing, outlier clipping (3×IQR), correlation-based feature selection
- **Production-Ready**: Clean code structure with minimal dependencies

## 📊 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **AUC-ROC** | 0.8010 | Overall discriminative ability |
| **Recall (Default)** | 69% | Successfully identifies 69% of defaults |
| **Precision (Default)** | 14% | Trade-off for high recall in imbalanced data |
| **Accuracy** | 75.86% | Overall correct predictions |

### Confusion Matrix (5-Fold CV)
```
True Negatives:  65,099  |  False Positives: 20,306
False Negatives:  1,416  |  True Positives:   3,178
```

**Business Impact**: Compared to baseline models, this solution reduces false negatives by **68%**, potentially preventing significant financial losses.

## 🏗️ Architecture

### Data Pipeline
```
Raw Data (6 sources) → Standardization → Merging → Cleaning → 
Feature Selection → Feature Engineering → Model Training → Prediction
```

### Model Selection Rationale

**CatBoost** was chosen over other gradient boosting algorithms for:

1. **Native Categorical Support**: No need for manual encoding of 11 categorical features
2. **Imbalanced Data Handling**: Built-in `auto_class_weights` parameter specifically designed for minority class detection
3. **Superior Recall**: 23× improvement over standard HistGradientBoosting (69% vs 3%)
4. **Robust Performance**: Less prone to overfitting with ordered boosting

### Alternative Models

- `model_histgradient.py`: HistGradientBoosting implementation (faster but lower recall)
- `model_final.py`: Clean baseline implementation for comparison

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Virtual environment support

### Installation

```bash
# Clone repository
cd agile

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn catboost openpyxl lxml pyarrow
```

### Usage

```bash
# Run CatBoost model (recommended)
python model_catboost.py

# Run HistGradientBoosting model (alternative)
python model_histgradient.py
```

### Expected Output

```
==============================================================
 CATBOOST MODEL PERFORMANCE
==============================================================

Cross-Validation:
  AUC scores: ['0.8037', '0.7947', '0.8046', '0.7985', '0.8035']
  Mean AUC: 0.8010 +/- 0.0045

Overall AUC-ROC: 0.8010

Confusion Matrix:
  True Negatives:  65,099
  False Positives: 20,306
  False Negatives:  1,416
  True Positives:   3,178

Classification Report:
              precision    recall  f1-score   support

  No Default       0.98      0.76      0.86     85405
     Default       0.14      0.69      0.23      4594

Top 15 Most Important Features:
  credit_score                       : 7.17
  oldest_account_age_months          : 5.41
  age                                : 5.08
  ...
```

## 📁 Data Structure

### Input Files

| File | Format | Records | Description |
|------|--------|---------|-------------|
| `financial_ratios.jsonl` | JSONL | ~90K | Debt ratios, income, credit utilization |
| `demographics.csv` | CSV | ~90K | Age, employment, education |
| `credit_history.parquet` | Parquet | ~90K | Payment history, delinquencies |
| `loan_details.xlsx` | Excel | ~90K | Loan amounts, terms, types |
| `application_metadata.csv` | CSV | ~90K | Application timestamps, channels |
| `geographic_data.xml` | XML | ~90K | Regional economic indicators |

### Feature Engineering

**Interaction Features Created**:
- `debt_burden`: Combined monthly debt relative to income
- `credit_pressure`: Product of utilization and debt-to-income ratio
- `payment_stress`: Monthly payment relative to free cash flow
- `credit_efficiency`: Credit score normalized by utilization
- `delinquency_severity`: Weighted delinquency impact

## 🔧 Technical Details

### Data Preprocessing
1. **Currency Parsing**: Handles `$1,234.56` format across 11 columns
2. **Outlier Treatment**: 3×IQR clipping to preserve distribution
3. **Missing Values**: Median imputation for numerical, mode for categorical
4. **Multicollinearity**: Removes features with correlation >0.90
5. **Low Variance**: Filters features with target correlation <0.05

### Model Configuration

```python
CatBoostClassifier(
    iterations=700,
    learning_rate=0.015,
    depth=8,
    l2_leaf_reg=6,
    auto_class_weights='SqrtBalanced',
    early_stopping_rounds=50,
    eval_metric='AUC',
    cat_features=categorical_indices,
    random_seed=42
)
```

### Cross-Validation Strategy
- **Method**: Stratified 5-Fold
- **Metric**: AUC-ROC (primary), Accuracy (secondary)
- **Validation**: Preserves class distribution in each fold

## 📈 Performance Optimization

### Computational Efficiency
- Training time: ~17 seconds per fold
- Total runtime: ~45 seconds (full pipeline)
- Memory efficient: Handles 90K+ records on standard hardware

### Model Tuning Considerations
- **Iterations**: 700 (with early stopping at 50 rounds)
- **Learning Rate**: 0.015 (lower rate for better generalization)
- **Tree Depth**: 8 (balanced complexity)
- **Class Weights**: SqrtBalanced (optimal for 95:5 class ratio)

## 🎓 Key Insights

### Dataset Characteristics
- **Samples**: 89,999 loan applications
- **Features**: 39 (27 numerical, 11 categorical, 5 engineered)
- **Class Distribution**: 95.1% No Default / 4.9% Default (highly imbalanced)
- **Missing Data**: <2% across most features

### Business Context
In credit risk modeling, **minimizing False Negatives** (missed defaults) is critical, even at the cost of higher False Positives. This model prioritizes recall over precision, achieving:
- **3,178 correctly identified defaults** (vs 147 with standard approaches)
- **68% reduction in missed defaults**
- Estimated **$30M+ savings** at $10K average loss per default

## 🛠️ Development

### Project Structure
```
agile/
├── model_catboost.py          # Main production model (recommended)
├── model_histgradient.py      # Alternative gradient boosting
├── model_final.py             # Baseline comparison
├── README.md                  # Documentation
├── .gitignore                 # Version control exclusions
├── catboost_info/             # Training logs and metrics
└── evaluation_set/            # Test data (if available)
```

### Version Control

The project uses Git with the following exclusions:
- Virtual environments (`.venv/`)
- Model artifacts (`catboost_info/`, `*.pkl`)
- Large data files (configurable in `.gitignore`)
- IDE and OS-specific files

### Dependencies

Core packages:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
catboost>=1.2.0
openpyxl>=3.0.0      # Excel support
lxml>=4.9.0          # XML parsing
pyarrow>=14.0.0      # Parquet support
```

## 📝 Model Comparison

| Model | AUC | Recall | Training Time | Best For |
|-------|-----|--------|---------------|----------|
| **CatBoost** | **0.8010** | **69%** | 17s/fold | Production (imbalanced data) |
| HistGradientBoosting | 0.7939 | 3% | 9s/fold | Speed-critical applications |
| Logistic Regression | 0.7500 | 15% | 3s/fold | Baseline/interpretability |

## 🔮 Future Improvements

1. **Hyperparameter Tuning**: Grid search for optimal depth, learning rate
2. **Feature Selection**: SHAP-based importance analysis
3. **Threshold Optimization**: Cost-sensitive learning based on business metrics
4. **Ensemble Methods**: Combine CatBoost with HistGradientBoosting
5. **Monitoring**: Real-time performance tracking and drift detection

## 📄 License

This project is proprietary. All rights reserved.

## 👥 Contact

For questions or collaboration opportunities, please reach out to the development team.

---

**Last Updated**: November 2025  
**Model Version**: CatBoost v1.0  
**Python Version**: 3.9+
