# Credit Default Prediction

ML model for credit default risk prediction using CatBoost.

## Performance

- **AUC-ROC**: 0.8041 (±0.0061)
- **Accuracy**: 93.44%
- **Precision**: 33% | **Recall**: 27%
- **Features**: 29 total (6 categorical, 5 engineered)

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn catboost openpyxl lxml pyarrow

# Run model
python model_catboost.py
```

## Data Sources

- `financial_ratios.jsonl` - Debt ratios, income
- `demographics.csv` - Age, employment, education
- `credit_history.parquet` - Payment history
- `loan_details.xlsx` - Loan details
- `application_metadata.csv` - Application data
- `geographic_data.xml` - Regional data

## Key Features

**Top 5 Important Features:**
1. credit_score (11.89)
2. annual_income (7.76)
3. age (7.40)
4. loan_type (4.90)
5. monthly_free_cash_flow (4.88)

**Engineered Features (5):**
- `debt_burden` - Combined monthly debt relative to income
- `credit_pressure` - Utilization × debt-to-income ratio
- `payment_stress` - Payment relative to free cash flow
- `credit_efficiency` - Credit score / utilization
- `delinquency_severity` - Delinquencies × DTI ratio

## Model Config

```python
CatBoostClassifier(
    iterations=700,
    learning_rate=0.015,
    depth=8,
    auto_class_weights='SqrtBalanced',
    eval_metric='AUC',
    cat_features=categorical_indices
)
```

## Dataset

- **Samples**: 89,999 loan applications
- **Class Distribution**: 94.9% No Default / 5.1% Default
- **Training Time**: ~60s (5-fold CV)