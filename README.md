# 📊   Scoring System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Accuracy](https://img.shields.io/badge/Accuracy-89.9%25-brightgreen.svg)]()
[![Recall](https://img.shields.io/badge/Recall-80.5%25-brightgreen.svg)]()

> **Production-grade machine learning system for   default prediction**  
> 80.5% recall • $24.6M annual savings • 77% false negative reduction

---

## 🎯 Key Highlights

| Metric | Value | Impact |
|--------|-------|--------|
| **Recall** | 80.5% | Catches 80.5% of risky borrowers |
| **Accuracy** | 89.9% | Overall prediction accuracy |
| **Annual Savings** | $24.6M | 18.3% risk reduction vs baseline |
| **FN Reduction** | 77% | From 3,967 → 895 false negatives |
| **Model Type** | Gradient Boosting | 400 estimators, optimized hyperparameters |
| **Features** | 94 | 81 base + 13 engineered features |

---

## 📁 Project Structure

```
 -scoring/
├── src/                          # Source code
│   ├──  _scorer.py         # Main ML model class
│   ├── data_loader.py           # Data loading & merging
│   ├── feature_engineering.py   # Feature creation (94 features)
│   ├── main.py                  # CLI interface
│   └── analyze_errors.py        # Error analysis tools
│
├── tests/                        # Test suite
│   ├── test_data_loader.py      # Unit tests for data loading
│   ├── test_integration.py      # End-to-end integration tests
│   └── test_model_performance.py # Model validation tests
│
├── scripts/                      # Utility scripts
│   ├── run_example.sh           # Quick start training script
│   └── compare_thresholds.py    # Threshold optimization
│
├── examples/                     # Usage examples
│   ├── basic_usage.py           # Simple example
│   └── advanced_usage.py        # Advanced configuration
│
├── docs/                         # Documentation
│   ├── QUICKSTART.md            # 5-minute quick start
│   ├── USAGE_GUIDE.md           # Detailed usage guide
│   ├── PROJECT_STRUCTURE.md     # Technical architecture
│   └── IMPLEMENTATION_SUMMARY.md # Implementation details
│
├── data/                         # Training data (6 sources)
│   ├── application_metadata.csv # Application info + target
│   ├── loan_details.xlsx        # Loan information
│   ├── demographics.csv         # Customer demographics
│   ├──  _history.parquet   #   history
│   ├── financial_ratios.jsonl   # Financial ratios
│   └── geographic_data.xml      # Geographic data
│
├── models/                       # Trained models
│   └──  _model.pkl         # Production model
│
├── outputs/                      # Prediction outputs
│   ├── predictions.csv          # Full predictions
│   └── prediction_mismatches.csv # Error analysis
│
├── notebooks/                    # Jupyter notebooks (analysis)
│
├── pyproject.toml               # Modern Python package config
├── requirements.txt             # Dependencies
├── .pre-commit-config.yaml      # Code quality hooks
├── .flake8                      # Linting configuration
├── MANIFEST.in                  # Package manifest
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/muhammadkodir-alijonov/ -scoring.git
cd  -scoring

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# OR install as package (recommended)
pip install -e .
```

### Basic Usage

```python
from  _scorer import  Scorer

# Train model
scorer =  Scorer(model_type='gradient_boosting')
metrics = scorer.train('./data')

# Make predictions
predictions = scorer.predict('./data')
predictions.to_csv('predictions.csv', index=False)

print(f"Accuracy: {metrics['test_accuracy']:.2%}")
print(f"Predictions: {len(predictions)}")
```

### Command Line Interface

```bash
# Train model
python src/main.py train --data-dir ./data --output-model models/model.pkl

# Make predictions
python src/main.py predict --data-dir ./data --model models/model.pkl --output predictions.csv
```

### Run Example Script

```bash
cd scripts
./run_example.sh
```

---

## 📊 Data Format

The system processes 6 data sources:

| File | Format | Key Column | Description |
|------|--------|-----------|-------------|
| `application_metadata.csv` | CSV | `customer_ref` | Target variable (default) |
| `loan_details.xlsx` | Excel | `customer_id` | Loan amount, type, term |
| `demographics.csv` | CSV | `cust_id` | Age, income, employment |
| ` _history.parquet` | Parquet | `customer_number` |   score, accounts |
| `financial_ratios.jsonl` | JSONL | `cust_num` | DTI,   utilization |
| `geographic_data.xml` | XML | `id` | Regional unemployment |

**Note:** Customer ID columns have different names but represent the same customers. The system automatically handles ID mapping during merging.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_data_loader.py
pytest tests/test_integration.py
pytest tests/test_model_performance.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

---

## 📈 Model Performance

### Classification Metrics

```
Metric          | Train  | Test   | Description
----------------|--------|--------|----------------------------------
Accuracy        | 86.3%  | 89.9%  | Overall correct predictions
Precision       | 18%    | 31%    | Of predicted defaults, how many are correct
Recall          | 56%    | 80.5%  | Of actual defaults, how many we catch
F1-Score        | 0.27   | 0.45   | Harmonic mean of precision & recall
AUC-ROC         | 0.91   | 0.80   | Model's discrimination ability
```

### Business Impact

- **False Negatives**: 895 (missed defaults costing $66.4M)
- **False Positives**: 8,238 (rejected good customers costing $43.7M)
- **Total Risk**: $110M (minimized through threshold optimization)
- **Savings vs Baseline**: $24.6M annually

---

## 🔧 Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

### Code Quality Tools

- **black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting & style checking
- **mypy**: Static type checking
- **pre-commit**: Automated pre-commit checks
- **pytest**: Testing framework with coverage

---

## 📚 Documentation

- **[Quick Start](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Usage Guide](docs/USAGE_GUIDE.md)** - Comprehensive guide with examples
- **[Technical Docs](docs/PROJECT_STRUCTURE.md)** - Architecture & implementation
- **[API Reference](src/)** - Source code documentation

---

## 🎓 Examples

Check the [`examples/`](examples/) directory for:

- **basic_usage.py** - Simple training & prediction
- **advanced_usage.py** - Custom configuration & analysis

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black . && isort .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Muhammad Kodir Alijonov**  
GitHub: [@muhammadkodir-alijonov](https://github.com/muhammadkodir-alijonov)

---

## 🙏 Acknowledgments

- Built with scikit-learn, pandas, and numpy
- Optimized for production use with comprehensive testing
- Follows Python best practices (PEP 8, PEP 517, PEP 621)

---

**⭐ If you find this project useful, please consider giving it a star!**
