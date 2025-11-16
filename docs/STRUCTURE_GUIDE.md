# 📁 Project Structure Guide

## Overview

This document explains the standardized project structure following Python best practices (PEP 517, PEP 621) and modern development standards.

---

## Directory Structure

```
credit-scoring/
│
├── 📂 src/                      # Source code (main package)
│   ├── credit_scorer.py        # Core ML model class
│   ├── data_loader.py          # Data loading & merging utilities
│   ├── feature_engineering.py  # Feature creation (94 features)
│   ├── main.py                 # CLI interface
│   ├── analyze_errors.py       # Error analysis tools
│   ├── evaluate.py             # Model evaluation utilities
│   └── __init__.py             # Package initialization
│
├── 📂 tests/                    # Test suite (pytest)
│   ├── test_data_loader.py     # Unit tests for data loading
│   ├── test_integration.py     # End-to-end integration tests
│   ├── test_model_performance.py # Model validation tests
│   └── __init__.py             # Test package initialization
│
├── 📂 scripts/                  # Utility scripts
│   ├── run_example.sh          # Full pipeline demo script
│   ├── compare_thresholds.py   # Threshold optimization tool
│   └── README.md               # Scripts documentation
│
├── 📂 examples/                 # Usage examples
│   ├── basic_usage.py          # Simple training & prediction
│   ├── advanced_usage.py       # Advanced configuration
│   └── README.md               # Examples documentation
│
├── 📂 docs/                     # Documentation
│   ├── QUICKSTART.md           # 5-minute quick start guide
│   ├── USAGE_GUIDE.md          # Comprehensive usage guide
│   ├── PROJECT_STRUCTURE.md    # Technical architecture
│   ├── IMPLEMENTATION_SUMMARY.md # Implementation details
│   └── README_OLD.md           # Legacy README (backup)
│
├── 📂 data/                     # Training data (6 sources)
│   ├── application_metadata.csv # Application info + target variable
│   ├── loan_details.xlsx       # Loan information
│   ├── demographics.csv        # Customer demographics
│   ├── credit_history.parquet  # Credit history
│   ├── financial_ratios.jsonl  # Financial ratios
│   └── geographic_data.xml     # Geographic data
│
├── 📂 models/                   # Trained models
│   └── credit_model.pkl        # Production-ready model
│
├── 📂 outputs/                  # Prediction outputs
│   ├── predictions.csv         # Full predictions with probabilities
│   ├── predictions_simple.csv  # Simple submission format
│   ├── prediction_mismatches.csv # Error analysis
│   └── feature_importance.csv  # Feature importance rankings
│
├── 📂 notebooks/                # Jupyter notebooks (optional)
│   └── (analysis notebooks)    # Data exploration, visualization
│
├── 📄 pyproject.toml            # Modern Python package configuration (PEP 517, 621)
├── 📄 requirements.txt          # Production dependencies
├── 📄 MANIFEST.in               # Package distribution manifest
├── 📄 .gitignore                # Git ignore patterns
├── 📄 .flake8                   # Linting configuration
├── 📄 .pre-commit-config.yaml  # Pre-commit hooks
├── 📄 README.md                 # Main project README
└── 📄 LICENSE                   # MIT License

```

---

## Key Improvements

### ✅ Modern Python Standards

1. **pyproject.toml** (replaces setup.py)
   - PEP 517: Build system requirements
   - PEP 621: Project metadata
   - Tool configurations (black, isort, pytest, mypy)

2. **Organized Directory Structure**
   - Clear separation of concerns
   - Industry-standard layout
   - Easy navigation and maintenance

3. **Development Tools**
   - pre-commit hooks for code quality
   - black for code formatting
   - flake8 for linting
   - mypy for type checking
   - pytest for testing

---

## Installation Methods

### Method 1: Development Mode (Recommended)
```bash
pip install -e .
```
- Editable installation
- Changes reflected immediately
- Good for development

### Method 2: Standard Installation
```bash
pip install .
```
- Production installation
- Package installed in site-packages

### Method 3: Dependencies Only
```bash
pip install -r requirements.txt
```
- Just install dependencies
- Use for manual development

---

## Usage Patterns

### For Developers
```bash
# Clone and setup
git clone <repo>
cd credit-scoring
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/
```

### For Users
```bash
# Install package
pip install credit-scoring

# Use in Python
from credit_scorer import CreditScorer
scorer = CreditScorer()
```

### For Scripts
```bash
# Run example pipeline
./scripts/run_example.sh

# Run threshold optimization
python scripts/compare_thresholds.py
```

---

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_CASE`
- **Documentation**: `UPPER_CASE.md`

---

## Testing Strategy

```
tests/
├── test_data_loader.py      # Unit tests (fast)
├── test_integration.py      # Integration tests (slow)
└── test_model_performance.py # Performance validation
```

Run tests by category:
```bash
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests
pytest -m "not slow"    # Skip slow tests
```

---

## Documentation Structure

```
docs/
├── QUICKSTART.md           # New users start here
├── USAGE_GUIDE.md          # Detailed how-to guide
├── PROJECT_STRUCTURE.md    # Technical architecture
└── IMPLEMENTATION_SUMMARY.md # Implementation notes
```

---

## Development Workflow

1. **Make changes** in `src/`
2. **Add tests** in `tests/`
3. **Run tests**: `pytest`
4. **Format code**: `black . && isort .`
5. **Check quality**: `flake8 . && mypy src/`
6. **Commit** (pre-commit hooks run automatically)
7. **Push** to repository

---

## Continuous Integration

Pre-commit hooks automatically run:
- ✅ Code formatting (black, isort)
- ✅ Linting (flake8)
- ✅ Type checking (mypy)
- ✅ Tests (pytest)
- ✅ File checks (trailing whitespace, etc.)

---

## Best Practices Followed

1. ✅ **PEP 517** - Modern build system
2. ✅ **PEP 621** - Project metadata in pyproject.toml
3. ✅ **PEP 8** - Code style
4. ✅ **Type hints** - Better code documentation
5. ✅ **Docstrings** - Function/class documentation
6. ✅ **Tests** - Comprehensive test coverage
7. ✅ **Git** - Version control with proper .gitignore
8. ✅ **Documentation** - Clear, structured docs

---

## Migration from Old Structure

### What Changed?

| Old | New | Reason |
|-----|-----|--------|
| Root files scattered | Organized in folders | Better organization |
| setup.py | pyproject.toml | Modern standard (PEP 621) |
| No dev tools | .flake8, .pre-commit | Code quality |
| Mixed outputs | outputs/ folder | Clean root directory |
| Scripts in root | scripts/ folder | Clear separation |
| Docs in root | docs/ folder | Centralized documentation |

### Backward Compatibility

All existing code still works! Just update import paths if needed:

```python
# Still works
from credit_scorer import CreditScorer
from data_loader import merge_all_data
```

---

## Further Reading

- [PEP 517](https://peps.python.org/pep-0517/) - Build system requirements
- [PEP 621](https://peps.python.org/pep-0621/) - Project metadata
- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)

---

**Last Updated**: November 15, 2025  
**Version**: 1.0.0 (Standardized Structure)
