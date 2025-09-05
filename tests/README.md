# Tests Directory

This directory contains unit tests and integration tests for the BPM Prediction Project.

## Structure

```
tests/
├── __init__.py
├── test_data_preprocessing.py    # Tests for data preprocessing functions
├── test_feature_engineering.py   # Tests for feature engineering
├── test_models.py                # Tests for model functionality
└── test_utils.py                 # Tests for utility functions
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest tests/ --cov=src
```

## Test Requirements

Make sure to install pytest and related packages:

```bash
pip install pytest pytest-cov
```
