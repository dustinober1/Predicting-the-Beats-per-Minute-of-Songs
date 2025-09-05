# Models Directory

This directory contains trained models and model artifacts for the BPM Prediction Project.

## Structure

```
models/
├── README.md              # This file
├── ridge_model.pkl        # Trained Ridge regression model
├── lasso_model.pkl        # Trained Lasso regression model
├── xgboost_model.pkl      # Trained XGBoost model
└── model_metrics.json    # Model performance metrics
```

## Model Files

- **ridge_model.pkl**: Best performing model (RMSE: 9.26)
- **lasso_model.pkl**: Alternative regularized model
- **xgboost_model.pkl**: Gradient boosting model
- **model_metrics.json**: Performance metrics for all models

## Usage

Models can be loaded using pickle:

```python
import pickle

# Load the best model
with open('models/ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)
```
