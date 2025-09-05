"""Train a RandomForest on processed data and save a trained model for inference.

Usage:
    python scripts/predict_bpm.py --train-path data/processed/train_experimental.csv --out models/rf_model.pkl
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib


def prepare_numeric(X: pd.DataFrame, ref_columns=None) -> pd.DataFrame:
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.fillna(0)
    if ref_columns is not None:
        X_num = X_num.reindex(columns=ref_columns, fill_value=0)
    return X_num


def main(train_path, out_path):
    train_path = Path(train_path)
    out_path = Path(out_path)
    df = pd.read_csv(train_path)
    # find target
    target = None
    for name in ['beats_per_minute','bpm','tempo']:
        if name in df.columns:
            target = name
            break
    if target is None:
        possible = [c for c in df.columns if 'bpm' in c.lower() or 'tempo' in c.lower()]
        if possible:
            target = possible[0]
    if target is None:
        raise RuntimeError('No target column found in training data')

    y = df[target].values
    X = df.drop(columns=[target])
    X = prepare_numeric(X)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    val_mae = np.mean(np.abs(y_val - model.predict(X_val)))
    print(f'Trained RandomForest. Validation MAE (approx): {val_mae:.4f}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print('Saved model to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', default='data/processed/train_experimental.csv')
    parser.add_argument('--out', default='models/rf_model.pkl')
    args = parser.parse_args()
    main(args.train_path, args.out)
