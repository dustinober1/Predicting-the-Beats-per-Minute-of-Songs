import tempfile
from pathlib import Path
import joblib
import os

from scripts.predict_bpm import main


def test_predict_script_trains_and_saves_model():
    train_path = Path('data/processed/train_experimental.csv')
    assert train_path.exists(), 'Processed train CSV is required for this test'

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'rf_model.pkl'
        main(str(train_path), str(out))
        assert out.exists(), 'Model file was not created'
        # Ensure model can be loaded
        m = joblib.load(out)
        assert hasattr(m, 'predict')
