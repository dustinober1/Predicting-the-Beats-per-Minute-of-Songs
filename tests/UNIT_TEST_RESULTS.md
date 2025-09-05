# Unit Test Results

Run date: 2025-09-05

Summary:

- Total tests: 2
- Passed: 2
- Failed: 0

Raw pytest output is available in `tests/test_results.txt`.

Key tests:

- `test_predict_script.py`: trains a RandomForest on `data/processed/train_experimental.csv` and asserts a model artifact is saved and loadable.
- `test_notebook_execution.py`: executes `notebooks/model_comparison.ipynb` with `jupyter nbconvert --execute` and asserts an executed notebook is produced.

Notes:

- The tests run on the repository's small sample data (10 samples in processed CSVs) and are intended as fast smoke tests for CI.
- If you add or change the notebook, re-run tests locally or via CI to confirm execution still succeeds.
