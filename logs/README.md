# Logs Directory

This directory contains execution logs from running BPM prediction scripts.

## Log Files

### Current Logs
- **`logs_experimental.txt`** - Output from experimental_approaches.py
- **`logs_pipeline.txt`** - Output from run_pipeline.py

## Log Content

### Experimental Logs
Contains output from advanced feature engineering:
- Feature creation progress
- Outlier detection results
- PCA/ICA component analysis
- Clustering statistics
- Quantile transformation results
- File generation confirmations

### Pipeline Logs
Contains output from main modeling pipeline:
- Data loading and preprocessing
- Feature engineering steps
- Model training progress
- Performance metrics
- Prediction generation
- File saving confirmations

## Usage

### Viewing Logs
```bash
# View recent experimental log
tail -f logs/logs_experimental.txt

# View recent pipeline log
tail -f logs/logs_pipeline.txt

# Search for specific information
grep "RMSE" logs/logs_pipeline.txt
grep "features created" logs/logs_experimental.txt
```

### Log Analysis
```bash
# Count errors
grep -i "error" logs/*.txt

# Check completion status
grep -i "complete" logs/*.txt

# View performance metrics
grep -E "(RMSE|MAE|R²)" logs/*.txt
```

## Log Management

### Automatic Generation
- Logs are automatically created when scripts run
- Previous logs are overwritten on each execution
- Use redirection to create timestamped logs:
  ```bash
  python scripts/run_pipeline.py > logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1
  ```

### Manual Management
- Archive important logs before re-running scripts
- Clear old logs periodically to save space
- Consider log rotation for production environments

### Debugging
Logs are essential for:
- Troubleshooting execution errors
- Verifying feature engineering results
- Confirming model performance
- Tracking file generation
- Understanding processing flow
