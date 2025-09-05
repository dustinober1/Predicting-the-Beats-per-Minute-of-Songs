# Scripts Directory

This directory contains all executable scripts for the BPM prediction project.

## Scripts Overview

### Core Scripts
- **`experimental_approaches.py`** - Advanced feature engineering with 53+ features
- **`run_pipeline.py`** - Main modeling pipeline with multiple algorithms
- **`run_complete_evaluation.py`** - Comprehensive evaluation and analysis
- **`project_summary.py`** - Final project summary and results

## Usage

### Run Individual Scripts
```bash
# From project root
python scripts/experimental_approaches.py
python scripts/run_pipeline.py
python scripts/run_complete_evaluation.py
python scripts/project_summary.py
```

### Run via Main Entry Point
```bash
# From project root
python main.py --experimental     # Run experimental_approaches.py
python main.py --pipeline         # Run run_pipeline.py
python main.py --evaluate         # Run run_complete_evaluation.py
python main.py --summary          # Run project_summary.py
python main.py --run-all          # Run all scripts in sequence
```

## Script Dependencies

All scripts depend on:
- `src/` modules for core functionality
- `data/` for input datasets
- `config/` for configuration parameters
- `outputs/` for saving results

## Execution Order

For best results, run scripts in this order:
1. `experimental_approaches.py` - Generate enhanced features
2. `run_pipeline.py` - Train models and make predictions
3. `run_complete_evaluation.py` - Analyze results
4. `project_summary.py` - Generate final summary
