# Configuration Directory

This directory contains configuration files and parameters for the BPM prediction project.

## Configuration Files

### `config.py`
Main configuration file containing:
- **Data paths** - Input and output file locations
- **Model parameters** - Random seeds, validation splits
- **Feature engineering parameters** - PCA components, clustering settings
- **Model hyperparameters** - Alpha values, tree depths, etc.
- **Logging configuration** - Log levels and formats
- **Output file names** - Standardized naming conventions

### `feature_config.py`
Feature-specific configuration containing:
- **Column mappings** - Test to train dataset standardization
- **Feature groups** - Audio, rhythmic, content, temporal features
- **Feature lists** - Organized by feature type
- **Exclusion lists** - Non-feature columns to ignore
- **Target definition** - Target column specification

## Usage

### Importing Configuration
```python
# Import main config
from config.config import *

# Import feature config
from config.feature_config import TEST_COLUMN_MAPPING, ALL_NUMERIC_FEATURES

# Use in scripts
train_df = pd.read_csv(TRAIN_FILE)
test_df = test_df.rename(columns=TEST_COLUMN_MAPPING)
```

### Modifying Parameters
1. Edit configuration files directly
2. All scripts will automatically use updated parameters
3. No need to modify individual script files

### Adding New Configurations
1. Add new parameters to appropriate config file
2. Import in scripts that need them
3. Document new parameters in this README

## Configuration Categories

### Data Configuration
- File paths and naming conventions
- Data loading parameters
- Output directories

### Model Configuration  
- Algorithm hyperparameters
- Training parameters
- Validation settings

### Feature Configuration
- Feature engineering parameters
- Column mappings and transformations
- Feature group definitions

### System Configuration
- Logging settings
- Random seeds for reproducibility
- Performance parameters
