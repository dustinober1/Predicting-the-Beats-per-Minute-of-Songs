# BPM Prediction Project Configuration

# Data paths
DATA_DIR = "data"
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
OUTPUTS_DIR = "outputs"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

# Feature engineering parameters
N_PCA_COMPONENTS = 5
N_ICA_COMPONENTS = 3
N_CLUSTERS = 8
OUTLIER_CONTAMINATION = 0.1

# Model hyperparameters
LASSO_ALPHA = 0.1
RIDGE_ALPHA = 1.0
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 5

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Output file names
SUBMISSION_FILE = "submission_final.csv"
EXPERIMENTAL_TRAIN = "train_experimental.csv"
EXPERIMENTAL_TEST = "test_experimental.csv"
