"""
Configuration file, containing the constants and settings for running the main file
"""
from sklearn.preprocessing import (
    PolynomialFeatures,
    QuantileTransformer,
    StandardScaler
)

CONFIG_FILE_PATHS = {
    'model_file_path': 'artifacts/model.pkl',
    'pipeline_file_path': 'artifacts/pipeline.jsonc',
    'data_file_path': 'data/dataset.parquet',
    'log_file_path': 'logs/failure.lo',
}

TRANSFORMER_CLASSES = {
    "PolynomialFeatures": PolynomialFeatures,
    "QuantileTransformer": QuantileTransformer,
    "StandardScaler": StandardScaler
}