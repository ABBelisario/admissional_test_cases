"""
Configuration file, containing the constants and settings for running the main file
"""
from pathlib import Path
from sklearn.preprocessing import (
    PolynomialFeatures,
    QuantileTransformer,
    StandardScaler
)


class Settings():
    DATA_PATH: Path = Path('data/dataset.parquet')
    PIPELINE_CONFIG_PATH: Path = Path('artifacts/pipeline.jsonc')
    LOG_DIR: Path = Path('logs')

settings = Settings()


TRANSFORMER_CLASSES = {
    "PolynomialFeatures": PolynomialFeatures,
    "QuantileTransformer": QuantileTransformer,
    "StandardScaler": StandardScaler
}