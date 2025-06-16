"""
Pipeline for Vibration Analysis

This module handles loading data, preprocessing, and scoring using a trained model.
It includes proper error handling, logging, and configuration management.
"""
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from config import settings, TRANSFORMER_CLASSES


def configure_logging():
    """
    Configure logging (for storing INFO, WARNING, ERROR, and CRITICAL messages.)
    """

    # Create the logs folder if it does not exist
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file_name = 'failure.lo'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # format
        handlers=[
            logging.FileHandler(f'{settings.LOG_DIR}/{log_file_name}'), # save
            logging.StreamHandler() # print to console
        ]
    )


def load_parquet(file_path: str, required_cols: list[str]) -> pd.DataFrame:
    """
    Load parquet data file.
    
    Args: 
        file_path: Path to parquet file

    Returns: 
        Loaded dataframe
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data is empty
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError('Data file not found at {file_path}.')
    
    data = pd.read_parquet(file_path)

    # Validate data
    if data.empty:
        raise ValueError('Loaded data is empty.')

    # Validate schema
    if not set(required_cols).issubset(data.columns):
        raise ValueError(
            f'Missing required columns: {required_cols - set(data.columns)}.'
            )
    return data


class PipelineConfig:
    """Configure model and pipeline."""

    def __init__(self, file_path: Path):
        """Load pipeline from cached jsonc
        
        Args: 
            file_path: Path to model file
        """
        self.pipeline_config = self._load_and_parse_jsonc(file_path)
        
    def load_model(self) -> BaseEstimator:
        """Load model from cached JSONC configuration, removing comments.

        Returns: 
            Loaded model

        Raises:
            ValueError: If model path is invalid or loading fails
        """
        # joblib is more suitable for loading sklearn models than pickle
        if (
            model_in_steps := self.pipeline_config.get('steps', {}).get('model')
            ) is None:
            raise ValueError(
                'Model not found under `steps` in pipeline configuration.'
                )
        try:
            return joblib.load(model_in_steps) 
        
        except:
            raise ValueError(
                'Not possible to load model from pipeline configuration steps.'
                )
        

    def build_pipeline(self) -> Pipeline:
        """Build pipeline from cached JSONC configuration.

        Returns: 
            Configured sklearn Pipeline

        Raises:
            ValueError: For invalid config or missing transformers
        """
        # Create pipeline steps
        steps = []
        
        # Add preprocessing steps
        for step_name, step_config in self.pipeline_config['steps'].items():
            if step_name == 'model':
                continue  # Handle model separately
                
            # Get the transformer class and parameters
            transformer_name, params = next(iter(step_config.items()))

            if transformer_name not in TRANSFORMER_CLASSES.keys():
                raise ValueError(
                    f'Could not import transformer {transformer_name} as it is not included in `TRANSFORMER_CLASSES`.'
                    ) 
            transformer_class = TRANSFORMER_CLASSES[transformer_name]
            
            steps.append((step_name, transformer_class(**params)))

        return Pipeline(steps)
    
    @staticmethod
    @lru_cache(maxsize=None)  # Cache parsed JSONC files
    def _load_and_parse_jsonc(file_path: Path) -> dict[str, dict[str, Any]]:
        """Private method to load, parse and cache JSONC files.
        
        Args:
            file_path: Path to JSONC file
            
        Returns:
            Parsed dictionary from JSONC
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON parsing fails
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found at {file_path}.')
        
        with open(file_path) as f:
            content = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', f.read())

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in file {file_path}: {str(e)}')


class ModelScorer:
    """Handles model scoring with proper data validation and transformation."""
    
    def __init__(self, model: BaseEstimator, pipeline: Pipeline):
        """
        Initialize scorer with model and pipeline.
        
        Args:
            model: Trained model instance
            pipeline: Data preprocessing pipeline
        """
        self.model = model
        self.pipeline = pipeline

    def preprocess_data(self, data: pd.DataFrame):
        """Preprocess input data using the pipeline.
        
        Args:
            data: Input data to be transformed
            
        Returns:
            Transformed data

        Raises:
            ValueError: If transformation fails
        """
        try:
            
            return self.pipeline.fit_transform(data)
            
        except Exception as e:
            raise ValueError('Data preprocessing failed.') 
            
        
    def score(self, data: pd.DataFrame, input_columns: list[str]) -> np.ndarray:
        """Score input data using the model.
        
        Args:
            data: Input data for scoring
            input_columns: columns names to be included as input
            
        Returns:
            Model predictions

        Raises:
            RuntimeError: If scoring fails
        """

        work_data = data[input_columns].copy()
        try:
            transformed_data = self.preprocess_data(work_data)
            if len(transformed_data) == 0:
                raise RuntimeError('No data to score.')
            return self.model.predict(transformed_data)
        
        except Exception as e:
            raise RuntimeError('Model scoring failed.') from e


def run_pipeline() -> Optional[np.ndarray]:
    """
    Main execution function for standalone script operation.
    
    Returns:
        Model predictions or None if failed
    """
    # Create logger object
    configure_logging()
    logger = logging.getLogger(__name__)
    try:
        logger.info('Starting scoring process...')
        
        # Load assets
        input_columns = ['vibration_x', 'vibration_y', 'vibration_z']
        data = load_parquet(
            file_path=settings.DATA_PATH, required_cols=input_columns
            ) 
        pipeline_config = PipelineConfig(
            settings.PIPELINE_CONFIG_PATH
            )
        model = pipeline_config.load_model()
        pipeline = pipeline_config.build_pipeline()
        
        # Score data
        scorer = ModelScorer(model, pipeline)
        predictions = scorer.score(data, input_columns)
        logger.info('Scoring completed successfully.')
        
        return predictions
        
    except Exception as e:
        logger.exception(f'Scoring process failed ({str(e)}) - check logs for details.')
        raise


if __name__ == '__main__':
    print(run_pipeline())
    