"""
Pipeline for Vibration Analysis

This module handles loading data, preprocessing, and scoring using a trained model.
It includes proper error handling, logging, and configuration management.
"""
import joblib
import json
import logging
import os
import pickle
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
# from sklearn.svm import OneClassSVM, LinearSVC, NuSVR

from config import CONFIG_FILE_PATHS, TRANSFORMER_CLASSES


# Create the logs folder if it does not exist
logs_folder = 'logs/'
os.makedirs(logs_folder, exist_ok=True)

# Configure logging (for storing INFO, WARNING, ERROR, and CRITICAL messages.)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # format
    handlers=[
        logging.FileHandler(CONFIG_FILE_PATHS['log_file_path']), # save
        logging.StreamHandler() # print to console
    ]
)
# Create logger object
logger = logging.getLogger(__name__)


class DataLoader:
    """Loading properly the different asset types."""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
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
            logger.error(
                f'FileNotFoundError: Data file not found at {file_path}.'
                )
            raise FileNotFoundError('Data file not found at {file_path}.')
        
        data = pd.read_parquet(file_path)

        if data.empty:
            logger.error('ValueError: Loaded data is empty.')
            raise ValueError('Loaded data is empty.')

        return data
        
    @staticmethod
    def load_model(file_path: str) -> BaseEstimator:
        """Load model from cached JSONC configuration, removing comments.

        Args: 
            file_path: Path to model file

        Returns: 
            Loaded model

        Raises:
            ValueError: If model path is invalid or loading fails
        """
        config = DataLoader._load_and_parse_jsonc(file_path)
        # joblib is more suitable for sklearn models than pickle
        if (model_in_steps := config.get("steps", {}).get("model")) is None:
            logger.error('Model not found in config steps.')
            raise ValueError('Model not found in config steps.')
        
        try:
            return joblib.load(model_in_steps) 
        
        except:
            logger.error('Not possible to load model from config steps.')
            raise ValueError('Not possible to load model from config steps.')
        

    @staticmethod
    def load_pipeline(file_path: str) -> Pipeline:
        """Build pipeline from cached JSONC configuration.

        Args: 
            file_path: Path to pipeline configuration file

        Returns: 
            Configured sklearn Pipeline

        Raises:
            ValueError: For invalid config or missing transformers
        """
        
        # Load pipeline from cached jsonc
        pipeline_config = DataLoader._load_and_parse_jsonc(file_path)

        # Create pipeline steps
        steps = []
        
        # Add preprocessing steps
        for step_name, step_config in pipeline_config["steps"].items():
            if step_name == "model":
                continue  # Handle model separately
                
            # Get the transformer class and parameters
            transformer_name, params = next(iter(step_config.items()))

            if transformer_name not in TRANSFORMER_CLASSES.keys():
                logger.error(
                    f'Could not import transformer {transformer_name} as it is not included in `TRANSFORMER_CLASSES`.'
                    )
                raise ValueError(
                    f'Could not import transformer {transformer_name} as it is not included in `TRANSFORMER_CLASSES`.'
                    ) 
            transformer_class = TRANSFORMER_CLASSES[transformer_name]
            
            steps.append((step_name, transformer_class(**params)))

        # Create the pipeline
        pipeline = Pipeline(steps)
        
        return pipeline
    
    @staticmethod
    @lru_cache(maxsize=None)  # Cache parsed JSONC files
    def _load_and_parse_jsonc(file_path: str) -> dict:
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
            
            transformed_data = self.pipeline.fit_transform(data)
            
            return transformed_data
        
        except Exception as e:
            logger.error('Data preprocessing failed.')
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
                logger.error('No data to score.')
                raise RuntimeError('No data to score.')
            return self.model.predict(transformed_data)
        
        except Exception as e:
            logger.error(f'Scoring failed: {str(e)}.')
            raise RuntimeError('Model scoring failed.') from e


def run_pipeline() -> Optional[np.ndarray]:
    """
    Main execution function for standalone script operation.
    
    Returns:
        Model predictions or None if failed
    """
    try:
        logger.info('Starting scoring process...')
        
        # Load assets
        data = DataLoader.load_data(CONFIG_FILE_PATHS['data_file_path'])
        model = DataLoader.load_model(CONFIG_FILE_PATHS['pipeline_file_path'])
        pipeline = DataLoader.load_pipeline(
            CONFIG_FILE_PATHS['pipeline_file_path']
            )
        input_columns = ['vibration_x', 'vibration_y', 'vibration_z']
        
        # Score data
        scorer = ModelScorer(model, pipeline)
        predictions = scorer.score(data, input_columns)
        logger.info('Scoring completed successfully.')

        return predictions
        
    except Exception as e:
        logger.exception('Scoring process failed.')
        return None


if __name__ == '__main__':
    predictions = run_pipeline()
    if predictions is not None:
        print(predictions)
    else:
        print('Scoring failed - check logs for details.')