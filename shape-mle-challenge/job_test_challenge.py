
import pandas as pd


from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM, LinearSVC, NuSVR
import datetime
from sklearn.preprocessing import *

from sklearn.pipeline import Pipeline
import pickle
import numpy as np

pipeline_file_path = 'artifacts/pipeline.jsonc'
data_file_path = 'data/dataset.parquet'


# Load any asset
def load(type, **kwargs):
    if (type == 'data') :
        # TODO: Implement a method to read the data file and return a dataframe
        ...
    if (type == 'model'):
        with open('artifacts/pipeline.jsonc', 'r') as f:
            str_json = '\n'.join(f.readlines()[3:])
        import json
        with open(json.loads(str_json)["steps"]['model'], 'rb') as f:
            return pickle.load(f)
    if (type == 'pipeline'):
        raise NotImplementedError()
    else:
        return None


def _log_failure( e ) :
    LOG_DUMP_PATH = 'logs/failure.log'
    with open(LOG_DUMP_PATH, 'a') as fLog:
        fLog.write(f'{datetime.datetime.now()} - Failure: %s\n' % (str(e)))
def load_pipeline(file_path: str) -> Pipeline:
    # TODO: Implement the function body to build the sklearn pipeline from the jsonc specification file
    pass

def score():
    """
    This function should score the model on the test data and return the score.
    """
    try:
        m = load('model')
        data = load('data' )
        pipe = load_pipeline(pipeline_file_path)

        data = data[['vibration_x', 'vibration_y', 'vibration_z']]
        # tr_data["vibration_x"].replace({np.nan: 0}, inplace=True)
        tr_data = pipe.fit_transform(data)

        if (not len(tr_data)):
            raise RuntimeError('No data to score')
        if (not hasattr(m, 'predict')):
            raise Exception('Model does not have a score function')



        return m.predict(tr_data)
    except Exception as e:
        print(e)
        _log_failure(e)

if __name__ == '__main__':
    print(score())
