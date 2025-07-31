import pandas as pd

def load_data(path='fraud_analysis_app/data/fraud_data.parquet'):
    '''Load fraud detection dataset from a Parquet file.'''
    return pd.read_parquet(path)
