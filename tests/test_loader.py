# fraud_analysis_app/tests/test_loader.py

from fraud_analysis_app.data.loader import load_data
import pandas as pd

def test_load_data_shape():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Class" in df.columns
