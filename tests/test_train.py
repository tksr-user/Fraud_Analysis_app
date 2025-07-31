# fraud_analysis_app/tests/test_train.py

from fraud_analysis_app.models import train_all_models

def test_main_runs():
    try:
        train_all_models.main()
        assert True
    except Exception as e:
        assert False, f"Training script failed with error: {e}"
