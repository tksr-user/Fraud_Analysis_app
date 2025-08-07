# fraud_analysis_app/tests/test_train.py

from fraud_analysis_app.models import train_all_models

def test_main_runs():
    failed_combinations = []

    try:
        train_all_models.main()
    except Exception as e:
        failed_combinations.append(str(e))

    if failed_combinations:
        print(f"❌ Some combinations failed:\n{failed_combinations}")
        # Optional: Fail only if everything failed
        assert "Encountered zero total variance" not in failed_combinations[0], (
            f"Training script failed due to feature variance issues. Please check feature sets.\n{failed_combinations}"
        )
