from setuptools import setup, find_packages

setup(
    name="fraud_analysis_app",
    version="0.1",
    packages=find_packages(include=["fraud_analysis_app", "fraud_analysis_app.*"]),
    include_package_data=True,
    install_requires=[
        "pandas",
        "scikit-learn",
        "mlflow",
        "xgboost",
        "optuna",
        "gpt4all",
        "joblib"
    ],
    entry_points={
        "console_scripts": [
            'run-training=fraud_analysis_app.models.train_all_models:main',
            "run-all-train=fraud_analysis_app.models.train_all_experiments:main",
            "run-inference = scripts.run_inference:main",
            "run-agent = scripts.run_agent:main"
        ]
    },
)
