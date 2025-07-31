from setuptools import setup, find_packages

setup(
    name='fraud_analysis_app',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'mlflow',
        'arize-phoenix',
        'xgboost',
        'optuna',
        'gpt4all',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'run-training=scripts.run_training:main',
            'run-inference=scripts.run_inference:main',
            'run-agent=scripts.run_agent:main',
        ],
    },
)
