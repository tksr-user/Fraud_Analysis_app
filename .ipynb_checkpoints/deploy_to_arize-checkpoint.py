# ---------------------- ✅ Deploy Logistic Regression to Arize ----------------------
import pandas as pd
import mlflow
import arize
import joblib
import os
from arize.pandas.logger import Client
from arize.utils.types import Schema, ModelTypes, Environments

# ✅ Arize credentials
space_id = "U3BhY2U6MjM3MTI6RThBTQ=="                  
api_key = "ak-8c93aa68-e105-4c23-b977-4ffb437fe7a5-rZPuli0UaGIrRAJ3x-OkK1sg_l5e5mFT"

# ✅ Arize model metadata
MODEL_ID = "logistic_fraud_model"
MODEL_VERSION = "v1"

# ✅ MLflow model details
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment = mlflow.get_experiment_by_name("Fraud_Detection_Comparison_v2")
logistic_model_name = "LogisticRegression_all_features"
# Find run ID for this model
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{logistic_model_name}'",
    order_by=["start_time DESC"]
)


if runs.empty:
    print("❌ No run found for LogisticRegression_all_features")
else:
    run_id = runs.iloc[0]["run_id"]
    print("✅ Deploying run:", run_id)

    # ✅ Load model from MLflow
    model_uri = f"runs:/{run_id}/{logistic_model_name}"
    model = mlflow.sklearn.load_model(model_uri)

    # ✅ Load test data
    X_test = pd.read_parquet("fraud_analysis_app/data/X_test.parquet")
    y_test = pd.read_parquet("fraud_analysis_app/data/y_test.parquet")
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()  # ensure it's a Series

    # ✅ Make predictions
    y_pred = model.predict(X_test)

    # Create dataframe for Arize
    df = X_test.copy()
    df["prediction_id"] = [f"id_{i}" for i in range(len(df))]
    df["prediction"] = y_pred
    df["actual"] = y_test.values
    df = df.reset_index(drop=True)
    # Log to Arize
    arize_client = Client(space_id=space_id, api_key=api_key)
    schema = Schema(
        prediction_id_column_name="prediction_id",
        prediction_label_column_name="prediction",
        actual_label_column_name="actual"
    )
    
   

    response = arize_client.log(
        dataframe=df,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        model_type=ModelTypes.BINARY_CLASSIFICATION,
        environment=Environments.PRODUCTION,
        schema=schema
    )

    print("🚀 Arize upload response:", response)
