# ---------------------- ✅ Deploy any model to Arize ----------------------

import pandas as pd
import mlflow
import joblib
import os
from arize.pandas.logger import Client
from arize.utils.types import Schema, ModelTypes, Environments

# ✅ Arize credentials
space_id = "U3BhY2U6MjM3MTI6RThBTQ=="                  
api_key = "ak-8c93aa68-e105-4c23-b977-4ffb437fe7a5-rZPuli0UaGIrRAJ3x-OkK1sg_l5e5mFT"

# ✅ Arize model metadata
MODEL_ID = "logistic_fraud_model_2"
MODEL_VERSION = "v1"

# ✅ MLflow experiment and model name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "Fraud_Detection_Comparison_v5"
logistic_model_name = "LogisticRegression_first_half"  # <--- change this to any model+feature combo

# ✅ Get experiment and run
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{logistic_model_name}'",
    order_by=["start_time DESC"]
)

if runs.empty:
    raise ValueError(f"❌ No run found for model: {logistic_model_name}")
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
    y_test = y_test.squeeze()

# ✅ Load the trained feature names for the model
feature_file = f"artifacts/feature_names_{logistic_model_name}.pkl"
if not os.path.exists(feature_file):
    raise FileNotFoundError(f"❌ Feature file not found: {feature_file}")
trained_features = joblib.load(feature_file)

# ✅ Align X_test with trained features
# 1. Drop any extra columns
X_test = X_test[[col for col in X_test.columns if col in trained_features]]

# 2. Add missing columns with 0
for col in trained_features:
    if col not in X_test.columns:
        X_test[col] = 0

# 3. Reorder columns
X_test = X_test[trained_features]

# ✅ Predict
y_pred = model.predict(X_test)

# ✅ Prepare dataframe for Arize
df = X_test.copy()
df["prediction_id"] = [f"id_{i}" for i in range(len(df))]
df["prediction"] = y_pred
df["actual"] = y_test.values
df = df.reset_index(drop=True)

# ✅ Send to Arize
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
