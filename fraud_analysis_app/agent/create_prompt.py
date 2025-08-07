import mlflow
import pandas as pd
from gpt4all import GPT4All

# Constants
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Fraud_Detection_Comparison_v5"

# Connect to local MLflow server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def fetch_mlflow_runs(experiment_name: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs_df

def create_comparison_prompt(df: pd.DataFrame, target_run_id: str) -> str:
    if df.empty or target_run_id not in df["run_id"].values:
        return "No valid runs or invalid target run ID."

    target_row = df[df["run_id"] == target_run_id].iloc[0]
    prompt = (
        f"Compare the following ML models to the target model (Run ID: {target_run_id}). "
        "Use accuracy, precision, and recall. Then rank all models from best to worst and explain your ranking.\n\n"
    )

    for idx, row in df.iterrows():
        if row["run_id"] == target_run_id:
            continue
        model_type = row.get("params.model", "N/A")
        prompt += f"Model {idx + 1} ({model_type}):\n"
        prompt += f" - Run ID: {row['run_id']}\n"
        prompt += f" - Model Type: {model_type}\n"
        for metric in ["accuracy", "precision", "recall"]:
            col = f"metrics.{metric}"
            if col in row and pd.notnull(row[col]):
                prompt += f" - {metric.capitalize()}: {row[col]:.4f}\n"
        prompt += "\n"

    prompt += (
        f"\n--- TARGET MODEL METRICS ---\n"
        f"Target Model ({df[df['run_id'] == target_run_id].index[0] + 1}):\n"
        f" - Run ID: {target_run_id}\n"
        f" - Accuracy: {target_row['metrics.accuracy']:.4f}\n"
        f" - Precision: {target_row['metrics.precision']:.4f}\n"
        f" - Recall: {target_row['metrics.recall']:.4f}\n"
    )

    prompt += (
        "\nRank all models from best to worst compared to the target model "
        "using both model name and Run ID for clarity."
    )

    return prompt
        
   
# Choose a specific run ID to compare others against (e.g., best Logistic Regression model)
df_runs = fetch_mlflow_runs("LogisticRegression_all_features")
target_run_id = df_runs.iloc[0]["run_id"]  # or pick based on best f1_score, etc.

# Load your local model — path must match your installed model
model = GPT4All("Llama-3.2-3B-Instruct-Q4_0.gguf")

# Generate response from prompt
prompt = create_comparison_prompt(df_runs, target_run_id)
response = model.generate(prompt, max_tokens=2048, temp=0.7)

with open("artifacts/ai_model_comparison.txt", "w", encoding="utf-8") as f:
    f.write(response)

print("AI Agent Response:")
print(response)