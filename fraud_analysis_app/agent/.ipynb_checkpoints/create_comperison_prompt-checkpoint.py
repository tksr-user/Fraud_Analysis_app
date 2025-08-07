# fraud_analysis_app/agent/create_prompt_bedrock.py
import mlflow
import pandas as pd
def fetch_mlflow_runs(experiment_name: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return df
def create_comparison_prompt(df: pd.DataFrame, target_run_id: str) -> str:
    target_row = df[df["run_id"] == target_run_id].iloc[0]

    prompt = f"Compare all ML models to the target model (Run ID: {target_run_id}).\n"
    prompt += "Use metrics: accuracy, precision, recall.\n\n"

    for idx, row in df.iterrows():
        if row["run_id"] == target_run_id:
            continue
        prompt += f"Model {idx + 1} ({row['params.model']}):\n"
        prompt += f" - Run ID: {row['run_id']}\n"
        for metric in ["accuracy", "precision", "recall"]:
            col = f"metrics.{metric}"
            if col in row and pd.notnull(row[col]):
                prompt += f" - {metric.capitalize()}: {row[col]:.4f}\n"

    prompt += "\nTARGET MODEL:\n"
    prompt += f"Run ID: {target_run_id}\n"
    prompt += f"Accuracy: {target_row['metrics.accuracy']:.4f}\n"
    prompt += f"Precision: {target_row['metrics.precision']:.4f}\n"
    prompt += f"Recall: {target_row['metrics.recall']:.4f}\n"

    prompt += "\nRank all models from best to worst, and explain why."
    return prompt
