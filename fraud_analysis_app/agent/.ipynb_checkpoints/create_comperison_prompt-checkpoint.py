# fraud_analysis_app/agent/create_prompt_bedrock.py
import mlflow
import pandas as pd


def fetch_mlflow_runs(experiment_name: str) -> pd.DataFrame:
    """
    Fetch all runs from a given MLflow experiment as a DataFrame.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id])


def create_comparison_prompt(df: pd.DataFrame, target_run_id: str) -> str:
    """
    Create a well-formatted prompt for comparing all models to the target model.
    Includes handling for missing metrics (N/A).
    """
    target_row = df[df["run_id"] == target_run_id].iloc[0]

    prompt_lines = []
    prompt_lines.append("You are an AI model evaluator.")
    prompt_lines.append(
        f"Compare all ML models in this experiment to the TARGET model (Run ID: {target_run_id})."
    )
    prompt_lines.append("Use the metrics: Accuracy, Precision, and Recall.")
    prompt_lines.append("Provide a ranking from best to worst and explain why.\n")

    prompt_lines.append("=== ALL MODELS ===")

    for idx, row in df.iterrows():
        if row["run_id"] == target_run_id:
            continue

        model_name = row.get("params.model", "Unknown Model")
        prompt_lines.append(f"\nModel {idx + 1}: {model_name}")
        prompt_lines.append(f"Run ID: {row['run_id']}")

        for metric in ["accuracy", "precision", "recall"]:
            col = f"metrics.{metric}"
            if col in row and pd.notnull(row[col]):
                prompt_lines.append(f"{metric.capitalize()}: {row[col]:.4f}")
            else:
                prompt_lines.append(f"{metric.capitalize()}: N/A")

    prompt_lines.append("\n=== TARGET MODEL ===")
    prompt_lines.append(f"Run ID: {target_run_id}")
    for metric in ["accuracy", "precision", "recall"]:
        col = f"metrics.{metric}"
        if col in target_row and pd.notnull(target_row[col]):
            prompt_lines.append(f"{metric.capitalize()}: {target_row[col]:.4f}")
        else:
            prompt_lines.append(f"{metric.capitalize()}: N/A")

    prompt_lines.append(
        "\nPlease rank all models from best to worst based on the above metrics "
        "and provide a clear, concise explanation for the ranking."
    )

    return "\n".join(prompt_lines)
