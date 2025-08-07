# fraud_analysis_app/agent/compare_with_bedrock.py

import mlflow
import pandas as pd

from fraud_analysis_app.agent.create_prompt_bedrock import create_comparison_prompt
from fraud_analysis_app.agent.run_bedrock_agent import run_bedrock_agent

# Step 1: Load experiment
EXPERIMENT_NAME = "Fraud_Detection_Comparison_v12"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if not experiment:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

df_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Step 2: Pick best run by accuracy
best_run = df_runs.sort_values("metrics.accuracy", ascending=False).iloc[0]
target_run_id = best_run["run_id"]

# Step 3: Generate prompt
prompt = create_comparison_prompt(df_runs, target_run_id)

# Step 4: Run through Bedrock agent
response = run_bedrock_agent(prompt)

# Step 5: Show result
print("\n=== AI Agent Response ===\n")
print(response)
