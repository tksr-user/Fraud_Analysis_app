import streamlit as st
import pandas as pd
import json
import os
import mlflow
from mlflow.tracking import MlflowClient

# AI Agent modules
from fraud_analysis_app.agent.run_agent_bedrock import run_agent_bedrock
from fraud_analysis_app.agent.create_comperison_prompt import fetch_mlflow_runs, create_comparison_prompt


# ------------------ Configuration ------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Fraud_Detection_Comparison_v5"
OPTUNA_DIR = "optuna_artifacts"
PROMPT_TEMPLATE = "fraud_analysis_app/agent/prompt_templates/comparison_prompt.txt"

# ------------------ Setup ------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

def get_all_runs():
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        return []
    runs = client.search_runs([experiment.experiment_id])
    return runs

def display_run_metrics(run):
    st.subheader("🔎 Model Details")
    st.write(f"**Run ID:** {run.info.run_id}")
    st.write(f"**Model:** {run.data.params['model']}")
    st.write(f"**Feature Set:** {run.data.params['feature_set']}")

    st.metric("Accuracy", f"{float(run.data.metrics['accuracy']):.4f}")
    st.metric("Precision", f"{float(run.data.metrics['precision']):.4f}")
    st.metric("Recall", f"{float(run.data.metrics['recall']):.4f}")
    st.metric("F1 Score", f"{float(run.data.metrics['f1_score']):.4f}")

def load_optuna_artifacts(model_name, feature_set_name):
    base_path = f"{OPTUNA_DIR}/{model_name}_{feature_set_name}"
    summary_path = os.path.join(base_path, "summary.json")
    history_path = os.path.join(base_path, "opt_history.png")
    param_path = os.path.join(base_path, "param_importance.png")
    trials_path = os.path.join(base_path, f"{model_name}_{feature_set_name}_trials.csv")

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    return summary, history_path, param_path, trials_path

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🕵️‍♂️ Fraud Detection ML Model Dashboard")

# ---------- Load & Display MLflow Runs ----------
runs = get_all_runs()

if not runs:
    st.warning("No runs found in MLflow.")
else:
    run_names = [f"{r.data.params['model']} | {r.data.params['feature_set']}" for r in runs]
    selected_idx = st.selectbox("Select a Run", range(len(run_names)), format_func=lambda i: run_names[i])
    selected_run = runs[selected_idx]

    display_run_metrics(selected_run)

    model_name = selected_run.data.params['model']
    feature_set_name = selected_run.data.params['feature_set']
    summary, history_img, param_img, trials_csv = load_optuna_artifacts(model_name, feature_set_name)

    # ---------- Optuna Summary ----------
    st.subheader("📋 Optuna Best Parameters")
    if summary:
        st.json(summary)
    else:
        st.warning("No Optuna summary found.")

    # ---------- Optuna Visualizations ----------
    st.subheader("📈 Optuna Visualizations")
    col1, col2 = st.columns(2)
    if os.path.exists(history_img):
        col1.image(history_img, caption="Optimization History")
    if os.path.exists(param_img):
        col2.image(param_img, caption="Parameter Importance")

    # ---------- Download Trials ----------
    if os.path.exists(trials_csv):
        with open(trials_csv, "rb") as f:
            st.download_button(
                label="Download Trial CSV",
                data=f,
                file_name=os.path.basename(trials_csv),
                mime="text/csv"
            )

# ------------------ AI Agent Comparison Section ------------------

st.markdown("---")
st.header("🤖 AI Agent: Compare ML Models")

df_runs = fetch_mlflow_runs(EXPERIMENT_NAME)

if df_runs.empty:
    st.warning("⚠️ No MLflow runs available to compare.")
else:
    target_run_id = st.selectbox("🎯 Select Target Model (Run ID)", df_runs["run_id"].tolist())

    if st.button("🧠 Run AI Agent"):
        with st.spinner("Generating comparison using Bedrock..."):
            try:
                mlflow_prompt = create_comparison_prompt(df_runs, target_run_id)

                response = run_agent_bedrock(prompt=mlflow_prompt)

                   

                st.success("✅ AI Agent Response")
                st.text_area("🧾 Model Comparison Report", value=response, height=400)

                with open("artifacts/ai_model_comparison.txt", "w", encoding="utf-8") as f:
                    f.write(response)

                st.download_button(
                    label="⬇️ Download AI Report",
                    data=response,
                    file_name="ai_model_comparison.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
