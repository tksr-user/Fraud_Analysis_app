import os
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as opt_viz

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient


def main():
    # ✅ Set MLflow tracking + experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Fraud_Detection_Comparison_v3")

    # ✅ Load dataset from Parquet
    df = pd.read_parquet("fraud_analysis_app/data/fraud_data.parquet").head(1000)

    # ✅ Set target and preprocess
    target = "Class"
    X_full = df.drop(columns=[target])
    y = df[target]

    # ✅ Drop high-cardinality object columns
    for col in X_full.select_dtypes(include="object").columns:
        if X_full[col].nunique() > 100:
            print(f"Dropping column: {col} (unique: {X_full[col].nunique()})")
            X_full = X_full.drop(columns=col)

    X_full = pd.get_dummies(X_full)
    all_features = X_full.columns.tolist()

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(all_features, "artifacts/feature_names.pkl")

    # ✅ Define feature subsets
    feature_sets = {
        "all_features": all_features,
    }

    # ✅ Define models and hyperparameters
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=500),
            "params": {"C": [0.1, 1, 10]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {"n_estimators": [50, 100], "max_depth": [3, 5]}
        },
        "SVC": {
            "model": SVC(),
            "params": {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
        }
    }

    # ✅ Train each model
    for feature_set_name, feature_set in feature_sets.items():
        X = X_full[feature_set]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        os.makedirs("datas", exist_ok=True)
        X_test.to_parquet("fraud_analysis_app/data/X_test.parquet")
        y_test.to_frame(name="Class").to_parquet("fraud_analysis_app/data/y_test.parquet", index=False)

        for model_name in models:
            print(f"\n🔍 Running Optuna study for {model_name} with {feature_set_name}...")

            def objective(trial):
                params = {}
                if model_name == "LogisticRegression":
                    params["C"] = trial.suggest_categorical("C", [0.1, 1, 10])
                    model = LogisticRegression(**params, max_iter=500)
                elif model_name == "RandomForest":
                    params["n_estimators"] = trial.suggest_categorical("n_estimators", [50, 100])
                    params["max_depth"] = trial.suggest_categorical("max_depth", [3, 5])
                    model = RandomForestClassifier(**params)
                elif model_name == "SVC":
                    params["C"] = trial.suggest_categorical("C", [0.1, 1])
                    params["kernel"] = trial.suggest_categorical("kernel", ["linear", "rbf"])
                    model = SVC(**params)

                score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20, timeout=300)
            best_params = study.best_params

            # Train final model
            if model_name == "LogisticRegression":
                best_model = LogisticRegression(**best_params, max_iter=500)
            elif model_name == "RandomForest":
                best_model = RandomForestClassifier(**best_params)
            elif model_name == "SVC":
                best_model = SVC(**best_params)

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            with mlflow.start_run(run_name=f"{model_name}_{feature_set_name}"):
                mlflow.log_param("model", model_name)
                mlflow.log_param("feature_set", feature_set_name)
                mlflow.log_params(best_params)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)

                # ✅ Log model
                artifact_path = f"{model_name}_{feature_set_name}"
                mlflow.sklearn.log_model(best_model, artifact_path=artifact_path)

                client = MlflowClient()
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/{artifact_path}"
                registered_model_name = f"{model_name}_{feature_set_name}_Model"
                mlflow.register_model(model_uri, registered_model_name)

                # ✅ Log Optuna artifacts
                optuna_dir = f"optuna_artifacts/{model_name}_{feature_set_name}"
                os.makedirs(optuna_dir, exist_ok=True)

                with open(f"{optuna_dir}/summary.json", "w") as f:
                    json.dump({
                        "best_params": best_params,
                        "best_value": study.best_value,
                        "best_trial": study.best_trial.number
                    }, f, indent=4)

                # Save plots
                plt.figure()
                opt_viz.plot_optimization_history(study)
                plt.title("Optuna Optimization History")
                plt.savefig(f"{optuna_dir}/opt_history.png")
                plt.close()

                plt.figure()
                opt_viz.plot_param_importances(study)
                plt.title("Optuna Param Importance")
                plt.savefig(f"{optuna_dir}/param_importance.png")
                plt.close()

                # Log artifacts
                mlflow.log_artifact(f"{optuna_dir}/summary.json")
                mlflow.log_artifact(f"{optuna_dir}/opt_history.png")
                mlflow.log_artifact(f"{optuna_dir}/param_importance.png")

                print(f"✅ Run logged for {model_name} with {feature_set_name}")
                print("🚀 Run ID:", run_id)


# ✅ Only runs when script is directly executed
if __name__ == "__main__":
    main()
