import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import uuid
from datetime import datetime
import os
import kagglehub
# Download dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

# Load data
csv_path = os.path.join(path, "creditcard.csv")
df = pd.read_csv(csv_path)
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# Check missing values
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Drop original columns
df = df.drop(['Amount', 'Time'], axis=1)

# Add required columns
df["TransactionID"] = [str(uuid.uuid4()) for _ in range(len(df))]
df["EventTime"] = datetime.utcnow().isoformat()

# Ensure folder exists and save-
parquet_path =  r"C:/Users/aman1/Fraud_Analysis_app/fraud_analysis_app/data/fraud_data.parquet" #update location to save file 
df.to_parquet(parquet_path, index=False)
print("✅ Parquet file saved:", parquet_path)

# Read Parquet file into DataFrame
df = pd.read_parquet(parquet_path)
print("✅ Parquet file loaded successfully.")
