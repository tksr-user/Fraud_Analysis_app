# fraud_analysis_app/agent/bedrock_client.py
# fraud_analysis_app/agent/bedrock_client.py
import boto3

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="ap-south-1")

