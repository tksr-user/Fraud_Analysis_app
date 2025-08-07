import json
from fraud_analysis_app.agent.bedrock_client import get_bedrock_client

def run_agent_bedrock(prompt: str, model_id="amazon.titan-text-lite-v1") -> str:
    """
    Calls the Bedrock model with the given prompt and returns the generated response.
    """
    client = get_bedrock_client()

    body = json.dumps({
        "inputText": prompt,
        
    })

    response = client.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    result = json.loads(response['body'].read().decode())
    return result.get("generation", "[No response generated]")
