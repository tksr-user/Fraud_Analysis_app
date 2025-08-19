import json
import logging
from fraud_analysis_app.agent.bedrock_client import get_bedrock_client

logging.basicConfig(level=logging.DEBUG)

def run_agent_bedrock(prompt: str, model_id="amazon.titan-text-lite-v1") -> str:
    """
    Calls the Bedrock Titan model with the given prompt and returns the generated response.
    """
    client = get_bedrock_client()

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 800,
            "temperature": 0.7,
            "stopSequences": []
        }
    })

    try:
        logging.debug(f"Sending request to Bedrock model: {model_id}")
        logging.debug(f"Prompt: {prompt}")

        response = client.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )

        raw_response = response["body"].read().decode()
        logging.debug(f"Raw Bedrock response: {raw_response}")

        result = json.loads(raw_response)

        if "results" in result and len(result["results"]) > 0:
            return result["results"][0].get("outputText", "").strip()

        return "[Empty response from Titan model]"

    except Exception as e:
        logging.error("Error calling Bedrock Titan model", exc_info=True)
        return f"[Error: {str(e)}]"
