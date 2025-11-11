import boto3
import json
from botocore.exceptions import ClientError

prompt_data = """
Act as a Shakespeare and write a poem on machine learning
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens":512,
    "temperature":0.5,
    "top_p":0.9,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_data}],
        }
    ],
}
body = json.dumps(payload)
model_id = "anthropic.claude-3-haiku-20240307-v1:0"


try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json")
    
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)


response_body = json.loads(response.get("body").read())
response_text = response_body.get("content")[0].get("text")
print(response_text)
