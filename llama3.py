import boto3
import json
from botocore.exceptions import ClientError

prompt_data = """
Act as a Shakespeare and write a poem on machine learning
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt":"[INST]" + prompt_data + "[/INST]",
    "max_gen_len":512,
    "top_p":0.9
}
body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0"


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
response_text = response_body["generation"]
print(response_text)
