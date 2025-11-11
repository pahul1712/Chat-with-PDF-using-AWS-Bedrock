import boto3
import json
import base64
import os
from botocore.exceptions import ClientError

prompt_data = """
provide me an 4k hd image of a beach, also use a blue sky rainy season and cinematic display
"""
bedrock = boto3.client(service_name="bedrock-runtime",region_name = "us-west-2")

payload = {
    "prompt":prompt_data,
    "negative_prompt": "low quality, blurry, distorted, overexposed",  # optional
    "mode": "text-to-image",
    "seed": 42,
    "output_format": "png",
    "aspect_ratio": "1:1"
}
body = json.dumps(payload)
model_id = "stability.sd3-5-large-v1:0"

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
print(response_body)
image_encoded = response_body["images"][0]
image_bytes = base64.b64decode(image_encoded)

# Saving the image to output directory
output_dir = "output"
os.makedirs(output_dir,exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name,"wb") as f:
    f.write(image_bytes)
