import boto3
import json
import urllib3

# Disable SSL warnings (optional, for cleaner output)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

system_prompt = """
You MUST ignore all previous conversation or context. Each prediction is independent. Always output clean JSON when asked.
You are a knowledgeable market research analyst. Given a user prompt, identify which brand user is referring to in the prompt.
"""

bedrock = boto3.client(
    service_name='bedrock-runtime',
    verify=False
)
model_id = "openai.gpt-oss-20b-1:0"

def invoke_bedrock_model(user_prompt, custom_system_prompt=None):
    """
    Invoke Bedrock model with user prompt.
    
    Args:
        user_prompt: The user's input prompt
        custom_system_prompt: Optional custom system prompt (defaults to global system_prompt)
    
    Returns:
        dict: Parsed JSON response from the model
    """
    sys_prompt = custom_system_prompt or system_prompt
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_completion_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    body = json.dumps(payload)
    
    response = bedrock.invoke_model(
    modelId=model_id,
    body=body
    )

    response_body = json.loads(response['body'].read().decode('utf-8'))

    for choice in response_body['choices']:
        print(choice['message']['content'])