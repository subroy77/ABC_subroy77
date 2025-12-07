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
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.95
    }
    
    body = json.dumps(payload)
    
    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = response['body'].read().decode('utf-8')
        print(f"[DEBUG] Raw response: {response_body}")
        
        if not response_body.strip():
            print("[DEBUG] Empty response body")
            return {}
        
        response_data = json.loads(response_body)
        response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not response_text.strip():
            print("[DEBUG] Empty response text")
            return {}
        
        print(f"[DEBUG] Response text: {response_text}")
        return json.loads(response_text)
    
    except json.JSONDecodeError as je:
        print(f"Error parsing JSON response: {je}")
        return {}
    except Exception as e:
        print(f"Error invoking Bedrock model: {e}")
        return {}