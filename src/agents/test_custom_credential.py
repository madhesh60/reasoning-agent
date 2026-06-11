import asyncio
import os
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")


from azure.ai.projects import AIProjectClient
from azure.core.credentials import AccessToken

class CustomApiKeyCredential:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_token(self, *scopes, **kwargs):
        # Expose the API key as an AccessToken expiring in 1 hour
        return AccessToken(self.api_key, int(time.time()) + 3600)

async def test_conn():
    endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    print(f"Endpoint: {endpoint}")
    print(f"API Key set: {bool(api_key)}")
    
    if not endpoint or not api_key:
        print("Missing endpoint or API key in environment.")
        return
        
    try:
        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=CustomApiKeyCredential(api_key),
        )
        
        openai_client = project_client.get_openai_client()
        
        print("Creating responses creation test...")
        response = openai_client.responses.create(
            input=[{"role": "user", "content": "Hello, what can you do? Reply with a short JSON."}],
            extra_body={"agent_reference": {"name": "planner-agent", "version": "9", "type": "agent_reference"}},
        )
        print("Success! Output:")
        print(response.output_text)
    except Exception as e:
        print("Failed:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_conn())
