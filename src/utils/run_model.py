"""
Azure AI Foundry — Quick Model Connection Test
=============================================
Tests both gpt-4o-mini and gpt-4o deployments
using the API key from .env (no Azure CLI login required).

Run from project root:
    python -m src.utils.run_model
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Ensure project root is on sys.path when run directly
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

# ── Connection details pulled from .env ─────────────────────────────────────
ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]  # e.g. https://<res>.services.ai.azure.com/openai/v1
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOY = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Normalise: strip /chat/completions if accidentally left in
if ENDPOINT.endswith("/chat/completions"):
    ENDPOINT = ENDPOINT[: -len("/chat/completions")]

print("=" * 64)
print("Azure AI Foundry — Model Connection Test")
print("=" * 64)
print(f"  Endpoint   : {ENDPOINT}")
print(f"  Deployment : {DEPLOY}")
print()

client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)

TEST_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a concise reasoning assistant. " "Show your reasoning steps before answering."
        ),
    },
    {
        "role": "user",
        "content": "What are 3 main investment risks in the Indian EV market? "
        "Reason step-by-step then give a short final answer.",
    },
]

print("Sending test prompt...")
print("-" * 64)

completion = client.chat.completions.create(
    model=DEPLOY,
    messages=TEST_MESSAGES,
    max_tokens=800,
)

msg = completion.choices[0].message
print(f"Role    : {msg.role}")
print()
print("Response:")
print(msg.content)
print()
print("=" * 64)
print(f"Finish reason : {completion.choices[0].finish_reason}")
print(f"Total tokens  : {completion.usage.total_tokens if completion.usage else 'N/A'}")
print("=" * 64)
