import os
import sys
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    print(f"Loaded .env from {env_path}")
except ImportError:
    print("python-dotenv not installed")

import anthropic

key = os.environ.get("ANTHROPIC_API_KEY")
if not key:
    print("ERROR: ANTHROPIC_API_KEY not found in environment")
    sys.exit(1)

print(f"API Key present: {key[:8]}...")

client = anthropic.Anthropic(api_key=key)

# Try a very standard, older model first to rule out auth/connection issues
model_to_test = "claude-3-opus-20240229"

print(f"Testing model: {model_to_test}...")

try:
    message = client.messages.create(
        model=model_to_test,
        max_tokens=10,
        messages=[
            {"role": "user", "content": "Hello, world"}
        ]
    )
    print("Success!")
    print(message.content[0].text)
except Exception as e:
    print(f"Failed: {e}")
