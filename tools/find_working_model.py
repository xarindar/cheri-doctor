import os
import sys
import anthropic
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Candidates based on 2026 release
candidates = [
    "claude-4-6-sonnet-latest",
    "claude-4-6-sonnet-20260217",
    "claude-sonnet-4-6",
    "claude-4-opus-latest",
    "claude-4-sonnet-latest",
    "claude-4-haiku-latest",
]

print("Testing model IDs...")

for model in candidates:
    print(f"Trying: {model}", end=" ... ")
    try:
        client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print("SUCCESS! ✅")
        break
    except anthropic.BadRequestError as e:
        print(f"FAILED (Bad Request): {e.message}")
    except anthropic.NotFoundError as e:
        print(f"FAILED (404 Not Found)")
    except Exception as e:
        print(f"FAILED ({type(e).__name__}): {e}")
