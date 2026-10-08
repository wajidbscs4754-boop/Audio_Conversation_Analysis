from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")


# OpenRouter client
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# Read transcript
transcript_path = Path("data/transcript.txt")
transcript = transcript_path.read_text(encoding="utf-8").strip()


# Send transcript to LLM
response = client.chat.completions.create(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an audio conversation analysis assistant. "
                "Analyze the provided transcript clearly and accurately."
            )
        },
        {
            "role": "user",
            "content": f"""
Analyze this transcript.

Transcript:
{transcript}

Provide:
1. Summary
2. Sentiment
3. Main topics
4. Key points
"""
        }
    ]
)


# Print result
analysis = response.choices[0].message.content

print("\nAI Conversation Analysis")
print("------------------------")
print(analysis)

output_path = Path("data/analysis.txt")

output_path.write_text(analysis, encoding="utf-8")

print(f"\nAnalysis saved to: {output_path}")