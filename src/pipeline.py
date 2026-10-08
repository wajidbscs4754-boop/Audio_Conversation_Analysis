
from pathlib import Path
import os
import json
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found in .env"
    )


# =====================================================
# FILE PATHS
# =====================================================

audio_path = Path(
    "data/conversation.wav"
)

transcript_path = Path(
    "data/transcript.txt"
)

timestamp_path = Path(
    "data/transcript_timestamps.txt"
)

analysis_path = Path(
    "data/analysis.txt"
)

json_path = Path(
    "data/analysis.json"
)


# =====================================================
# CHECK AUDIO
# =====================================================

if not audio_path.exists():

    raise FileNotFoundError(
        f"Audio file not found: {audio_path}"
    )


# =====================================================
# HELPER FUNCTION
# =====================================================

def run_script(script_path, step_name):

    print("\n==============================")
    print(step_name)
    print("==============================")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:

        if result.stderr:
            print(result.stderr)

        raise RuntimeError(
            f"{script_path} failed."
        )

    return result


# =====================================================
# STEP 1: TRANSCRIPTION + SPEAKER DIARIZATION
# =====================================================

run_script(
    Path("src/combine_speakers.py"),
    "STEP 1: Transcription + Speaker Diarization"
)


# =====================================================
# CHECK TRANSCRIPT
# =====================================================

if not transcript_path.exists():

    raise RuntimeError(
        "transcript.txt was not created."
    )


transcript = transcript_path.read_text(
    encoding="utf-8"
).strip()


# =====================================================
# STEP 2: CONVERSATION TURNS
# =====================================================

run_script(
    Path("src/conversation_turns.py"),
    "STEP 2: Conversation Turns"
)


# =====================================================
# STEP 3: LATENCY ANALYSIS
# =====================================================

run_script(
    Path("src/latency_analysis.py"),
    "STEP 3: Latency Analysis"
)


# =====================================================
# STEP 4: INTERRUPTION ANALYSIS
# =====================================================

run_script(
    Path("src/interruption_analysis.py"),
    "STEP 4: Interruption Analysis"
)


# =====================================================
# STEP 5: OPENROUTER CLIENT
# =====================================================

print("\n==============================")
print("STEP 5: OpenRouter LLM Analysis")
print("==============================")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# =====================================================
# STEP 6: LLM ANALYSIS
# =====================================================

response = client.chat.completions.create(

    model="openrouter/free",

    messages=[

        {
            "role": "system",
            "content": (
                "You are an audio conversation "
                "analysis assistant. "
                "Analyze the conversation accurately. "
                "Do not invent information."
            )
        },

        {
            "role": "user",
            "content": f"""
Analyze the following conversation transcript.

Transcript:
{transcript}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "summary": "Short summary of the conversation",
    "sentiment": "Overall sentiment or tone",
    "main_topics": [],
    "key_points": [],
    "questions": [],
    "action_items": [],
    "decisions": []
}}

Rules:

- main_topics: list the main topics discussed.
- key_points: list important points.
- questions: list questions actually asked.
- action_items: list tasks that need to be completed.
- decisions: list important decisions made.
- If there is no information for a section,
  return an empty list [].
- Do not invent information.
- Do not add Markdown.
- Do not add ```json.
- Return only the JSON object.
"""
        }
    ]
)


# =====================================================
# STEP 7: CHECK OPENROUTER RESPONSE
# =====================================================

print("\nFULL OPENROUTER RESPONSE:")
print(response)


if getattr(response, "error", None):

    error = response.error

    message = error.get(
        "message",
        "Unknown OpenRouter error"
    )

    code = error.get(
        "code",
        "Unknown"
    )

    raise RuntimeError(
        f"OpenRouter API error: {message} "
        f"(code: {code})"
    )


if not response.choices:

    raise RuntimeError(
        "OpenRouter returned no choices."
    )


raw_analysis = (
    response.choices[0].message.content
)


if raw_analysis is None:

    raise RuntimeError(
        "OpenRouter returned no text content."
    )


if not raw_analysis.strip():

    raise ValueError(
        "The LLM returned an empty response."
    )


# =====================================================
# STEP 8: PARSE JSON
# =====================================================

try:

    analysis_data = json.loads(
        raw_analysis
    )

except json.JSONDecodeError:

    raise ValueError(
        "The LLM did not return valid JSON.\n\n"
        f"LLM response:\n{raw_analysis}"
    )


# =====================================================
# STEP 9: CREATE TEXT ANALYSIS
# =====================================================

analysis_text = f"""
Summary:
{analysis_data["summary"]}

Sentiment:
{analysis_data["sentiment"]}

Main Topics:
{chr(10).join("- " + topic for topic in analysis_data["main_topics"])}

Key Points:
{chr(10).join("- " + point for point in analysis_data["key_points"])}

Questions:
{chr(10).join("- " + question for question in analysis_data["questions"])}

Action Items:
{chr(10).join("- " + item for item in analysis_data["action_items"])}

Important Decisions:
{chr(10).join("- " + decision for decision in analysis_data["decisions"])}
""".strip()


# =====================================================
# STEP 10: SAVE TEXT ANALYSIS
# =====================================================

analysis_path.write_text(
    analysis_text,
    encoding="utf-8"
)

print(
    f"\nAnalysis saved to: "
    f"{analysis_path}"
)


# =====================================================
# STEP 11: SAVE JSON ANALYSIS
# =====================================================

json_path.write_text(
    json.dumps(
        analysis_data,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print(
    f"JSON analysis saved to: "
    f"{json_path}"
)


# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n==============================")
print("Audio Conversation Analysis")
print("==============================")

print("\nTranscript:")
print("------------------------------")
print(transcript)

print("\nAI Analysis:")
print("------------------------------")
print(analysis_text)

print("\nFiles created:")
print("------------------------------")
print(f"- {transcript_path}")
print(f"- {timestamp_path}")
print("- data/speaker_transcript.txt")
print("- data/conversation_turns.json")
print(f"- {analysis_path}")
print(f"- {json_path}")

print("\nPipeline completed successfully.")

