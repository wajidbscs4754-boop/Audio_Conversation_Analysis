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


if not transcript:

    raise RuntimeError(
        "transcript.txt is empty."
    )


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
# LLM PROMPTS
# =====================================================

system_prompt = """
You are an audio conversation analysis assistant.

Analyze the conversation accurately.

Do not invent information.

IMPORTANT:
Return ONLY one valid JSON object.

Do not return:
- explanations
- Markdown
- ```json
- safety messages
- text before the JSON
- text after the JSON

The JSON must contain exactly these keys:

summary
sentiment
main_topics
key_points
questions
action_items
decisions
"""


user_prompt = f"""
Analyze the following conversation transcript.

Transcript:
{transcript}

Return JSON using exactly this structure:

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

- summary: Give a short accurate summary.
- sentiment: Describe the overall tone of the conversation.
- main_topics: List the main topics discussed.
- key_points: List important points.
- questions: List questions actually asked.
- action_items: List tasks that need to be completed.
- decisions: List important decisions made.
- If there is no information for a section, return [].
- Do not invent information.
- Return ONLY the JSON object.
"""


# =====================================================
# FUNCTION: CALL OPENROUTER
# =====================================================

def call_openrouter(
    system_message,
    user_message
):

    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        temperature=0
    )

    return response


# =====================================================
# STEP 6: FIRST LLM REQUEST
# =====================================================

print("\nSending conversation to OpenRouter...")


response = call_openrouter(
    system_prompt,
    user_prompt
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
    response.choices[0]
    .message
    .content
)


if raw_analysis is None:

    raise RuntimeError(
        "OpenRouter returned no text content."
    )


raw_analysis = raw_analysis.strip()


if not raw_analysis:

    raise RuntimeError(
        "The LLM returned an empty response."
    )


# =====================================================
# STEP 8: PARSE FIRST JSON RESPONSE
# =====================================================

try:

    analysis_data = json.loads(
        raw_analysis
    )

    print(
        "\nFirst LLM response was valid JSON."
    )

except json.JSONDecodeError:

    print(
        "\nFirst LLM response was not valid JSON."
    )

    print(
        "Retrying OpenRouter..."
    )


    # =================================================
    # RETRY PROMPT
    # =================================================

    retry_system_prompt = """
You are a JSON formatting assistant.

Return ONLY a valid JSON object.

No explanation.
No Markdown.
No ```json.
No safety message.
No text before the JSON.
No text after the JSON.

Use exactly these keys:

summary
sentiment
main_topics
key_points
questions
action_items
decisions
"""


    retry_user_prompt = f"""
Analyze this conversation and return ONLY valid JSON.

Conversation transcript:

{transcript}

Required structure:

{{
    "summary": "",
    "sentiment": "",
    "main_topics": [],
    "key_points": [],
    "questions": [],
    "action_items": [],
    "decisions": []
}}

Rules:

- Use only information contained in the transcript.
- Do not invent information.
- questions must contain only questions actually asked.
- main_topics must contain topics actually discussed.
- key_points must contain important points from the transcript.
- action_items should contain tasks only if they are actually mentioned.
- decisions should contain decisions only if they are actually mentioned.
- If a section has no information, return [].
- Return ONLY the JSON object.
"""


    # =============================================
    # RETRY REQUEST
    # =============================================

    retry_response = call_openrouter(
        retry_system_prompt,
        retry_user_prompt
    )


    # =============================================
    # CHECK RETRY RESPONSE
    # =============================================

    print("\nFULL RETRY RESPONSE:")
    print(retry_response)


    if getattr(
        retry_response,
        "error",
        None
    ):

        error = retry_response.error

        message = error.get(
            "message",
            "Unknown OpenRouter error"
        )

        code = error.get(
            "code",
            "Unknown"
        )

        raise RuntimeError(
            f"OpenRouter retry error: "
            f"{message} (code: {code})"
        )


    if not retry_response.choices:

        raise RuntimeError(
            "OpenRouter retry returned no choices."
        )


    retry_raw = (
        retry_response
        .choices[0]
        .message
        .content
    )


    if retry_raw is None:

        raise RuntimeError(
            "OpenRouter retry returned no text."
        )


    retry_raw = retry_raw.strip()


    if not retry_raw:

        raise RuntimeError(
            "OpenRouter retry returned empty text."
        )


    # =============================================
    # PARSE RETRY JSON
    # =============================================

    try:

        analysis_data = json.loads(
            retry_raw
        )

        print(
            "\nRetry response was valid JSON."
        )

    except json.JSONDecodeError:

        raise ValueError(
            "The LLM returned invalid JSON "
            "even after retry.\n\n"
            f"First response:\n"
            f"{raw_analysis}\n\n"
            f"Retry response:\n"
            f"{retry_raw}"
        )


# =====================================================
# STEP 9: VALIDATE JSON STRUCTURE
# =====================================================

required_fields = [

    "summary",

    "sentiment",

    "main_topics",

    "key_points",

    "questions",

    "action_items",

    "decisions"

]


for field in required_fields:

    if field not in analysis_data:

        raise ValueError(
            f"LLM JSON is missing required "
            f"field: {field}"
        )


# =====================================================
# VALIDATE LIST FIELDS
# =====================================================

list_fields = [

    "main_topics",

    "key_points",

    "questions",

    "action_items",

    "decisions"

]


for field in list_fields:

    if not isinstance(
        analysis_data[field],
        list
    ):

        print(
            f"Warning: {field} was not a list. "
            f"Converting to empty list."
        )

        analysis_data[field] = []


# =====================================================
# STEP 10: CREATE TEXT ANALYSIS
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
# STEP 11: SAVE TEXT ANALYSIS
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
# STEP 12: SAVE JSON ANALYSIS
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

print(
    f"- {transcript_path}"
)

print(
    f"- {timestamp_path}"
)

print(
    "- data/speaker_transcript.txt"
)

print(
    "- data/conversation_turns.json"
)

print(
    f"- {analysis_path}"
)

print(
    f"- {json_path}"
)


print("\nPipeline completed successfully.")