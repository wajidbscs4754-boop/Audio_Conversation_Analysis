
from pathlib import Path
import os
import json

from dotenv import load_dotenv
from openai import OpenAI


TURNS_PATH = Path("data/conversation_turns.json")
OUTPUT_PATH = Path("data/conversation_understanding.json")


load_dotenv()


api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")


if not TURNS_PATH.exists():
    raise FileNotFoundError(
        f"Conversation turns file not found: {TURNS_PATH}"
    )


with open(TURNS_PATH, "r", encoding="utf-8") as file:
    turns = json.load(file)


client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


def clean_json_text(text):
    """
    Remove Markdown code fences if the model returns JSON inside ```json ... ```.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def normalize_evaluation(evaluation):
    """
    Make the LLM output consistent.

    The model may return:
    relevance = "high" OR 1.0
    understanding_score = 90 OR 0.9
    """

    # -----------------------------
    # Normalize relevance
    # -----------------------------
    relevance = evaluation.get("relevance", "unknown")

    if isinstance(relevance, (int, float)):
        if relevance >= 0.75:
            relevance = "high"
        elif relevance >= 0.40:
            relevance = "medium"
        else:
            relevance = "low"

    # -----------------------------
    # Normalize understanding score
    # -----------------------------
    score = evaluation.get("understanding_score", 0)

    if isinstance(score, (int, float)):
        # Model sometimes returns 0.9 instead of 90
        if 0 <= score <= 1:
            score = score * 100

        score = round(score, 2)

    # -----------------------------
    # Normalize boolean fields
    # -----------------------------
    answers_question = bool(
        evaluation.get("answers_question", False)
    )

    topic_continuity = bool(
        evaluation.get("topic_continuity", False)
    )

    off_topic = bool(
        evaluation.get("off_topic", False)
    )

    reason = str(
        evaluation.get(
            "reason",
            "No explanation provided."
        )
    )

    return {
        "relevance": relevance,
        "answers_question": answers_question,
        "topic_continuity": topic_continuity,
        "understanding_score": score,
        "off_topic": off_topic,
        "reason": reason
    }


def analyze_response(previous_turn, current_turn):

    prompt = f"""
You are analyzing a conversation between two people.

Your task is to determine how well the current speaker
understood and responded to the previous speaker.

Previous speaker:
{previous_turn["speaker"]}

Previous message:
{previous_turn["text"]}

Current speaker:
{current_turn["speaker"]}

Current response:
{current_turn["text"]}

Evaluate the response using these criteria:

1. relevance:
   Is the response related to what the previous speaker said?

2. answers_question:
   If the previous speaker asked a question, does the
   current response address that question?
   If there was no question, use true if the response
   appropriately responds to the previous statement.

3. topic_continuity:
   Does the conversation remain on the same topic?

4. understanding_score:
   Give a score from 0 to 100.
   This represents how well the current response
   demonstrates understanding of the previous speaker.

5. off_topic:
   Is the response clearly unrelated to the previous turn?

6. reason:
   Briefly explain why you gave this evaluation.

Important:

- Casual conversation is valid.
- Do not require every response to answer a question.
- A natural response to a statement can demonstrate understanding.
- Do not judge grammar.
- Do not invent information.
- Evaluate only the relationship between these two turns.

Return ONLY valid JSON.

Use exactly these fields:

{{
    "relevance": "high",
    "answers_question": true,
    "topic_continuity": true,
    "understanding_score": 90,
    "off_topic": false,
    "reason": "The response directly relates to the previous speaker's point."
}}
"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conversation understanding "
                    "and response relevance evaluator. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # --------------------------------
    # Check response
    # --------------------------------
    if not response.choices:
        raise RuntimeError(
            "OpenRouter returned no choices.\n"
            f"Response object:\n{response}"
        )

    raw_output = response.choices[0].message.content

    if not raw_output:
        raise RuntimeError(
            "OpenRouter returned empty content.\n"
            f"Response object:\n{response}"
        )

    # --------------------------------
    # Clean JSON
    # --------------------------------
    raw_output = clean_json_text(raw_output)

    # --------------------------------
    # Parse JSON
    # --------------------------------
    try:
        evaluation = json.loads(raw_output)

    except json.JSONDecodeError:
        raise ValueError(
            "LLM did not return valid JSON.\n\n"
            f"Response:\n{raw_output}"
        )

    # --------------------------------
    # Normalize result
    # --------------------------------
    return normalize_evaluation(evaluation)


print("\n==============================")
print("Conversation Understanding")
print("==============================\n")


results = []


for i in range(1, len(turns)):

    previous_turn = turns[i - 1]
    current_turn = turns[i]

    print(
        f"Analyzing Turn {i}: "
        f"{previous_turn['speaker']} -> "
        f"{current_turn['speaker']}"
    )

    evaluation = analyze_response(
        previous_turn,
        current_turn
    )

    result = {
        "turn_number": i,
        "from_speaker": previous_turn["speaker"],
        "to_speaker": current_turn["speaker"],
        "previous_text": previous_turn["text"],
        "response_text": current_turn["text"],
        "evaluation": evaluation
    }

    results.append(result)


# --------------------------------
# Overall understanding score
# --------------------------------

scores = [
    item["evaluation"]["understanding_score"]
    for item in results
    if isinstance(
        item["evaluation"]["understanding_score"],
        (int, float)
    )
]


if scores:
    overall_score = round(
        sum(scores) / len(scores),
        2
    )
else:
    overall_score = 0


final_output = {
    "overall_understanding_score": overall_score,
    "total_responses_analyzed": len(results),
    "responses": results
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_output,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n------------------------------")
print(
    f"Overall Understanding: "
    f"{overall_score}%"
)
print("------------------------------")


for item in results:

    evaluation = item["evaluation"]

    print(
        f"\n{item['from_speaker']} -> "
        f"{item['to_speaker']}"
    )

    print(
        f"Response: "
        f"{item['response_text']}"
    )

    print(
        f"Relevance: "
        f"{evaluation['relevance']}"
    )

    print(
        f"Answers Question: "
        f"{evaluation['answers_question']}"
    )

    print(
        f"Topic Continuity: "
        f"{evaluation['topic_continuity']}"
    )

    print(
        f"Understanding Score: "
        f"{evaluation['understanding_score']}%"
    )

    print(
        f"Off Topic: "
        f"{evaluation['off_topic']}"
    )

    print(
        f"Reason: "
        f"{evaluation['reason']}"
    )


print(
    f"\nResults saved to: "
    f"{OUTPUT_PATH}"
)

print("\nConversation understanding completed.")

