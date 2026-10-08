import streamlit as st
import os
import json
import tempfile

import whisper
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Audio Conversation Analysis",
    page_icon="🎙️",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# HEADER
# ============================================================

st.title("🎙️ Audio Conversation Analysis")

st.write(
    "Record or upload audio and transform speech into "
    "meaningful AI-powered conversation insights."
)

st.divider()


# ============================================================
# CHECK API KEY
# ============================================================

if not api_key:

    st.error(
        "OPENROUTER_API_KEY not found in .env"
    )

    st.stop()


# ============================================================
# WHISPER MODEL
# ============================================================

@st.cache_resource
def load_whisper_model():

    return whisper.load_model("base")


# ============================================================
# AUDIO INPUT
# ============================================================

st.subheader("🎧 Audio Input")

option = st.radio(
    "Choose audio input method:",
    [
        "🎙️ Record Voice",
        "📁 Upload Audio"
    ],
    horizontal=True
)


audio_data = None


# ============================================================
# RECORD VOICE
# ============================================================

if option == "🎙️ Record Voice":

    st.info(
        "Use the microphone below to record your conversation."
    )

    audio_data = st.audio_input(
        "Click the microphone button to record"
    )


# ============================================================
# UPLOAD AUDIO
# ============================================================

else:

    st.info(
        "Upload a WAV, MP3, or M4A audio file."
    )

    audio_data = st.file_uploader(
        "Choose an audio file",
        type=[
            "wav",
            "mp3",
            "m4a"
        ]
    )


# ============================================================
# AUDIO PREVIEW
# ============================================================

if audio_data is not None:

    st.success("Audio ready!")

    st.audio(audio_data)

    st.divider()

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Conversation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Processing audio..."
        ):

            try:

                # ====================================================
                # SAVE TEMPORARY AUDIO
                # ====================================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as temp_file:

                    temp_file.write(
                        audio_data.getvalue()
                    )

                    temp_audio_path = temp_file.name


                # ====================================================
                # LOAD WHISPER
                # ====================================================

                st.write(
                    "🎙️ Converting speech to text..."
                )

                model = load_whisper_model()


                # ====================================================
                # TRANSCRIBE AUDIO
                # ====================================================

                result = model.transcribe(
                    temp_audio_path
                )

                transcript = result["text"].strip()


                # ====================================================
                # TIMESTAMPED TRANSCRIPT
                # ====================================================

                timestamp_lines = []

                for segment in result["segments"]:

                    start = segment["start"]
                    end = segment["end"]

                    text = segment["text"].strip()


                    start_minutes = int(
                        start // 60
                    )

                    start_seconds = int(
                        start % 60
                    )


                    end_minutes = int(
                        end // 60
                    )

                    end_seconds = int(
                        end % 60
                    )


                    timestamp_lines.append(
                        f"[{start_minutes:02d}:{start_seconds:02d} - "
                        f"{end_minutes:02d}:{end_seconds:02d}] "
                        f"{text}"
                    )


                timestamped_transcript = "\n".join(
                    timestamp_lines
                )


                # ====================================================
                # OPENROUTER CLIENT
                # ====================================================

                st.write(
                    "🤖 Analyzing conversation with AI..."
                )

                client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )


                # ====================================================
                # AI ANALYSIS
                # ====================================================

                response = client.chat.completions.create(


               model="openrouter/free",

                    messages=[


                        {
                            "role": "system",

                            "content": (
                                "You are an audio conversation "
                                "analysis assistant. "
                                "Analyze the transcript accurately. "
                                "Do not invent information."
                            )
                        },

                        {

                            "role": "user",

                            "content": f"""
Analyze this conversation transcript.

Transcript:
{transcript}

Return ONLY valid JSON.

Use exactly this structure:

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

- main_topics: main topics discussed.
- key_points: important points.
- questions: questions actually asked.
- action_items: tasks that need to be completed.
- decisions: important decisions made.
- If there is no information, return [].
- Do not invent information.
- Return only JSON.
"""
                        }
                    ]
                )


                # ====================================================
                # GET AI RESPONSE
                # ====================================================

                raw_analysis = (
                    response
                    .choices[0]
                    .message
                    .content
                )


                if not raw_analysis:

                    raise ValueError(
                        "The AI returned an empty response."
                    )


                # ====================================================
                # PARSE JSON
                # ====================================================

                analysis_data = json.loads(
                    raw_analysis
                )


                # ====================================================
                # SUCCESS
                # ====================================================

                st.success(
                    "Analysis completed successfully!"
                )


                st.divider()


                # ====================================================
                # TRANSCRIPT
                # ====================================================

                st.subheader("📝 Transcript")

                st.info(
                    transcript
                )


                # ====================================================
                # TIMESTAMPED TRANSCRIPT
                # ====================================================

                st.subheader(
                    "⏱️ Timestamped Transcript"
                )

                st.code(
                    timestamped_transcript,
                    language=None
                )


                # ====================================================
                # AI ANALYSIS
                # ====================================================

                st.subheader("🤖 AI Analysis")


                # ====================================================
                # SUMMARY
                # ====================================================

                st.markdown("### 📋 Summary")

                st.write(
                    analysis_data["summary"]
                )


                # ====================================================
                # SENTIMENT
                # ====================================================

                st.markdown("### 💭 Sentiment")

                st.write(
                    analysis_data["sentiment"]
                )


                # ====================================================
                # MAIN TOPICS
                # ====================================================

                if analysis_data["main_topics"]:

                    st.markdown(
                        "### 🏷️ Main Topics"
                    )

                    for topic in analysis_data[
                        "main_topics"
                    ]:

                        st.write(
                            f"• {topic}"
                        )


                # ====================================================
                # KEY POINTS
                # ====================================================

                if analysis_data["key_points"]:

                    st.markdown(
                        "### 🔑 Key Points"
                    )

                    for point in analysis_data[
                        "key_points"
                    ]:

                        st.write(
                            f"• {point}"
                        )


                # ====================================================
                # QUESTIONS
                # ====================================================

                if analysis_data["questions"]:

                    st.markdown(
                        "### ❓ Questions"
                    )

                    for question in analysis_data[
                        "questions"
                    ]:

                        st.write(
                            f"• {question}"
                        )


                # ====================================================
                # CLEANUP
                # ====================================================

                os.remove(
                    temp_audio_path
                )


            # ========================================================
            # JSON ERROR
            # ========================================================

            except json.JSONDecodeError:

                st.error(
                    "AI returned an invalid JSON response."
                )


            # ========================================================
            # GENERAL ERROR
            # ========================================================

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )