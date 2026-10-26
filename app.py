
import os
import re
import json
import tempfile

import streamlit as st
import whisper
from dotenv import load_dotenv
from openai import OpenAI

from src.wavlm_pipeline import PronunciationAssessorV2


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Audio Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENVIRONMENT
# ============================================================
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# MODEL LOADERS
# ============================================================
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


@st.cache_resource
def load_pronunciation_assessor():
    return PronunciationAssessorV2()


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9'\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def recognized_word_matches_target(
    spoken_text: str,
    target_word: str,
) -> bool:
    """Check whether recognized speech matches the target word."""
    target = normalize_text(target_word)
    spoken = normalize_text(spoken_text)

    return bool(target and spoken and spoken == target)



# ============================================================
# TRANSCRIPTION ACCURACY EVALUATION
# ============================================================
def calculate_word_errors(reference, hypothesis):
    """Return substitutions, deletions and insertions."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    previous = list(range(len(hyp_words) + 1))

    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]

        for j, hyp_word in enumerate(hyp_words, start=1):
            substitution = previous[j - 1] + (
                ref_word != hyp_word
            )
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1

            current.append(
                min(substitution, deletion, insertion)
            )

        previous = current

    return previous[-1], len(ref_words)


def evaluate_speechocean_transcription(limit=100):
    """Evaluate Whisper against SpeechOcean762 test transcripts."""
    dataset_dir = os.path.join(
        os.getcwd(), "evaluation", "speechocean762"
    )
    test_dir = os.path.join(dataset_dir, "test")

    text_path = os.path.join(test_dir, "text")
    wav_scp_path = os.path.join(test_dir, "wav.scp")

    if not os.path.isfile(text_path):
        raise FileNotFoundError(
            f"Reference transcript not found: {text_path}"
        )

    if not os.path.isfile(wav_scp_path):
        raise FileNotFoundError(
            f"Audio manifest not found: {wav_scp_path}"
        )

    references = {}

    with open(text_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                references[parts[0]] = parts[1]

    audio_paths = {}

    with open(wav_scp_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                audio_paths[parts[0]] = parts[1]

    # Keep only IDs that have both a transcript and audio.
    sample_ids = [
        audio_id
        for audio_id in references
        if audio_id in audio_paths
    ][:limit]

    if not sample_ids:
        raise ValueError(
            "No matching audio and reference transcripts found."
        )

    model = load_whisper_model()

    total_errors = 0
    total_reference_words = 0
    evaluated = 0
    skipped = 0

    for audio_id in sample_ids:
        relative_path = audio_paths[audio_id]

        audio_path = os.path.normpath(
            os.path.join(dataset_dir, relative_path)
        )

        if not os.path.isfile(audio_path):
            skipped += 1
            continue

        result = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            fp16=False,
        )

        hypothesis = str(result.get("text") or "").strip()
        reference = references[audio_id]

        errors, reference_word_count = calculate_word_errors(
            reference, hypothesis
        )

        total_errors += errors
        total_reference_words += reference_word_count
        evaluated += 1

    if evaluated == 0 or total_reference_words == 0:
        raise ValueError(
            "No valid samples were evaluated."
        )

    wer = total_errors / total_reference_words * 100

    return {
        "evaluated": evaluated,
        "skipped": skipped,
        "wer": round(wer, 2),
        "match_score": round(max(0, 100 - wer), 2),
    }



# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎙️ Audio Intelligence")
    st.caption("AI-powered speech analysis platform")
    st.divider()

    feature = st.radio(
        "WORKSPACE",
        [
            "🗣️ Pronunciation Practice",
            "💬 Conversation Analysis",
        ],
    )

    st.divider()
    st.caption("AI Speech Analysis")
    st.caption("Whisper • WavLM • OpenRouter")


# ============================================================
# MAIN HEADER
# ============================================================
st.title("🎙️ Audio Intelligence Platform")
st.caption("Speech analysis and pronunciation assessment")
st.divider()


# ============================================================
# PRONUNCIATION PRACTICE
# ============================================================
if feature == "🗣️ Pronunciation Practice":

    st.header("🗣️ Pronunciation Practice")

    st.write(
        "Students pronounce a target word. Whisper attempts to "
        "recognize the spoken word, while WavLM assesses "
        "pronunciation against the target."
    )

    st.divider()

    st.subheader("🎯 Target Word")

    target_word = st.text_input(
        "Enter the word the student should pronounce:",
        value="computer",
    ).strip()

    if target_word:
        st.info(f"🎯 Student should say: **{target_word}**")

    st.divider()

    st.subheader("🎙️ Student Recording")

    st.write(
        "Click the microphone button and say the target word."
    )

    pronunciation_audio = st.audio_input(
        "Record pronunciation"
    )

    if pronunciation_audio is not None:

        audio_bytes = pronunciation_audio.getvalue()

        if not audio_bytes:
            st.warning(
                "The recording is empty. Please record the word again."
            )

    if pronunciation_audio is not None:

        st.success("✓ Recording captured successfully!")

        st.audio(pronunciation_audio)

        st.divider()

        if st.button(
            "🔍 Analyze Pronunciation",
            type="primary",
            use_container_width=True,
        ):

            if not target_word:
                st.warning("Please enter a target word.")
                st.stop()

            pronunciation_path = None

            with st.spinner(
                "Analyzing speech and pronunciation..."
            ):

                try:
                    # Save browser recording temporarily.
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav",
                    ) as temp_file:

                        temp_file.write(
                            pronunciation_audio.getvalue()
                        )

                        pronunciation_path = temp_file.name

                    # ----------------------------------------
                    # WHISPER SPEECH RECOGNITION
                    # ----------------------------------------
                    whisper_model = load_whisper_model()

                    whisper_result = whisper_model.transcribe(
                        pronunciation_path,
                        language="en",
                        task="transcribe",
                        fp16=False,
                        temperature=0,
                        condition_on_previous_text=False,
                    )

                    spoken_text = str(
                        whisper_result.get("text") or ""
                    ).strip()

                    # ----------------------------------------
                    # WAVLM PRONUNCIATION ASSESSMENT
                    # ----------------------------------------
                    assessor = load_pronunciation_assessor()

                    assessment = assessor.assess(
                        pronunciation_path,
                        target_word,
                    )

                    if not isinstance(assessment, dict):
                        raise RuntimeError(
                            "The pronunciation model returned "
                            "an unexpected result."
                        )

                    if assessment.get("error"):
                        raise RuntimeError(
                            str(assessment["error"])
                        )

                    if assessment.get("overall_score") is None:
                        raise RuntimeError(
                            "The pronunciation model did not "
                            "return a score."
                        )

                    pronunciation_score = round(
                        float(assessment["overall_score"])
                    )

                    # Keep the displayed score within 0-100.
                    pronunciation_score = max(
                        0,
                        min(100, pronunciation_score),
                    )

                    # ----------------------------------------
                    # WORD RECOGNITION CHECK
                    # ----------------------------------------
                    recognition_match = (
                        recognized_word_matches_target(
                            spoken_text,
                            target_word,
                        )
                    )

                    # A good result requires both:
                    # 1. Recognized speech matches target.
                    # 2. Pronunciation score is at least 80%.
                    word_match = (
                        recognition_match
                        and pronunciation_score >= 80
                    )

                    # ----------------------------------------
                    # ASSESSMENT RESULTS
                    # ----------------------------------------
                    st.divider()
                    st.subheader("📊 Assessment Result")

                    if not spoken_text:

                        st.warning(
                            "⚠️ Speech Could Not Be Recognized"
                        )

                        st.caption(
                            "Please record again in a quiet place "
                            "and say the target word clearly. "
                            "The pronunciation score is shown "
                            "for reference only."
                        )

                    elif not recognition_match:

                        st.warning(
                            "⚠️ Word Recognition Mismatch"
                        )

                        st.write(
                            f"**Expected word:** {target_word}"
                        )

                        st.write(
                            f"**Recognized speech:** {spoken_text}"
                        )

                        st.caption(
                            "The recognized text does not match "
                            "the target, so this attempt cannot "
                            "be confirmed as a correct word. "
                            "Try recording again."
                        )

                    elif pronunciation_score >= 80:

                        st.success(
                            "✅ Good Pronunciation Score"
                        )

                        st.caption(
                            "The recognized word matches the "
                            "target and the pronunciation model "
                            "score is 80% or higher."
                        )

                    else:

                        st.warning(
                            "⚠️ More Practice Recommended"
                        )

                        st.caption(
                            "The target word was recognized, "
                            "but the pronunciation score is "
                            "below 80%. Try again at a "
                            "comfortable pace."
                        )

                    # ----------------------------------------
                    # RESULT CARDS
                    # ----------------------------------------
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Pronunciation Score",
                            f"{pronunciation_score}%",
                        )

                    with col2:
                        st.metric(
                            "Target Word",
                            target_word,
                        )

                    with col3:
                        st.metric(
                            "Recognized Speech",
                            (
                                spoken_text
                                if spoken_text
                                else "Not recognized"
                            ),
                        )

                    # ----------------------------------------
                    # PRACTICE FEEDBACK
                    # ----------------------------------------
                    st.divider()
                    st.subheader("💡 Practice Feedback")

                    if not spoken_text:

                        st.info(
                            "Please try again. Record in a quiet "
                            "place and leave a short pause before "
                            "and after saying the word."
                        )

                    elif not recognition_match:

                        st.info(
                            "The speech recognizer heard a "
                            "different word. This does not by "
                            "itself prove the pronunciation was "
                            "wrong. Try another clear recording."
                        )

                    elif pronunciation_score >= 80:

                        st.success(
                            "Good effort! The target was "
                            "recognized and the pronunciation "
                            "model gave a strong score. "
                            "Keep practicing."
                        )

                    else:

                        st.info(
                            "Good effort! Listen to the target "
                            "word and try again. Speak at a "
                            "comfortable pace and pronounce "
                            "each sound clearly."
                        )

                    # ----------------------------------------
                    # OPTIONAL ATTEMPT DETAILS
                    # ----------------------------------------
                    with st.expander(
                        "📋 View Attempt Details"
                    ):

                        st.write(
                            f"**Target word:** {target_word}"
                        )

                        st.write(
                            "**Recognized speech:** "
                            f"{spoken_text or 'No speech detected'}"
                        )

                        st.write(
                            "**WavLM pronunciation score:** "
                            f"{pronunciation_score}%"
                        )

                        if "n_phonemes" in assessment:
                            st.write(
                                "**Phonemes assessed:** "
                                f"{assessment['n_phonemes']}"
                            )

                        if "n_errors" in assessment:
                            st.write(
                                "**Potential phoneme errors:** "
                                f"{assessment['n_errors']}"
                            )

                        if "error_rate" in assessment:
                            st.write(
                                "**Model error rate:** "
                                f"{assessment['error_rate']}%"
                            )

                        st.caption(
                            "Whisper recognition and WavLM "
                            "pronunciation scoring are different "
                            "measurements. A pronunciation score "
                            "does not independently verify "
                            "which word was spoken."
                        )

                except Exception as e:

                    st.error(
                        f"Pronunciation analysis failed: {e}"
                    )

                    st.info(
                        "Check that src/wavlm_pipeline.py and "
                        "its model dependencies are available."
                    )

                finally:

                    if (
                        pronunciation_path
                        and os.path.exists(pronunciation_path)
                    ):

                        try:
                            os.remove(pronunciation_path)

                        except OSError:
                            pass


# ============================================================
# CONVERSATION ANALYSIS
# ============================================================
else:

    st.header("💬 Conversation Analysis")




    # ----------------------------------------
    # DATASET TRANSCRIPTION EVALUATION
    # ----------------------------------------
    st.subheader("📊 Transcription Accuracy Evaluation")

    st.caption(
        "Evaluate Whisper using SpeechOcean762 test audio "
        "and its reference transcripts. This evaluates the "
        "model on dataset samples, not the current upload."
    )

    if st.button(
        "🧪 Evaluate Transcription Model",
        use_container_width=True,
    ):
        with st.spinner(
            "Evaluating Whisper on dataset samples..."
        ):
            try:
                evaluation = evaluate_speechocean_transcription(
                    limit=100
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Transcript Match Score",
                    f"{evaluation['match_score']}%",
                )

                col2.metric(
                    "Word Error Rate (WER)",
                    f"{evaluation['wer']}%",
                )

                col3.metric(
                    "Samples Evaluated",
                    evaluation["evaluated"],
                )

                if evaluation["skipped"] > 0:
                    st.warning(
                        f"{evaluation['skipped']} audio files "
                        "were skipped because they were missing."
                    )

                st.caption(
                    "Match Score = max(0, 100 − WER). "
                    "This is a WER-derived score, not a "
                    "separate measure of overall accuracy."
                )

            except Exception as e:
                st.error(
                    f"Dataset evaluation failed: {e}"
                )

    st.write(
        "Upload or record a conversation and generate "
        "structured AI-powered insights from speech."
    )

    st.divider()

    st.subheader("🎧 Audio Input")

    option = st.radio(
        "Choose audio input method:",
        [
            "🎙️ Record Voice",
            "📁 Upload Audio",
        ],
        horizontal=True,
    )

    audio_data = None
    uploaded_filename = None

    if option == "🎙️ Record Voice":

        st.info(
            "Use the microphone below to record your conversation."
        )

        audio_data = st.audio_input(
            "Click the microphone button to record"
        )

    else:

        st.info("Upload a WAV, MP3, or M4A audio file.")

        audio_data = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a"],
        )

        if audio_data is not None:
            uploaded_filename = audio_data.name

    if audio_data is not None:

        st.success("✓ Audio ready for analysis!")

        st.audio(audio_data)

        st.divider()

        if st.button(
            "🔍 Analyze Conversation",
            type="primary",
            use_container_width=True,
        ):

            temp_audio_path = None

            audio_suffix = (
                os.path.splitext(uploaded_filename)[1].lower()
                if uploaded_filename
                else ".wav"
            )

            if audio_suffix not in [
                ".wav",
                ".mp3",
                ".m4a",
            ]:
                audio_suffix = ".wav"

            with st.spinner("Processing conversation..."):

                try:
                    # ----------------------------------------
                    # CHECK OPENROUTER API KEY
                    # ----------------------------------------
                    if not api_key:
                        raise RuntimeError(
                            "OPENROUTER_API_KEY not found in .env. "
                            "Add the key and restart Streamlit."
                        )

                    # ----------------------------------------
                    # SAVE AUDIO TEMPORARILY
                    # ----------------------------------------
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=audio_suffix,
                    ) as temp_file:

                        temp_file.write(
                            audio_data.getvalue()
                        )

                        temp_audio_path = temp_file.name

                    # ----------------------------------------
                    # WHISPER TRANSCRIPTION
                    # ----------------------------------------
                    st.write("🎙️ Transcribing speech...")

                    model = load_whisper_model()

                    result = model.transcribe(
                        temp_audio_path,
                        language="en",
                        task="transcribe",
                        fp16=False,
                    )

                    transcript = str(
                        result.get("text") or ""
                    ).strip()

                    if not transcript:
                        raise ValueError(
                            "No speech was detected in the audio."
                        )

                    # ----------------------------------------
                    # TIMESTAMPED TRANSCRIPT
                    # ----------------------------------------
                    timestamp_lines = []

                    for segment in result.get("segments", []):

                        start = segment["start"]
                        end = segment["end"]
                        segment_text = segment["text"].strip()

                        start_minutes = int(start // 60)
                        start_seconds = int(start % 60)

                        end_minutes = int(end // 60)
                        end_seconds = int(end % 60)

                        timestamp_lines.append(
                            f"[{start_minutes:02d}:"
                            f"{start_seconds:02d} - "
                            f"{end_minutes:02d}:"
                            f"{end_seconds:02d}] "
                            f"{segment_text}"
                        )

                    timestamped_transcript = "\n".join(
                        timestamp_lines
                    )

                    # ----------------------------------------
                    # OPENROUTER AI ANALYSIS
                    # ----------------------------------------
                    st.write("🤖 Generating AI insights...")

                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                    )

                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an audio conversation "
                                    "analysis assistant. Analyze "
                                    "the supplied transcript "
                                    "accurately. Do not invent "
                                    "information. Return only "
                                    "valid JSON."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"""
Analyze this conversation transcript.

Transcript:
{transcript}

Return ONLY valid JSON using exactly this structure:
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
- If there is no information, return [] for list fields.
- Do not invent information.
- Return only JSON without Markdown code fences.
""",
                            },
                        ],
                    )

                    # ----------------------------------------
                    # PARSE AI RESPONSE
                    # ----------------------------------------
                    if not response.choices:
                        raise ValueError(
                            "The AI provider returned no response choices."
                        )

                    raw_analysis = (
                        response.choices[0].message.content
                    )

                    if not raw_analysis:
                        raise ValueError(
                            "The AI returned an empty response."
                        )

                    raw_analysis = raw_analysis.strip()

                    # Remove Markdown code fences if present.
                    if raw_analysis.startswith("```"):

                        lines = raw_analysis.splitlines()

                        if (
                            lines
                            and lines[0].startswith("```")
                        ):
                            lines = lines[1:]

                        if (
                            lines
                            and lines[-1].strip() == "```"
                        ):
                            lines = lines[:-1]

                        raw_analysis = "\n".join(
                            lines
                        ).strip()

                    analysis_data = json.loads(raw_analysis)

                    if not isinstance(analysis_data, dict):
                        raise ValueError(
                            "The AI response JSON must be an object."
                        )

                    # ----------------------------------------
                    # DISPLAY RESULTS
                    # ----------------------------------------
                    st.success(
                        "Conversation analysis completed!"
                    )

                    st.divider()

                    st.subheader("📝 Transcript")
                    st.info(transcript)

                    with st.expander(
                        "⏱️ View Timestamped Transcript"
                    ):
                        st.code(
                            timestamped_transcript,
                            language=None,
                        )

                    st.subheader(
                        "🤖 AI Conversation Insights"
                    )

                    st.markdown("### 📋 Summary")

                    st.info(
                        analysis_data.get(
                            "summary",
                            "No summary available.",
                        )
                    )

                    sentiment = analysis_data.get(
                        "sentiment",
                        "Not available",
                    )

                    topics = analysis_data.get(
                        "main_topics",
                        [],
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Sentiment",
                            str(sentiment),
                        )

                    with col2:
                        st.metric(
                            "Main Topics",
                            len(topics),
                        )

                    if topics:

                        st.markdown("### 🏷️ Main Topics")

                        for topic in topics:
                            st.write(f"• {topic}")

                    key_points = analysis_data.get(
                        "key_points",
                        [],
                    )

                    if key_points:

                        st.markdown("### 🔑 Key Points")

                        for point in key_points:
                            st.write(f"• {point}")

                    questions = analysis_data.get(
                        "questions",
                        [],
                    )

                    if questions:

                        st.markdown("### ❓ Questions")

                        for question in questions:
                            st.write(f"• {question}")

                    actions = analysis_data.get(
                        "action_items",
                        [],
                    )

                    if actions:

                        st.markdown("### ✅ Action Items")

                        for action in actions:
                            st.write(f"• {action}")

                    decisions = analysis_data.get(
                        "decisions",
                        [],
                    )

                    if decisions:

                        st.markdown("### 📌 Decisions")

                        for decision in decisions:
                            st.write(f"• {decision}")

                except json.JSONDecodeError:

                    st.error(
                        "The AI returned invalid JSON. "
                        "Please try the analysis again."
                    )

                except Exception as e:

                    st.error(
                        f"Conversation analysis failed: {e}"
                    )

                finally:

                    if (
                        temp_audio_path
                        and os.path.exists(temp_audio_path)
                    ):

                        try:
                            os.remove(temp_audio_path)

                        except OSError:
                            pass
