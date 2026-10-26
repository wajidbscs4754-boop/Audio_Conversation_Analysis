
"""
Voice Correction Pipeline v2.1
WavLM phoneme-level English pronunciation assessment.

Keeps the PronunciationAssessorV2 API compatible with app.py.
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F_audio
from g2p_en import G2p
from huggingface_hub import hf_hub_download
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    WavLMModel,
)

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
DEFAULT_PHERR_THRESHOLD = 0.70

HF_REPO_ID = "Jianshu001/wavlm-phoneme-scorer"
HF_CHECKPOINT_FILENAME = "wavlm_finetuned.pt"

# Candidate IPA representations for ARPABET phones.
# The CTC vocabulary is checked at runtime; the first matching
# representation is used.
ARPABET_TO_IPA = {
    "aa": ["ɑː", "ɑ", "ɒ", "a"],
    "ae": ["æ"],
    "ah": ["ʌ", "ə", "ɐ"],
    "ao": ["ɔː", "ɔ", "ɒ"],
    "aw": ["aʊ"],
    "ax": ["ə", "ɐ", "ʌ"],
    "ay": ["aɪ"],
    "b": ["b"],
    "ch": ["tʃ", "t͡ʃ"],
    "d": ["d"],
    "dh": ["ð"],
    "eh": ["ɛ", "e"],
    "er": ["ɜː", "ɝ", "ɚ", "ɜ"],
    "ey": ["eɪ"],
    "f": ["f"],
    "g": ["ɡ", "g"],
    "hh": ["h"],
    "ih": ["ɪ", "ᵻ"],
    "iy": ["iː", "i"],
    "ir": ["ɪɹ"],
    "jh": ["dʒ", "d͡ʒ"],
    "k": ["k"],
    "l": ["l"],
    "m": ["m"],
    "n": ["n"],
    "ng": ["ŋ"],
    "ow": ["oʊ", "o", "əʊ"],
    "oy": ["ɔɪ"],
    "p": ["p"],
    "r": ["ɹ", "r"],
    "s": ["s"],
    "sh": ["ʃ"],
    "t": ["t"],
    "th": ["θ"],
    "uh": ["ʊ"],
    "uw": ["uː", "u"],
    "ur": ["ʊɹ"],
    "v": ["v"],
    "w": ["w"],
    "y": ["j"],
    "z": ["z"],
    "zh": ["ʒ"],
    "ar": ["ɑːɹ", "ɑɹ"],
    "oo": ["ʊ", "uː", "u"],
    "dr": ["dɹ"],
    "tr": ["tɹ"],
    "ts": ["ts"],
    "dz": ["dz"],
}

ALL_PHONES = sorted(ARPABET_TO_IPA)
PHONE_TO_ID = {phone: i for i, phone in enumerate(ALL_PHONES)}
N_PHONE_TYPES = len(ALL_PHONES)


class PhoneScorerHead(nn.Module):
    """Must match the architecture used to train the checkpoint."""

    def __init__(
        self,
        hidden_dim=1024,
        n_phone_types=N_PHONE_TYPES,
        phone_emb_dim=32,
        mlp_dim=512,
    ):
        super().__init__()
        self.phone_emb = nn.Embedding(n_phone_types, phone_emb_dim)
        input_dim = hidden_dim + phone_emb_dim + 2

        self.shared = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.score_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.pherr_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, h, phone_id, gop, n_frames):
        emb = self.phone_emb(phone_id)
        x = torch.cat(
            [h, emb, gop.unsqueeze(-1), n_frames.unsqueeze(-1)], dim=-1
        )
        shared = self.shared(x)
        score = self.score_head(shared).squeeze(-1)
        pherr_logit = self.pherr_head(shared).squeeze(-1)
        return score, pherr_logit


class PronunciationAssessorV2:
    """Phoneme-level pronunciation assessment using fine-tuned WavLM."""

    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        pherr_threshold=DEFAULT_PHERR_THRESHOLD,
    ):
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.pherr_threshold = pherr_threshold
        self._checkpoint_path = checkpoint_path or self._resolve_checkpoint()

        self._backbone = None
        self._scorer = None
        self._fe_backbone = None
        self._ctc_model = None
        self._fe_ctc = None
        self._vocab = None
        self._blank_idx = None
        self._g2p = None

    @staticmethod
    def _resolve_checkpoint():
        local_path = Path(__file__).parent / HF_CHECKPOINT_FILENAME
        if local_path.exists():
            return str(local_path)

        print(
            f"Downloading model from huggingface.co/{HF_REPO_ID}...",
            file=sys.stderr,
        )
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_CHECKPOINT_FILENAME,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id=None,
        device=None,
        pherr_threshold=DEFAULT_PHERR_THRESHOLD,
    ):
        repo = repo_id or HF_REPO_ID
        checkpoint = hf_hub_download(
            repo_id=repo,
            filename=HF_CHECKPOINT_FILENAME,
        )
        return cls(
            checkpoint_path=checkpoint,
            device=device,
            pherr_threshold=pherr_threshold,
        )

    def _load_models(self):
        if self._backbone is not None:
            return

        print("Loading WavLM backbone and scoring head...", file=sys.stderr)

        ckpt = torch.load(
            self._checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = ckpt["model_state"]

        backbone_state = {}
        head_state = {}

        for key, value in state_dict.items():
            if key.startswith("backbone."):
                backbone_state[key[len("backbone."):]] = value
            else:
                head_state[key] = value

        self._backbone = WavLMModel.from_pretrained(
            "microsoft/wavlm-large",
            output_hidden_states=False,
            mask_time_prob=0.0,
        )

        # Do not silently accept a badly mismatched checkpoint.
        missing, unexpected = self._backbone.load_state_dict(
            backbone_state, strict=False
        )
        if missing:
            print(
                f"Warning: {len(missing)} WavLM checkpoint keys missing.",
                file=sys.stderr,
            )
        if unexpected:
            print(
                f"Warning: {len(unexpected)} unexpected WavLM keys.",
                file=sys.stderr,
            )

        self._backbone.to(self.device).eval()
        self._fe_backbone = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-large"
        )

        self._scorer = PhoneScorerHead()
        self._scorer.load_state_dict(head_state)
        self._scorer.to(self.device).eval()

        print("Loading CTC alignment model...", file=sys.stderr)
        ctc_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        self._ctc_model = Wav2Vec2ForCTC.from_pretrained(ctc_name)
        self._ctc_model.to(self.device).eval()

        self._fe_ctc = Wav2Vec2FeatureExtractor.from_pretrained(ctc_name)

        vocab_path = hf_hub_download(ctc_name, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as file:
            self._vocab = json.load(file)

        self._blank_idx = self._vocab.get("<pad>", 0)
        self._g2p = G2p()

        print(f"Models loaded (device={self.device})", file=sys.stderr)

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------
    @staticmethod
    def load_audio(audio_path):
        waveform, sr = torchaudio.load(str(audio_path))

        if waveform.numel() == 0:
            raise ValueError("The audio file is empty.")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != SAMPLE_RATE:
            waveform = F_audio.resample(
                waveform, orig_freq=sr, new_freq=SAMPLE_RATE
            )

        waveform = waveform.float()

        if not torch.isfinite(waveform).all():
            raise ValueError("Audio contains invalid numeric values.")

        if waveform.shape[-1] < int(0.10 * SAMPLE_RATE):
            raise ValueError("Recording is too short to assess.")

        return waveform

    # --------------------------------------------------------
    # G2P and vocabulary mapping
    # --------------------------------------------------------
    def _text_to_phonemes(self, text):
        words = re.sub(r"[^\w' ]", " ", text).split()
        result = []

        for word_idx, word in enumerate(words):
            for raw_phone in self._g2p(word):
                if raw_phone == " ":
                    continue

                # G2p normally returns ARPABET, e.g. AH0, K, M.
                clean = re.sub(r"\d", "", raw_phone).lower()
                if clean in ARPABET_TO_IPA:
                    result.append({
                        "phone": clean,
                        "word": word,
                        "word_idx": word_idx,
                    })

        return result

    def _arpabet_to_model_idx(self, phone):
        candidates = ARPABET_TO_IPA.get(phone, [])

        for ipa in candidates:
            if ipa in self._vocab:
                return self._vocab[ipa]

        # Some vocabularies may directly include ARPABET tokens.
        if phone in self._vocab:
            return self._vocab[phone]

        return -1

    # --------------------------------------------------------
    # CTC Viterbi forced alignment
    # --------------------------------------------------------
    @staticmethod
    def _viterbi_align(emissions, phone_indices, blank_idx):
        """
        Return (frame_index, state_index) pairs.

        The state index identifies the position in the expanded CTC
        sequence. Odd state indices correspond to expected phones;
        even state indices correspond to blanks.
        """
        T, _ = emissions.shape
        if not phone_indices or T == 0:
            return []

        extended = [blank_idx]
        for phone_id in phone_indices:
            extended.extend([phone_id, blank_idx])

        n_states = len(extended)
        if T < len(phone_indices):
            return []

        neg_inf = float("-inf")
        dp = np.full((T, n_states), neg_inf, dtype=np.float64)
        back = np.full((T, n_states), -1, dtype=np.int32)

        dp[0, 0] = emissions[0, extended[0]].item()
        if n_states > 1:
            dp[0, 1] = emissions[0, extended[1]].item()

        for t in range(1, T):
            for state in range(n_states):
                best_score = dp[t - 1, state]
                best_state = state

                if state > 0 and dp[t - 1, state - 1] > best_score:
                    best_score = dp[t - 1, state - 1]
                    best_state = state - 1

                # CTC skip transition is allowed only for nonblank
                # labels that differ from the label two states back.
                if (
                    state > 1
                    and extended[state] != blank_idx
                    and extended[state] != extended[state - 2]
                    and dp[t - 1, state - 2] > best_score
                ):
                    best_score = dp[t - 1, state - 2]
                    best_state = state - 2

                dp[t, state] = (
                    best_score + emissions[t, extended[state]].item()
                )
                back[t, state] = best_state

        last_blank = n_states - 1
        last_phone = n_states - 2

        state = (
            last_blank
            if dp[T - 1, last_blank] >= dp[T - 1, last_phone]
            else last_phone
        )

        if not np.isfinite(dp[T - 1, state]):
            return []

        path = []
        for t in range(T - 1, -1, -1):
            path.append((t, state))
            if t > 0:
                state = int(back[t, state])
                if state < 0:
                    return []

        path.reverse()
        return path

    # --------------------------------------------------------
    # Align audio and score each expected phoneme
    # --------------------------------------------------------
    @torch.no_grad()
    def _score_phonemes(self, waveform, phone_indices):
        wav_np = waveform.squeeze(0).cpu().numpy()

        ctc_inputs = self._fe_ctc(
            wav_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        ctc_logits = self._ctc_model(
            ctc_inputs.input_values.to(self.device)
        ).logits

        emissions = torch.log_softmax(
            ctc_logits, dim=-1
        ).squeeze(0).cpu()

        path = self._viterbi_align(
            emissions, phone_indices, self._blank_idx
        )
        if not path:
            return None

        # One segment for each expected phone, indexed by its exact
        # CTC state. Do not group merely by token ID: repeated phones
        # can otherwise be merged or assigned to the wrong phone.
        segments = [[] for _ in phone_indices]

        for frame_idx, state_idx in path:
            if state_idx % 2 == 1:
                phone_idx = (state_idx - 1) // 2
                if 0 <= phone_idx < len(segments):
                    segments[phone_idx].append(frame_idx)

        bb_inputs = self._fe_backbone(
            wav_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        hidden = self._backbone(
            bb_inputs.input_values.to(self.device)
        ).last_hidden_state.squeeze(0)

        hidden_len = hidden.shape[0]
        ctc_len = emissions.shape[0]
        scale = hidden_len / max(ctc_len, 1)

        results = []

        for phone_idx, expected_id in enumerate(phone_indices):
            frames = segments[phone_idx]

            if not frames:
                results.append({
                    "gop": -20.0,
                    "score": 0.0,
                    "pherr_prob": 1.0,
                    "aligned": False,
                })
                continue

            h_start = max(0, int(min(frames) * scale))
            h_end = min(
                hidden_len, max(h_start + 1, int((max(frames) + 1) * scale))
            )

            if h_start >= hidden_len or h_end <= h_start:
                results.append({
                    "gop": -20.0,
                    "score": 0.0,
                    "pherr_prob": 1.0,
                    "aligned": False,
                })
                continue

            h_pooled = hidden[h_start:h_end].mean(dim=0)

            frame_emissions = emissions[frames]
            target_lp = frame_emissions[:, expected_id].mean().item()

            nonblank_mask = torch.ones(
                emissions.shape[1], dtype=torch.bool
            )
            nonblank_mask[self._blank_idx] = False
            nonblank_mask[expected_id] = False

            if nonblank_mask.any():
                best_other = frame_emissions[:, nonblank_mask].max(
                    dim=-1
                ).values.mean().item()
            else:
                best_other = target_lp

            gop = float(target_lp - best_other)

            results.append({
                "h": h_pooled,
                "gop": gop,
                "n_frames": len(frames),
                "aligned": True,
            })

        valid_indices = [
            i for i, item in enumerate(results) if "h" in item
        ]

        if not valid_indices:
            return results, [], None, None, None

        h_batch = torch.stack(
            [results[i]["h"] for i in valid_indices]
        ).to(self.device)

        gop_batch = torch.tensor(
            [results[i]["gop"] for i in valid_indices],
            dtype=torch.float32,
            device=self.device,
        )

        nf_batch = torch.tensor(
            [results[i]["n_frames"] for i in valid_indices],
            dtype=torch.float32,
            device=self.device,
        )

        return results, valid_indices, h_batch, gop_batch, nf_batch

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def assess(self, audio_path, text):
        self._load_models()

        phone_info = self._text_to_phonemes(text)
        if not phone_info:
            return {
                "text": text,
                "overall_score": 0,
                "words": [],
                "error": "No phonemes extracted",
            }

        indices = []
        valid_info = []
        unmapped = []

        for info in phone_info:
            idx = self._arpabet_to_model_idx(info["phone"])
            if idx >= 0:
                indices.append(idx)
                valid_info.append(info)
            else:
                unmapped.append(info["phone"])

        if not indices:
            return {
                "text": text,
                "overall_score": 0,
                "words": [],
                "error": "No phonemes mapped to the CTC vocabulary",
                "unmapped_phonemes": unmapped,
            }

        waveform = self.load_audio(audio_path)
        result = self._score_phonemes(waveform, indices)

        if result is None:
            return {
                "text": text,
                "overall_score": 0,
                "words": [],
                "error": "Alignment failed",
            }

        raw_results, valid_idx, h_batch, gop_batch, nf_batch = result

        if h_batch is not None:
            phone_ids = [
                PHONE_TO_ID.get(valid_info[i]["phone"], 0)
                for i in valid_idx
            ]

            pid_batch = torch.tensor(
                phone_ids, dtype=torch.long, device=self.device
            )

            pred_score, pred_pherr_logit = self._scorer(
                h_batch, pid_batch, gop_batch, nf_batch
            )

            pred_score = pred_score.detach().cpu().numpy()
            pred_pherr = torch.sigmoid(
                pred_pherr_logit
            ).detach().cpu().numpy()

            for j, i in enumerate(valid_idx):
                raw_results[i]["score"] = float(
                    np.clip(pred_score[j], 0, 100)
                )
                raw_results[i]["pherr_prob"] = float(pred_pherr[j])

        phoneme_results = []

        for info, raw in zip(valid_info, raw_results):
            score = raw.get("score", 0.0)
            pherr_prob = raw.get("pherr_prob", 1.0)

            phoneme_results.append({
                "phone": info["phone"],
                "word": info["word"],
                "word_idx": info["word_idx"],
                "score": round(float(score), 1),
                "gop": round(float(raw["gop"]), 3),
                "pherr_prob": round(float(pherr_prob), 3),
                "error": pherr_prob >= self.pherr_threshold,
                "aligned": bool(raw.get("aligned", False)),
            })

        words_dict = {}

        for phone_result in phoneme_results:
            word_idx = phone_result["word_idx"]

            if word_idx not in words_dict:
                words_dict[word_idx] = {
                    "word": phone_result["word"],
                    "phonemes": [],
                }

            words_dict[word_idx]["phonemes"].append({
                "phone": phone_result["phone"],
                "score": phone_result["score"],
                "gop": phone_result["gop"],
                "pherr_prob": phone_result["pherr_prob"],
                "error": phone_result["error"],
                "aligned": phone_result["aligned"],
            })

        words_list = []

        for word_idx in sorted(words_dict):
            word_data = words_dict[word_idx]
            phones = word_data["phonemes"]
            scores = [p["score"] for p in phones]
            n_errors = sum(p["error"] for p in phones)

            words_list.append({
                "word": word_data["word"],
                "score": round(float(np.mean(scores)), 1),
                "n_errors": int(n_errors),
                "n_phonemes": len(phones),
                "has_error": bool(n_errors > 0),
                "phonemes": phones,
            })

        all_scores = [p["score"] for p in phoneme_results]
        total_errors = sum(p["error"] for p in phoneme_results)

        return {
            "text": text,
            "overall_score": round(float(np.mean(all_scores)), 1),
            "n_phonemes": len(phoneme_results),
            "n_errors": int(total_errors),
            "error_rate": round(
                total_errors / max(len(phoneme_results), 1) * 100, 1
            ),
            "words": words_list,
            "unmapped_phonemes": unmapped,
        }

    def assess_batch(self, items, show_progress=True):
        self._load_models()
        results = []

        for i, (audio_path, text) in enumerate(items):
            try:
                results.append(self.assess(audio_path, text))
            except Exception as exc:
                results.append({
                    "text": text,
                    "overall_score": 0,
                    "words": [],
                    "error": str(exc),
                })

            if show_progress and (i + 1) % 50 == 0:
                print(
                    f"Processed {i + 1}/{len(items)}...",
                    file=sys.stderr,
                )

        return results


def print_result(result):
    print("\n" + "=" * 60)
    print(f'Text: "{result.get("text", "")}"')
    print(
        f'Overall Score: {result.get("overall_score", 0):.1f}/100 '
        f'(errors: {result.get("n_errors", "?")}/'
        f'{result.get("n_phonemes", "?")})'
    )
    print("=" * 60)

    if result.get("error"):
        print("Assessment error:", result["error"])

    for word_data in result.get("words", []):
        status = "X" if word_data["has_error"] else "OK"
        print(
            f'\n{status} {word_data["word"]:<15s} '
            f'score={word_data["score"]:5.1f} '
            f'errors={word_data["n_errors"]}/{word_data["n_phonemes"]}'
        )

        for phone in word_data["phonemes"]:
            marker = " <- ERROR" if phone["error"] else ""
            print(
                f'    /{phone["phone"]:<4s}/ '
                f'score={phone["score"]:5.1f} '
                f'GOP={phone["gop"]:+6.2f} '
                f'pherr={phone["pherr_prob"]:.2f} '
                f'aligned={phone.get("aligned", False)}{marker}'
            )


def main():
    parser = argparse.ArgumentParser(
        description="Voice Correction Pipeline v2.1"
    )
    parser.add_argument("--audio", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_PHERR_THRESHOLD
    )
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--input", type=str)
    parser.add_argument("--audio-dir", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    assessor = PronunciationAssessorV2(
        checkpoint_path=args.checkpoint,
        device=device,
        pherr_threshold=args.threshold,
    )

    if args.batch:
        if not args.input:
            parser.error("--input is required in batch mode")

        import openpyxl

        audio_dir = (
            Path(args.audio_dir)
            if args.audio_dir
            else Path(args.input).parent / "audio_files"
        )

        workbook = openpyxl.load_workbook(args.input, read_only=True)
        worksheet = workbook.active
        items = []

        for row in worksheet.iter_rows(min_row=2, values_only=True):
            if len(row) < 2:
                continue

            filename, content = row[0], row[1]
            if not filename or not content:
                continue

            audio_path = audio_dir / str(filename)
            if audio_path.exists():
                items.append((str(audio_path), str(content)))

        workbook.close()

        if args.limit > 0:
            items = items[:args.limit]

        print(f"Processing {len(items)} files...", file=sys.stderr)
        results = assessor.assess_batch(items)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}", file=sys.stderr)
        else:
            for result in results:
                if args.json:
                    print(json.dumps(result, ensure_ascii=False))
                else:
                    print_result(result)

    else:
        if not args.audio or not args.text:
            parser.error("--audio and --text are required")

        result = assessor.assess(args.audio, args.text)

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_result(result)


if __name__ == "__main__":
    main()
