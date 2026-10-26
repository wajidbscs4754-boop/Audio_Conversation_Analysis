
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "evaluation" / "speechocean762"
SCORES_FILE = DATASET_DIR / "resource" / "scores.json"
OUTPUT_FILE = DATASET_DIR / "model_evaluation_results.csv"

# Import the existing model without modifying it
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wavlm_pipeline import PronunciationAssessorV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10,
        help="Maximum score difference considered acceptable"
    )
    args = parser.parse_args()

    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        human_scores = json.load(f)

    audio_files = sorted(
        (DATASET_DIR / "WAVE").rglob("*.WAV")
    )

    # Match audio filenames with human-rated records
    samples = []
    for audio_path in audio_files:
        sample_id = audio_path.stem
        record = human_scores.get(sample_id)

        if (
            record
            and record.get("text")
            and record.get("accuracy") is not None
        ):
            samples.append((audio_path, record))

    if args.limit > 0:
        samples = samples[:args.limit]

    if not samples:
        print("No matching audio files and human scores found.")
        return

    print(f"Samples to evaluate: {len(samples)}")
    print(f"Accepted score difference: <= {args.tolerance:.1f} points")
    print("Loading pronunciation model...")

    assessor = PronunciationAssessorV2()

    results = []
    skipped = 0

    for index, (audio_path, record) in enumerate(samples, start=1):
        try:
            prediction = assessor.assess(
                str(audio_path),
                str(record["text"])
            )

            if (
                prediction.get("error")
                or prediction.get("overall_score") is None
            ):
                skipped += 1
                print(f"[{index}] Skipped: {audio_path.name}")
                continue

            model_score = float(prediction["overall_score"])
            human_score = float(record["accuracy"]) * 10

            absolute_error = abs(model_score - human_score)
            within_tolerance = absolute_error <= args.tolerance

            results.append({
                "audio_id": audio_path.stem,
                "text": record["text"],
                "human_score_100": human_score,
                "model_score_100": model_score,
                "absolute_error": absolute_error,
                "within_tolerance": within_tolerance,
            })

            print(
                f"[{index}/{len(samples)}] {audio_path.name}: "
                f"Human={human_score:.1f}, "
                f"Model={model_score:.1f}, "
                f"Error={absolute_error:.1f}, "
                f"Within tolerance={within_tolerance}"
            )

        except Exception as exc:
            skipped += 1
            print(f"[{index}] Error on {audio_path.name}: {exc}")

    if not results:
        print("No samples were successfully evaluated.")
        return

    # Save per-sample results
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    human = np.array([r["human_score_100"] for r in results])
    model = np.array([r["model_score_100"] for r in results])
    errors = np.abs(model - human)

    # Percentage of predictions within the chosen tolerance
    agreement_count = sum(r["within_tolerance"] for r in results)
    agreement_percentage = agreement_count / len(results) * 100

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Successfully evaluated: {len(results)}")
    print(f"Skipped / failed: {skipped}")
    print(f"Mean Absolute Error (MAE): {np.mean(errors):.2f}")
    print(
        "Root Mean Squared Error (RMSE): "
        f"{np.sqrt(np.mean((model - human) ** 2)):.2f}"
    )
    print(
        f"Within-{args.tolerance:g}-point agreement: "
        f"{agreement_percentage:.2f}% "
        f"({agreement_count}/{len(results)} samples)"
    )

    if (
        len(results) >= 2
        and np.std(human) > 0
        and np.std(model) > 0
    ):
        print(
            f"Pearson correlation: "
            f"{pearsonr(human, model).statistic:.3f}"
        )
        print(
            f"Spearman correlation: "
            f"{spearmanr(human, model).statistic:.3f}"
        )
    else:
        print("Correlation requires at least two samples with varying scores.")

    print(f"\nCSV saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
