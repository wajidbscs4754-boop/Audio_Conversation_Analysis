import os
import json


INPUT_FILE = "data/conversation_turns.json"


def classify_latency(latency):
    """
    Classify the response gap.
    These are initial heuristic thresholds.
    """

    if latency < 0:
        return "overlap"

    elif latency < 0.5:
        return "very short"

    elif latency < 1.5:
        return "short"

    elif latency < 3.0:
        return "moderate"

    else:
        return "long"


def analyze_latency():

    if not os.path.exists(INPUT_FILE):

        print(
            f"File not found: {INPUT_FILE}"
        )

        return


    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        turns = json.load(file)


    if len(turns) < 2:

        print(
            "\nNot enough speaker turns "
            "to calculate latency."
        )

        print(
            "At least two conversation turns "
            "from different speakers are needed."
        )

        return


    latency_results = []


    for i in range(len(turns) - 1):

        current_turn = turns[i]

        next_turn = turns[i + 1]


        current_end = current_turn["end"]

        next_start = next_turn["start"]


        latency = round(
            next_start - current_end,
            2
        )


        classification = classify_latency(
            latency
        )


        result = {

            "from_speaker": current_turn[
                "speaker"
            ],

            "to_speaker": next_turn[
                "speaker"
            ],

            "current_end": current_end,

            "next_start": next_start,

            "latency": latency,

            "classification": classification
        }


        latency_results.append(result)


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print(
        "\nLatency Analysis\n"
    )

    print(
        "-" * 60
    )


    for result in latency_results:

        print(
            f"{result['from_speaker']} -> "
            f"{result['to_speaker']}"
        )

        print(
            f"Latency: "
            f"{result['latency']} seconds"
        )

        print(
            f"Type: "
            f"{result['classification']}"
        )

        print(
            "-" * 60
        )


    # =====================================================
    # AVERAGE LATENCY
    # =====================================================

    valid_latencies = [

        r["latency"]

        for r in latency_results

        if r["latency"] >= 0
    ]


    if valid_latencies:

        average = round(
            sum(valid_latencies)
            / len(valid_latencies),
            2
        )

        print(
            f"\nAverage latency: "
            f"{average} seconds"
        )


if __name__ == "__main__":

    analyze_latency()