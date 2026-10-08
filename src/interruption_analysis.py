import os
import json


INPUT_FILE = "data/conversation_turns.json"


def analyze_interruptions():

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
            "to analyze interruptions."
        )

        print(
            "At least two conversation turns "
            "are required."
        )

        return


    interruptions = []


    # =====================================================
    # CHECK CONSECUTIVE TURNS
    # =====================================================

    for i in range(len(turns) - 1):

        current = turns[i]

        next_turn = turns[i + 1]


        # Ignore same-speaker consecutive turns
        if current["speaker"] == next_turn["speaker"]:
            continue


        current_end = current["end"]

        next_start = next_turn["start"]


        # If next speaker starts before
        # current speaker finishes
        if next_start < current_end:

            overlap = round(
                current_end - next_start,
                2
            )


            interruption = {

                "interrupted_speaker":
                    current["speaker"],

                "interrupting_speaker":
                    next_turn["speaker"],

                "overlap_seconds":
                    overlap,

                "interruption_start":
                    next_start,

                "previous_speaker_end":
                    current_end
            }


            interruptions.append(
                interruption
            )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print(
        "\nInterruption Analysis\n"
    )

    print(
        "-" * 60
    )


    if not interruptions:

        print(
            "No interruptions detected."
        )

    else:

        for interruption in interruptions:

            print(
                f"{interruption['interrupting_speaker']} "
                f"interrupted "
                f"{interruption['interrupted_speaker']}"
            )

            print(
                f"Overlap: "
                f"{interruption['overlap_seconds']} seconds"
            )

            print(
                "-" * 60
            )


    print(
        f"\nTotal interruptions: "
        f"{len(interruptions)}"
    )


if __name__ == "__main__":

    analyze_interruptions()