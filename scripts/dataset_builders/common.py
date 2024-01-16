import pandas as pd

from fortepyan.midi import tools as midi_tools


def process_record_sustain(record: dict, sustain_threshold: int) -> dict:
    df = pd.DataFrame(record["notes"])
    cc = pd.DataFrame(record["control_changes"])

    # Filter control_changes to only include sustain pedal events and reset the index
    sustain = cc[cc.number == 64].reset_index(drop=True)

    # Apply sustain pedal events to notes data
    df = midi_tools.apply_sustain(
        df=df,
        sustain=sustain,
        sustain_threshold=sustain_threshold,
    )

    # Create a new dictionary with the processed notes data
    record_update = {"notes": df}

    return record_update
