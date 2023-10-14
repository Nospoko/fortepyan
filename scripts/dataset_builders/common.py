import json

import pandas as pd
from tqdm import tqdm

from fortepyan import MidiFile
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


def prepare_hf_records(records: list[dict]):
    hf_records = []
    for record in tqdm(records):
        path = record.pop("path")
        try:
            mf = MidiFile(str(path), apply_sustain=False)
            cc = mf._midi.instruments[0].control_changes
            cc_frame = pd.DataFrame(
                {
                    "number": [c.number for c in cc],
                    "value": [c.value for c in cc],
                    "time": [c.time for c in cc],
                }
            )
            record = {
                "notes": mf.df,
                "control_changes": cc_frame,
                "source": json.dumps(record),
            }
            hf_records.append(record)
        except Exception as e:
            print("Failed:", path)
            print(e)
            print("<++++++++++++>")

    return hf_records
