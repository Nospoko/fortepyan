import pandas as pd
from datasets import load_dataset

from fortepyan import config as C
from fortepyan.midi import tools as midi_tools


def process_record(record: dict, sustain_threshold: int) -> dict:
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


def main():
    new_dataset_name = "roszcz/pianofor-ai-sustain-v2"

    dataset_name = "roszcz/pianofor-ai-base"
    dataset = load_dataset(dataset_name, use_auth_token=C.HF_TOKEN)

    fn_kwargs = {"sustain_threshold": 62}
    new_dataset = dataset.map(process_record, fn_kwargs=fn_kwargs, load_from_cache_file=False)
    new_dataset = new_dataset.remove_columns("control_changes")

    new_dataset.push_to_hub(
        repo_id=new_dataset_name,
        token=C.HF_TOKEN,
    )


if __name__ == "__main__":
    main()
