import os

import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from fortepyan import MidiFile
from fortepyan import config as C
from fortepyan.web import database as db

bucket = "piano-for-ai"


def main():
    engine = db.make_engine()
    s3_client = db.make_s3_client()

    df = db.get_all_records(engine)

    records = make_pianoforai_records(df, s3_client)

    dataset = Dataset.from_list(records)

    dataset_name = "roszcz/pianofor-ai-base"

    dataset.push_to_hub(repo_id=dataset_name, token=C.HF_TOKEN, split="train")


def make_pianoforai_records(
    df: pd.DataFrame,
    s3_client,
) -> list[dict]:
    records = []

    for it, row in tqdm(df.iterrows(), total=df.shape[0], desc="Building Piano For AI records"):
        savepath = f"/tmp/{row.filename}"
        s3_client.download_file(Bucket=bucket, Key=row.key, Filename=savepath)
        mf = MidiFile(savepath, apply_sustain=False)

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
            "midi_filename": row.filename,
            "record_id": row.record_id,
            "user_id": row.user_id,
            "user": row.username,
        }
        records.append(record)
        os.remove(savepath)

    return records
