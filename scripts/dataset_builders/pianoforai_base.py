import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_dataset

from fortepyan import MidiFile
from fortepyan import config as C
from fortepyan.web import database as db

from scripts.dataset_builders.common import process_record_sustain

bucket = "piano-for-ai"


def main():
    # This will download data from PianoRoll
    engine = db.make_engine()
    s3_client = db.make_s3_client()

    df = db.get_all_records(engine)

    # Convert that data into HF dataset
    records = make_pianoforai_records(df=df, s3_client=s3_client)
    dataset = Dataset.from_list(records)

    # Upload
    dataset_name = "roszcz/pianofor-ai-base-v2"
    dataset.push_to_hub(repo_id=dataset_name, token=C.HF_TOKEN, split="train")


def make_pianoforai_records(
    df: pd.DataFrame,
    s3_client,
) -> list[dict]:
    records = []

    for it, row in tqdm(df.iterrows(), total=df.shape[0], desc="Building Piano For AI records"):
        savepath = f"tmp/pfa-2024-01-15/{row.filename}"
        try:
            s3_client.download_file(Bucket=bucket, Key=row.key, Filename=savepath)
            mf = MidiFile(savepath, apply_sustain=False)

            cc_frame = mf.control_frame
            source = {
                "midi_filename": row.filename,
                "dataset": "piano-for-ai",
                "record_id": row.record_id,
                "user_id": row.user_id,
                "user": row.username,
            }
            record = {
                "notes": mf.df,
                "control_changes": cc_frame,
                "source": json.dumps(source),
            }
            records.append(record)
        except Exception as e:
            print("Fail!", e)
            print(row.to_dict())
            print("=============" * 3)
        # os.remove(savepath)

    return records


def main_sustain():
    new_dataset_name = "roszcz/pianofor-ai-sustain-v2"

    dataset_name = "roszcz/pianofor-ai-base-v2"
    dataset = load_dataset(dataset_name, split="train")

    fn_kwargs = {"sustain_threshold": 62}
    new_dataset = dataset.map(process_record_sustain, fn_kwargs=fn_kwargs, load_from_cache_file=False, num_proc=10)
    new_dataset = new_dataset.remove_columns("control_changes")

    new_dataset.push_to_hub(
        repo_id=new_dataset_name,
        token=C.HF_TOKEN,
        split="train",
    )


if __name__ == "__main__":
    main()
    main_sustain()
