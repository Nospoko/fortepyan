import os
import json
import urllib
import zipfile
from tqdm import tqdm

import pandas as pd
from datasets import Dataset, load_dataset

from fortepyan import MidiFile
from fortepyan import config as C
from scripts.dataset_builders.common import process_record_sustain

ATEPP_URL = "https://storage.googleapis.com/sacrebleu-development/atepp-1.1.zip"


def prepare_atepp_records(df: pd.DataFrame, atepp_midi_folder: str) -> list[dict]:
    columns = ["artist", "track", "composer", "youtube_links", "midi_path"]
    records = []
    for it, row in df.iterrows():
        path = os.path.join(atepp_midi_folder, row.midi_path)
        record = row[columns].to_dict() | {"path": path, "dataset": "atepp-1.1"}
        records.append(record)

    return records


def prepare_hf_records(records: list[dict]):
    hf_records = []
    for record in tqdm(records):
        path = record.pop("path")
        try:
            mf = MidiFile(str(path), apply_sustain=False)
            cc_frame = mf.control_frame

            source = json.dumps(record)
            record = {
                "notes": mf.df,
                "control_changes": cc_frame,
                "source": source,
            }
            hf_records.append(record)
        except Exception as e:
            print("Failed:", path)
            print(e)
            print("<++++++++++++>")

    return hf_records


def main():
    atepp_midi_folder = download_atepp_midi()
    meta_path = os.path.join(atepp_midi_folder, "ATEPP-metadata-1.1.csv")
    df = pd.read_csv(meta_path)

    atepp_records = prepare_atepp_records(df=df, atepp_midi_folder=atepp_midi_folder)

    records = prepare_hf_records(atepp_records)
    dataset = Dataset.from_list(records)

    dataset_name = "roszcz/atepp-1.1-base-v2"

    dataset.push_to_hub(repo_id=dataset_name, token=C.HF_TOKEN, split="train")


def main_sustain():
    new_dataset_name = "roszcz/atepp-1.1-sustain-v2"

    dataset_name = "roszcz/atepp-1.1-base-v2"
    dataset = load_dataset(dataset_name, split="train", use_auth_token=C.HF_TOKEN)

    fn_kwargs = {"sustain_threshold": 62}
    new_dataset = dataset.map(process_record_sustain, fn_kwargs=fn_kwargs, load_from_cache_file=False, num_proc=10)
    new_dataset = new_dataset.remove_columns("control_changes")

    new_dataset.push_to_hub(
        repo_id=new_dataset_name,
        token=C.HF_TOKEN,
    )


def download_atepp_midi():
    tmp_zip_path = "/tmp/atepp_midi.zip"
    urllib.request.urlretrieve(ATEPP_URL, tmp_zip_path)

    file_zip = zipfile.ZipFile(tmp_zip_path)
    file_zip.extractall(path="/tmp/atepp_midi")

    atepp_midi_root = "/tmp/atepp_midi/atepp"
    return atepp_midi_root


if __name__ == "__main__":
    # main()
    main_sustain()
