import urllib
import zipfile
from pathlib import Path

import pandas as pd
from datasets import Dataset

from fortepyan import MidiFile
from fortepyan import config as C

HANON_URL = "https://storage.googleapis.com/sacrebleu-development/hanon.zip"


def main():
    hanon_folder = download_hanon()

    records = prepare_records(hanon_folder)
    dataset = Dataset.from_list(records)

    dataset_name = "roszcz/hanon"

    dataset.push_to_hub(repo_id=dataset_name, token=C.HF_TOKEN, split="train")


def download_hanon():
    tmp_zip_path = "/tmp/hanon.zip"
    urllib.request.urlretrieve(HANON_URL, tmp_zip_path)

    file_zip = zipfile.ZipFile(tmp_zip_path)
    file_zip.extractall(path="/tmp")

    hanon_root = "/tmp/hanon"
    return hanon_root


def prepare_records(hanon_folder: str):
    hanon = Path(hanon_folder)
    paths = hanon.rglob("*.mid")

    records = []
    for path in paths:
        mf = MidiFile(str(path), apply_sustain=False)
        cc = mf._midi.instruments[0].control_changes
        cc_frame = pd.DataFrame(
            {
                "number": [c.number for c in cc],
                "value": [c.value for c in cc],
                "time": [c.time for c in cc],
            }
        )
        hanon_label = path.parent.parts[-1]
        record = {
            "notes": mf.df,
            "label": hanon_label,
            "control_changes": cc_frame,
            "midi_filename": path.name,
        }
        records.append(record)

    return records


if __name__ == "__main__":
    main()
