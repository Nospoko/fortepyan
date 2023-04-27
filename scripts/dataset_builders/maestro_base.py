import urllib
import zipfile

import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from fortepyan import MidiFile
from fortepyan import config as C

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def download_maestro():
    maestro_name = "maestro-v3.0.0"
    tmp_zip_path = f"/tmp/{maestro_name}.zip"

    urllib.request.urlretrieve(MAESTRO_URL, tmp_zip_path)

    file_zip = zipfile.ZipFile(tmp_zip_path)
    file_zip.extractall(path="/tmp")

    maestro_root = f"/tmp/{maestro_name}"

    return maestro_root


def load_maestro(maestro_root: str) -> pd.DataFrame:
    df = pd.read_csv(f"{maestro_root}/maestro-v3.0.0.csv")
    df["midi_path"] = df.midi_filename.apply(lambda mf: f"{maestro_root}/{mf}")

    return df


def make_maestro_records(mdf: pd.DataFrame) -> list[dict]:
    records = []
    for it, row in tqdm(mdf.iterrows(), total=mdf.shape[0], desc="Building maestro records"):
        mf = MidiFile(row.midi_path, apply_sustain=False)
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
            "composer": row.canonical_composer,
            "title": row.canonical_title,
            "split": row.split,
            "year": row.year,
            "midi_filename": row.midi_filename,
        }
        records.append(record)

    return records


def main():
    maestro_root = download_maestro()
    mdf = load_maestro(maestro_root)

    records = make_maestro_records(mdf)
    dataset = Dataset.from_list(records)

    dataset_name = "roszcz/maestro-v1"

    for split in ["validation", "test", "train"]:
        dataset_split = dataset.filter(lambda r: r["split"] == split)
        dataset_split = dataset_split.remove_columns("split")

        dataset_split.push_to_hub(repo_id=dataset_name, token=C.HF_TOKEN, split=split)


if __name__ == "__main__":
    main()
