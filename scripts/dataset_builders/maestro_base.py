import json
import urllib
import zipfile

import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_dataset

from fortepyan import MidiFile
from fortepyan import config as C
from scripts.dataset_builders.common import process_record_sustain

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
        source = {
            "composer": row.canonical_composer,
            "title": row.canonical_title,
            "split": row.split,
            "year": row.year,
            "midi_filename": row.midi_filename,
            "dataset": "maestro",
        }
        record = {
            "notes": mf.df,
            "control_changes": cc_frame,
            "source": json.dumps(source),
            # Maestro specific, remove before upload
            "split": row.split,
        }
        records.append(record)

    return records


def main():
    maestro_root = download_maestro()
    mdf = load_maestro(maestro_root)

    records = make_maestro_records(mdf)
    dataset = Dataset.from_list(records)

    dataset_name = "roszcz/maestro-base-v2"

    for split in ["validation", "test", "train"]:
        dataset_split = dataset.filter(lambda r: r["split"] == split)
        dataset_split = dataset_split.remove_columns("split")

        dataset_split.push_to_hub(repo_id=dataset_name, token=C.HF_TOKEN, split=split)


def main_sustain():
    new_dataset_name = "roszcz/maestro-sustain-v2"

    for split in ["test", "validation", "train"]:
        dataset = load_dataset("roszcz/maestro-base-v2", split=split)

        fn_kwargs = {"sustain_threshold": 62}
        new_dataset = dataset.map(
            process_record_sustain,
            fn_kwargs=fn_kwargs,
            load_from_cache_file=False,
            num_proc=8,
        )
        new_dataset = new_dataset.remove_columns("control_changes")

        new_dataset.push_to_hub(
            repo_id=new_dataset_name,
            token=C.HF_TOKEN,
            split=split,
        )


if __name__ == "__main__":
    main()
