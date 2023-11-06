import urllib
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_dataset

from fortepyan import MidiFile
from fortepyan import config as C
from scripts.dataset_builders.common import process_record_sustain

GIANT_MIDI_URL = "https://storage.googleapis.com/sacrebleu-development/giant_midis_v1.2.zip"


def main():
    giant_midi_folder = download_giant_midi()

    giant_midi = Path(giant_midi_folder)
    paths = giant_midi.rglob("*.mid")
    paths = list(paths)

    records = prepare_records(paths)
    dataset = Dataset.from_list(records)

    dataset_name = "roszcz/giant-midi-base"

    dataset.push_to_hub(repo_id=dataset_name, token=C.HF_TOKEN, split="train")


def download_giant_midi():
    tmp_zip_path = "/tmp/giant_midi.zip"
    urllib.request.urlretrieve(GIANT_MIDI_URL, tmp_zip_path)

    file_zip = zipfile.ZipFile(tmp_zip_path)
    file_zip.extractall(path="/tmp/giant_midi")

    giant_midi_root = "/tmp/giant_midi/midis/"
    return giant_midi_root


def prepare_records(paths: list[str]):
    records = []
    for path in tqdm(paths):
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
                "midi_filename": path.name,
            }
            records.append(record)
        except Exception as e:
            print("Failed:", path)
            print(e)
            print("<++++++++++++>")

    return records


def main_sustain():
    new_dataset_name = "roszcz/giant-midi-sustain"

    dataset_name = "roszcz/giant-midi-base"
    dataset = load_dataset(dataset_name, split="train")

    fn_kwargs = {"sustain_threshold": 62}
    new_dataset = dataset.map(process_record_sustain, fn_kwargs=fn_kwargs, load_from_cache_file=False, num_proc=10)
    new_dataset = new_dataset.remove_columns("control_changes")

    new_dataset.push_to_hub(
        repo_id=new_dataset_name,
        token=C.HF_TOKEN,
    )


if __name__ == "__main__":
    main()
    main_sustain()