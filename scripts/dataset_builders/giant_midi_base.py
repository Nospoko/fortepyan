import json
import urllib
import zipfile
from pathlib import Path

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

    dataset_name = "roszcz/giant-midi-base-v2"

    dataset.push_to_hub(repo_id=dataset_name, token=C.HF_TOKEN, split="train")


def download_giant_midi():
    tmp_zip_path = "/tmp/giant_midi.zip"
    urllib.request.urlretrieve(GIANT_MIDI_URL, tmp_zip_path)

    file_zip = zipfile.ZipFile(tmp_zip_path)
    file_zip.extractall(path="/tmp/giant_midi")

    giant_midi_root = "/tmp/giant_midi/midis/"
    return giant_midi_root


def decode_filename(name: str) -> dict:
    parts = name.split(",")
    first_name = parts[1]
    second_name = parts[0]
    artist = f"{first_name} {second_name}"
    decoded = dict(
        artist=artist,
        title=" ".join(parts[2:-2]),
        youtube_id=parts[-1].split(".")[0],
    )
    return decoded


def prepare_records(paths: list[str]):
    records = []
    for path in tqdm(paths):
        try:
            mf = MidiFile(str(path), apply_sustain=False)
            cc_frame = mf.control_frame
            source = decode_filename(path.name) | {"dataset": "giant-midi"}
            record = {
                "notes": mf.df,
                "control_changes": cc_frame,
                "source": json.dumps(source),
            }
            records.append(record)
        except Exception as e:
            print("Failed:", path)
            print(e)
            print("<++++++++++++>")

    return records


def main_sustain():
    new_dataset_name = "roszcz/giant-midi-sustain-v2"

    dataset_name = "roszcz/giant-midi-base-v2"
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
