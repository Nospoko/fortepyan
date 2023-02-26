from datasets import Dataset

from fortepyan import MidiFile
from fortepyan import config as C
from fortepyan.web import database as db


def prepare_records(engine, s3_client) -> list[dict]:
    users_map = {
        9: "bb",
        3: "nk",
        6: "js",
        11: "ad",
        1: "tr",
        4: "mb",
    }

    records = []
    for user_id, user_tag in users_map.items():
        data = db.get_user_records(user_id, engine)
        # Order by length
        data = data.sort_values("number_of_notes", ascending=False)

        practice_file = data.iloc[0]
        print(practice_file)
        print("---" * 10)
        savepath = f"tmp/{practice_file.filename}"
        s3_client.download_file(
            Bucket="piano-for-ai",
            Key=practice_file.key,
            Filename=savepath,
        )
        mf = MidiFile(savepath)

        record = {
            "notes": mf.df,
            "control_changes": mf.control_frame,
            "user": user_tag,
            "record_id": practice_file.name,
        }
        records.append(record)

    return records


if __name__ == "__main__":
    engine = db.make_engine()
    s3_client = db.make_s3_client()
    records = prepare_records(engine=engine, s3_client=s3_client)

    dataset = Dataset.from_list(records)
    dataset_name = "roszcz/internship-midi-data-science"
    dataset.push_to_hub(
        repo_id=dataset_name,
        token=C.HF_TOKEN,
    )
