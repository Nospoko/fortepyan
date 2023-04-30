import boto3
import pandas as pd
import sqlalchemy as sa

from fortepyan import config as C


def make_engine():
    url = sa.engine.make_url(C.POSTGRES_DSN)
    engine = sa.create_engine(url)
    return engine


def make_s3_client():
    s3_client = boto3.client("s3", endpoint_url=C.AWS_ENDPOINT)
    return s3_client


def get_user_records(user_id: int, engine) -> pd.DataFrame:
    query = f"""
        SELECT
            blob.key,
            blob.filename,
            blob.checksum,
            midi_record.created_at,
            midi_record.duration,
            midi_record.number_of_notes,
            u.email
        FROM
            active_storage_blobs AS blob
        INNER JOIN
            active_storage_attachments AS asa ON blob.id = asa.blob_id
        INNER JOIN
            midi_records AS midi_record ON asa.record_id = midi_record.id
        INNER JOIN
            users AS u ON midi_record.user_id = u.id
        WHERE asa.record_type = 'MidiRecord' AND midi_record.user_id = {user_id}
    """
    df = pd.read_sql(query, con=engine)
    return df


def get_all_records(engine) -> pd.DataFrame:
    query = """
        SELECT
            blob.key,
            blob.filename,
            blob.checksum,
            midi_record.id as record_id,
            midi_record.created_at,
            midi_record.duration,
            midi_record.number_of_notes,
            u.email,
            u.username,
            u.id as user_id
        FROM
            midi_records AS midi_record
        INNER JOIN
            active_storage_attachments AS asa ON midi_record.id = asa.record_id
        INNER JOIN
            active_storage_blobs AS blob ON asa.blob_id = blob.id
        INNER JOIN
            users AS u ON midi_record.user_id = u.id
    """
    df = pd.read_sql(query, con=engine)
    return df
