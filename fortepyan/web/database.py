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
            b.key,
            b.filename,
            b.checksum,
            c.created_at,
            c.duration,
            c.number_of_notes,
            u.email
        FROM
            active_storage_blobs AS b
        INNER JOIN
            active_storage_attachments AS a ON b.id = a.blob_id
        INNER JOIN
            midi_records AS c ON a.record_id = c.id
        INNER JOIN
            users AS u ON c.user_id = u.id
        WHERE a.record_type = 'MidiRecord' AND c.user_id = {user_id}
    """
    df = pd.read_sql(query, con=engine)
    return df
