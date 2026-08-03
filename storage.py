import os

import boto3

from config import settings

"""
upload_dataset(job_id, content: bytes) → str   returns the S3 key
download_dataset(dataset_s3_key) → bytes        returns the file content
upload_adapter(job_id, local_dir) → str         returns the S3 prefix

"""

_profile = settings.aws_profile if settings.aws_profile else None
session = boto3.Session(profile_name=_profile)
s3 = session.client("s3")
# When aws_profile is set, uses ~/.aws/config profile (local dev).
# When empty/None, boto3 falls back to AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY from env.


def upload_dataset(job_id, content: bytes):

    key = f"jobs/{job_id}/dataset.jsonl"
    s3.put_object(Bucket=settings.s3_bucket, Key=key, Body=content)

    return key


def download_dataset(dataset_s3_key):

    response = s3.get_object(Bucket=settings.s3_bucket, Key=dataset_s3_key)
    dataset = response["Body"].read()

    return dataset


def upload_adapter(job_id, local_dir) -> str:
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        s3_key = f"jobs/{job_id}/adapter/{filename}"
        s3.upload_file(local_path, settings.s3_bucket, s3_key)
    return f"s3://{settings.s3_bucket}/jobs/{job_id}/adapter/"


def download_adapter(job_id: str, adapter_s3_path: str) -> str:
    """
    Downloads all adapter files from S3 to a local temp directory.
    Returns the local directory path.

    adapter_s3_path format: s3://bucket/jobs/{id}/adapter/
    """

    local_dir = f"/tmp/{job_id}/adapter"
    os.makedirs(local_dir, exist_ok=True)

    # strip s3://bucket/ prefix to get the S3 prefix
    prefix = adapter_s3_path.replace(f"s3://{settings.s3_bucket}/", "")

    # list all files under this prefix
    response = s3.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)
        s3.download_file(settings.s3_bucket, key, local_path)

    return local_dir
