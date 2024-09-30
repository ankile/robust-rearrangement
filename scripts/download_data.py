import os
import argparse
import boto3
from botocore import UNSIGNED
from botocore.client import Config

def download_s3_directory(s3_client, bucket, s3_path, local_path):
    """
    Download a directory from S3 to a local path
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=s3_path):
        if "Contents" in result:
            for key in result["Contents"]:
                relative_path = os.path.relpath(key["Key"], s3_path)
                local_file_path = os.path.join(local_path, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3_client.download_file(bucket, key["Key"], local_file_path)
                print(f"Downloaded: {key['Key']} to {local_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Download data from S3 for a specific task.")
    parser.add_argument("--task", required=True, help="Task name (e.g., lamp)")
    args = parser.parse_args()

    # S3 bucket and base path
    bucket = "iai-robust-rearrangement"
    base_path = f"data/processed/diffik/sim/{args.task}/teleop"

    # Determine local directory for downloads
    data_dir = os.getenv("DATA_DIR_PROCESSED", "./data")
    local_dir = os.path.join(data_dir, "processed", "diffik", "sim", args.task, "teleop")

    # Create S3 client without requiring credentials (for public buckets)
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    print(f"Downloading files for task: {args.task}")
    print(f"From S3 path: s3://{bucket}/{base_path}")
    print(f"To local directory: {local_dir}")

    download_s3_directory(s3_client, bucket, base_path, local_dir)

    print("Download complete.")

if __name__ == "__main__":
    main()