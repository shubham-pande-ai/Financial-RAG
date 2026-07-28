import os
from pathlib import Path
from minio import Minio

# MinIO Config
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Ensure buckets exist
for bucket in ["annual-reports", "concall-transcripts"]:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

# Upload docs
docs_dir = Path("./screener_docs")
for symbol_dir in docs_dir.iterdir():
    if not symbol_dir.is_dir():
        continue
        
    symbol = symbol_dir.name.lower()
    for pdf in symbol_dir.rglob("*.pdf"):
        # Very simple heuristic based on filename
        if "Transcript" in pdf.name or "concall" in pdf.name.lower() or "call" in pdf.name.lower():
            bucket = "concall-transcripts"
        else:
            bucket = "annual-reports"
            
        object_name = f"{symbol}/{pdf.name}"
        print(f"Uploading {pdf.name} to {bucket}/{object_name}")
        client.fput_object(bucket, object_name, str(pdf))

print("Upload complete!")
