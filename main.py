import os, time
import asyncio
import shutil
import uuid
import fitz as pymupdf                        # MuPDF (pip install pymupdf)

from typing import List 
from pathlib import Path

from app.services import downloaded_asset
from app.batch_tools import load_filename_list, download_s3_file, extract_docx_text, translate_docx
from app.model.data_schemas import Bucket

from app.config import APIConfig, LLMModel

config = APIConfig()
llm_config = LLMModel()

base = Path("patent")
path = Path("./app/filename_list.txt")

filenames = load_filename_list(path)
# print(f"Loaded {len(filenames)} filenames")
# print(filenames[:10])


bucket = Bucket(
        name="aip-rag-ai",
        access_key=config.S3_UPLOAD_BUCKET_ACCESS_KEY,
        secret_key=config.S3_UPLOAD_BUCKET_SECRET_KEY,
)

for filename in filenames:
    file_path = base / filename

    local_path = download_s3_file(file_path, bucket)
    # texts = extract_docx_text(local_path)

    t0 = time.perf_counter()
    translate_docx(
        src_docx_path=local_path,
        out_docx_path=f"./outputs/{filename}",
        model="gpt-oss-120b",
        max_chars_per_chunk=2500,
        base_url=llm_config.GPT_OSS_URL_API,
        api_key="dummy"
    )
    dt = time.perf_counter() - t0
    print(f"[TIME] {Path(filename).name}: {dt:.2f}s")
    break
