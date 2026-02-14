from __future__ import annotations

import asyncio
import os
import time
from functools import partial
from pathlib import Path
from typing import List

from app.config import APIConfig, LLMModel
from app.model.data_schemas import Bucket, SecretValue
from app.translate import translate_docx
from app.utils import (
    download_s3_file,
    list_local_docx_files,
    load_filename_list,
    resolve_local_file,
)

# ============================================================
# Settings
# ============================================================

MAX_CONCURRENCY = 5
BASE_DIR = Path("./documents")
FILENAME_LIST_PATH = Path("./app/filename_list.txt")
OUTPUT_DIR = Path("./outputs")
USE_S3 = os.getenv("USE_S3", "false").lower() in {"1", "true", "yes"}


def make_bucket(config: APIConfig) -> Bucket:
    return Bucket(
        name=config.S3_UPLOAD_BUCKET_NAME,
        access_key=SecretValue(config.S3_UPLOAD_BUCKET_ACCESS_KEY),
        secret_key=SecretValue(config.S3_UPLOAD_BUCKET_SECRET_KEY),
    )


def load_targets() -> List[str]:
    if FILENAME_LIST_PATH.exists():
        return load_filename_list(FILENAME_LIST_PATH)
    return list_local_docx_files(BASE_DIR)


async def translate_one(
    filename: str,
    bucket: Bucket | None,
    llm_config: LLMModel,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        try:
            if USE_S3:
                if bucket is None:
                    raise ValueError("S3 mode requires bucket config.")
                local_path = download_s3_file(BASE_DIR / filename, bucket)
            else:
                local_path = resolve_local_file(filename, BASE_DIR)

            out_path = OUTPUT_DIR / local_path.name

            loop = asyncio.get_running_loop()
            task = partial(
                translate_docx,
                src_docx_path=local_path,
                out_docx_path=out_path,
                model=llm_config.GPT_OSS_MODEL,
                target_lang="English",
                max_chars_per_chunk=2500,
                context_chars=1000,
                chunk_overlap=0,
                layout_mode="relaxed",
                abstract_target_words=150,
                abstract_word_tolerance=5,
                base_url=llm_config.GPT_OSS_URL_API,
                api_key=llm_config.GPT_OSS_API_KEY,
                prompt_default="patent_kr2en_body_v1",
                prompt_claims="patent_kr2en_claims_v1",
                prompt_abstract="patent_kr2en_abstract_v1",
            )
            await loop.run_in_executor(None, task)
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            return

        dt = time.perf_counter() - t0
        print(f"[DONE] {filename}: {dt:.2f}s")


async def main() -> None:
    config = APIConfig()
    llm_config = LLMModel()
    bucket = make_bucket(config) if USE_S3 else None

    filenames = load_targets()
    if not filenames:
        print("No filenames found. Add .docx files to ./documents or create app/filename_list.txt")
        return

    concurrency = min(MAX_CONCURRENCY, max(1, len(filenames)))
    sem = asyncio.Semaphore(concurrency)

    tasks = [asyncio.create_task(translate_one(fn, bucket, llm_config, sem)) for fn in filenames]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
