import asyncio
import time
from functools import partial
from pathlib import Path
from typing import List

from app.batch_tools import load_filename_list, download_s3_file, translate_docx
from app.config import APIConfig, LLMModel
from app.model.data_schemas import Bucket

# ============================================================
# Settings
# ============================================================

MAX_CONCURRENCY = 5  # target 4–5 concurrent docs
BASE_DIR = Path("patent")
FILENAME_LIST_PATH = Path("./app/filename_list.txt")
OUTPUT_DIR = Path("./outputs")


# ============================================================
# Helpers
# ============================================================

def make_bucket(config: APIConfig) -> Bucket:
    return Bucket(
        name="aip-rag-ai",
        access_key=config.S3_UPLOAD_BUCKET_ACCESS_KEY,
        secret_key=config.S3_UPLOAD_BUCKET_SECRET_KEY,
    )


async def translate_one(
    filename: str,
    bucket: Bucket,
    llm_config: LLMModel,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        file_path = BASE_DIR / filename
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            local_path = download_s3_file(file_path, bucket)
            out_path = OUTPUT_DIR / filename

            # translate_docx is synchronous; run in default thread executor
            loop = asyncio.get_running_loop()
            task = partial(
                translate_docx,
                src_docx_path=local_path,
                out_docx_path=out_path,
                model="gpt-oss-120b",
                target_lang="English",
                max_chars_per_chunk=2500,
                context_chars=1000,
                chunk_overlap=0,
                base_url=llm_config.GPT_OSS_URL_API,
                api_key="dummy",
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


# ============================================================
# Entry
# ============================================================

async def main():
    config = APIConfig()
    llm_config = LLMModel()
    bucket = make_bucket(config)

    filenames: List[str] = load_filename_list(FILENAME_LIST_PATH)
    if not filenames:
        print("No filenames found to process.")
        return

    concurrency = min(MAX_CONCURRENCY, max(1, len(filenames)))
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        asyncio.create_task(translate_one(fn, bucket, llm_config, sem))
        for fn in filenames
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
