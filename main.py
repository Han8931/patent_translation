# main_batch.py

import asyncio
import time
import random
import traceback
from functools import partial
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from app.batch_tools import download_s3_file, translate_docx, list_s3_keys
from app.config import APIConfig, LLMModel
from app.model.data_schemas import Bucket

# ============================================================
# Settings
# ============================================================

MAX_CONCURRENCY = 5
OUTPUT_DIR = Path("./outputs")
FAILED_KEYS_PATH = OUTPUT_DIR / "failed_keys.txt"

MAX_SECTION_CHARS = 80_000
S3_PREFIX = "patent/260227"

RETRY_ROUNDS = 2
S3_RETRIES_PER_DOC = 2
LLM_RETRIES_PER_DOC = 2

BASE_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 20.0
JITTER_SEC = 0.3


def make_bucket(config: APIConfig) -> Bucket:
    return Bucket(
        name="aip-rag-ai",
        access_key=config.S3_UPLOAD_BUCKET_ACCESS_KEY,
        secret_key=config.S3_UPLOAD_BUCKET_SECRET_KEY,
    )


def backoff_delay(attempt_idx: int) -> float:
    delay = min(MAX_BACKOFF_SEC, BASE_BACKOFF_SEC * (2 ** attempt_idx))
    delay += random.uniform(0, JITTER_SEC)
    return delay


def load_failed_keys(path: Path) -> List[str]:
    if not path.exists():
        return []
    keys: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if key:
                keys.append(key)
    return list(dict.fromkeys(keys))


def save_failed_keys(path: Path, keys: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key in keys:
            f.write(f"{key}\n")


@dataclass
class AttemptResult:
    key: str
    ok: bool
    stage: str
    out_path: Optional[Path]
    error: Optional[str]
    elapsed_s: float


async def _retry_async(fn, *, retries: int, what: str) -> Tuple[bool, Optional[str]]:
    last_err = None
    for i in range(retries + 1):
        try:
            await fn()
            return True, None
        except Exception as e:
            last_err = (
                f"{what} attempt {i+1}/{retries+1} failed: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            if i < retries:
                await asyncio.sleep(backoff_delay(i))
    return False, last_err


async def translate_one(key: str, bucket, llm_config, semaphore, *, compare: bool = False) -> AttemptResult:
    async with semaphore:

        key = str(key)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"[START] {key}")

        t0 = time.perf_counter()
        out_path: Optional[Path] = None

        local_path_holder: Dict[str, Path] = {}

        async def _do_s3():
            loop = asyncio.get_running_loop()
            p = await loop.run_in_executor(None, partial(download_s3_file, key, bucket))
            local_path_holder["path"] = p

        ok, err = await _retry_async(_do_s3, retries=S3_RETRIES_PER_DOC, what=f"[S3] {key}")
        if not ok:
            return AttemptResult(key, False, "s3", None, err, time.perf_counter() - t0)

        local_path = local_path_holder["path"]

        async def _do_llm():
            nonlocal out_path

            out_path = OUTPUT_DIR / Path(key).name
            if out_path.suffix.lower() != ".docx":
                out_path = out_path.with_suffix(".docx")

            loop = asyncio.get_running_loop()

            task = partial(
                translate_docx,
                src_docx_path=local_path,
                out_docx_path=out_path,
                compare=compare,
                model="gpt-oss-120b",
                target_lang="English",
                max_chars_per_chunk=5000,
                context_chars=1000,
                chunk_overlap=0,
                max_section_chars=MAX_SECTION_CHARS,
                base_url=llm_config.GPT_OSS_URL_API,
                api_key="dummy",
                prompt_default="patent_kr2en_body_v2_compact",
                prompt_claims="patent_kr2en_claims_v2_compact",
                prompt_claims_indep="patent_kr2en_claims_indep_v2_compact",
                prompt_claims_dep="patent_kr2en_claims_dep_v2_compact",
                prompt_abstract="patent_kr2en_abstract_v2_compact",
            )

            await loop.run_in_executor(None, task)

        ok, err = await _retry_async(_do_llm, retries=LLM_RETRIES_PER_DOC, what=f"[LLM] {key}")

        if not ok:
            return AttemptResult(key, False, "llm", out_path, err, time.perf_counter() - t0)

        return AttemptResult(key, True, "ok", out_path, None, time.perf_counter() - t0)
