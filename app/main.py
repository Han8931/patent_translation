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

MAX_CONCURRENCY = 5  # target 4–5 concurrent docs
OUTPUT_DIR = Path("./outputs")

MAX_SECTION_CHARS = 80_000
S3_PREFIX = "patent/260227"

# Robustness knobs
RETRY_ROUNDS = 2          # after the first full pass, retry failed docs this many times
S3_RETRIES_PER_DOC = 2    # retry S3 download inside a single attempt
LLM_RETRIES_PER_DOC = 2   # retry LLM translation inside a single attempt

BASE_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 20.0
JITTER_SEC = 0.3


# ============================================================
# Helpers
# ============================================================

def make_bucket(config: APIConfig) -> Bucket:
    return Bucket(
        name="aip-rag-ai",
        access_key=config.S3_UPLOAD_BUCKET_ACCESS_KEY,
        secret_key=config.S3_UPLOAD_BUCKET_SECRET_KEY,
    )


def backoff_delay(attempt_idx: int) -> float:
    # attempt_idx starts at 0
    delay = min(MAX_BACKOFF_SEC, BASE_BACKOFF_SEC * (2 ** attempt_idx))
    delay += random.uniform(0, JITTER_SEC)
    return delay


@dataclass
class AttemptResult:
    key: str
    ok: bool
    stage: str                 # "ok" | "s3" | "llm"
    out_path: Optional[Path]
    error: Optional[str]
    elapsed_s: float


async def _retry_async(fn, *, retries: int, what: str) -> Tuple[bool, Optional[str]]:
    """
    Retry an awaitable 'fn()' up to 'retries' times (total attempts = retries+1).
    Returns (ok, error_str).
    """
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


async def translate_one(key: str, bucket, llm_config, semaphore) -> AttemptResult:
    async with semaphore:
        key = str(key)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"[START] {key}")  # live progress

        t0 = time.perf_counter()
        out_path: Optional[Path] = None

        # 1) S3 download with retry
        local_path_holder: Dict[str, Path] = {}

        async def _do_s3():
            loop = asyncio.get_running_loop()
            p = await loop.run_in_executor(None, partial(download_s3_file, key, bucket))
            local_path_holder["path"] = p

        ok, err = await _retry_async(_do_s3, retries=S3_RETRIES_PER_DOC, what=f"[S3] {key}")
        if not ok:
            return AttemptResult(
                key=key,
                ok=False,
                stage="s3",
                out_path=None,
                error=err,
                elapsed_s=time.perf_counter() - t0,
            )
        local_path = local_path_holder["path"]

        # 2) LLM translation with retry
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
            return AttemptResult(
                key=key,
                ok=False,
                stage="llm",
                out_path=out_path,
                error=err,
                elapsed_s=time.perf_counter() - t0,
            )

        return AttemptResult(
            key=key,
            ok=True,
            stage="ok",
            out_path=out_path,
            error=None,
            elapsed_s=time.perf_counter() - t0,
        )


async def run_round(keys: List[str], bucket, llm_config, concurrency: int) -> Tuple[List[AttemptResult], List[str]]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(translate_one(k, bucket, llm_config, sem)) for k in keys]
    results: List[AttemptResult] = await asyncio.gather(*tasks)

    failed = [r.key for r in results if not r.ok]
    return results, failed


def print_round_summary(round_idx: int, results: List[AttemptResult]) -> None:
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]

    print(f"\n===== ROUND {round_idx} SUMMARY =====")
    print(f"OK: {len(ok)} | FAILED: {len(fail)}")

    for r in ok:
        print(f"[DONE] {r.key}: {r.elapsed_s:.2f}s -> {r.out_path}")

    for r in fail:
        print(f"[FAIL][{r.stage.upper()}] {r.key}: {r.elapsed_s:.2f}s")
        if r.error:
            print(r.error)


# ============================================================
# Entry
# ============================================================

async def main():
    config = APIConfig()
    llm_config = LLMModel()
    bucket = make_bucket(config)

    keys = list_s3_keys(
        bucket=bucket,
        prefix=S3_PREFIX,
        suffix=".docx",
    )

    if not keys:
        print(f"No .docx found under s3://{bucket.name}/{S3_PREFIX}")
        return

    keys = [str(k) for k in keys]
    keys.sort()

    # Print plan (name + total count)
    print(f"\n[PLAN] Documents to be processed: {len(keys)}")
    for i, k in enumerate(keys, start=1):
        print(f"  [{i:03d}/{len(keys):03d}] {k}")
    print()

    concurrency = min(MAX_CONCURRENCY, max(1, len(keys)))

    # Round 0: try everything once
    results0, failed = await run_round(keys, bucket, llm_config, concurrency)
    print_round_summary(0, results0)

    # Retry rounds: only failed docs
    all_results: List[AttemptResult] = results0[:]
    for r in range(1, RETRY_ROUNDS + 1):
        if not failed:
            break
        print(f"\n>>> Retrying {len(failed)} failed docs (round {r}/{RETRY_ROUNDS})...")
        results_r, failed = await run_round(failed, bucket, llm_config, concurrency)
        all_results.extend(results_r)
        print_round_summary(r, results_r)

    # Final summary (per key final status)
    final_by_key: Dict[str, AttemptResult] = {}
    for res in all_results:
        final_by_key[res.key] = res  # keep latest attempt

    final_ok = [k for k, r in final_by_key.items() if r.ok]
    final_fail = [k for k, r in final_by_key.items() if not r.ok]

    print("\n===== FINAL SUMMARY =====")
    print(f"TOTAL: {len(final_by_key)} | OK: {len(final_ok)} | FAILED: {len(final_fail)}")
    if final_fail:
        print("Still failing:")
        for k in final_fail:
            r = final_by_key[k]
            print(f" - {k} (stage={r.stage}, last_elapsed={r.elapsed_s:.2f}s)")


if __name__ == "__main__":
    asyncio.run(main())
