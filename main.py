import argparse
import asyncio
from pathlib import Path

from app.main import main as run_main
from app.translate import generate_compare_reports_for_docx


def parse_args():
    parser = argparse.ArgumentParser(description="Patent translation batch runner")
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Retry only the files listed in --failed_list",
    )
    parser.add_argument(
        "--failed_list",
        default="outputs/failed_keys.txt",
        help="Path to failed key list file (default: outputs/failed_keys.txt)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate source-vs-output comparison reports during translation",
    )
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="Generate compare reports from existing source and translated DOCX files",
    )
    parser.add_argument(
        "--src_docx",
        default=None,
        help="Path to source DOCX (used with --compare_only)",
    )
    parser.add_argument(
        "--translated_docx",
        default=None,
        help="Path to translated/output DOCX (used with --compare_only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.compare_only:
        if args.retry_failed:
            raise SystemExit("--compare_only cannot be combined with --retry_failed")
        if not args.src_docx or not args.translated_docx:
            raise SystemExit("--compare_only requires both --src_docx and --translated_docx")
        chunk_md_path, line_tsv_path = generate_compare_reports_for_docx(
            src_docx_path=Path(args.src_docx),
            out_docx_path=Path(args.translated_docx),
        )
        print(f"[COMPARE_ONLY] Chunk report: {chunk_md_path}")
        print(f"[COMPARE_ONLY] Line report: {line_tsv_path}")
        raise SystemExit(0)

    asyncio.run(
        run_main(
            retry_failed=args.retry_failed,
            failed_keys_path=Path(args.failed_list),
            compare=args.compare,
        )
    )
