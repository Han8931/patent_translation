import argparse
import asyncio
from pathlib import Path

from app.main import main as run_main


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_main(
            retry_failed=args.retry_failed,
            failed_keys_path=Path(args.failed_list),
        )
    )
