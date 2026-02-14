import os
import boto3

from docx import Document
from pathlib import Path

S3_DRIVE = "http://s3.dataplatform.samsungds.net:9020"
FILE_DOWNLOAD_DIR = Path("./uploads/")


def extract_docx_text(docx_path: str | Path) -> str:
    docx_path = Path(docx_path)
    doc = Document(str(docx_path))

    parts: list[str] = []

    # Paragraphs
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)

    # Tables (optional but usually important)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            line = "\t".join([c for c in cells if c])
            if line:
                parts.append(line)

    return "\n".join(parts)



def load_filename_list(file_path: Path) -> list[str]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        # remove surrounding quotes if any
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()

        # if the line is a path, keep only the filename
        out.append(Path(s).name)

    return out

def list_local_docx_files(documents_dir: str | Path) -> list[str]:
    base = Path(documents_dir)
    if not base.exists():
        return []
    return sorted(p.name for p in base.glob("*.docx") if p.is_file())


def resolve_local_file(filepath: str | Path, documents_dir: str | Path = "./documents") -> Path:
    p = Path(filepath)
    if p.is_absolute() and p.exists():
        return p
    candidate = Path(documents_dir) / p.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Local document not found: {candidate}")


def download_s3_file(filepath: str | Path, bucket: "Bucket") -> Path:
    src = Path(filepath)
    client = boto3.client(
        "s3",
        aws_access_key_id=str(bucket.access_key.get_secret_value()),
        aws_secret_access_key=str(bucket.secret_key.get_secret_value()),
        endpoint_url=S3_DRIVE,
    )

    filename = src.name
    path = FILE_DOWNLOAD_DIR / filename

    # Ensure directory exists
    os.makedirs(FILE_DOWNLOAD_DIR, exist_ok=True)

    client.download_file(
            Bucket=bucket.name, 
            Key=str(src), 
            Filename=str(path)
        )

    return Path(path)
