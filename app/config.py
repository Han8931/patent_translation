from __future__ import annotations

import os


class APIConfig:
    def __init__(self) -> None:
        self.S3_UPLOAD_BUCKET_ACCESS_KEY = os.getenv("S3_UPLOAD_BUCKET_ACCESS_KEY", "")
        self.S3_UPLOAD_BUCKET_SECRET_KEY = os.getenv("S3_UPLOAD_BUCKET_SECRET_KEY", "")
        self.S3_UPLOAD_BUCKET_NAME = os.getenv("S3_UPLOAD_BUCKET_NAME", "aip-rag-ai")
        self.S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://s3.dataplatform.samsungds.net:9020")


class LLMModel:
    def __init__(self) -> None:
        # Ollama OpenAI-compatible endpoint
        self.GPT_OSS_URL_API = os.getenv("GPT_OSS_URL_API", "http://localhost:11434/v1")
        self.GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "gpt-oss-120b")
        self.GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY", "ollama")
