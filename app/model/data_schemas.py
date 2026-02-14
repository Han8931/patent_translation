from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SecretValue:
    value: str

    def get_secret_value(self) -> str:
        return self.value


@dataclass(frozen=True)
class Bucket:
    name: str
    access_key: SecretValue
    secret_key: SecretValue

