from hashlib import sha1
from typing import Any


def create_hash(val: str) -> str:
    return sha1(val.encode()).hexdigest()


def add_prefix(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in data.items()}
