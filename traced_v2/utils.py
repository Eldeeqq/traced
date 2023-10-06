from hashlib import sha1
from typing import Any


def create_hash(val: str, max_size:int=12) -> str:
    return sha1(val.encode()).hexdigest()[:max_size]


def add_prefix(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in data.items()}


def remove_duplicates(data: list[Any]) -> list[Any]:
    curr = data[0]
    resp = [curr]
    for val in data[1:]:
        if val != curr:
            resp.append(val)
            curr = val
    return resp
