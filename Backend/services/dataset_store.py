"""In-memory dataset registry."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

import pandas as pd

_MAX_DATASETS = 20
_TTL_SECONDS = 6 * 60 * 60


@dataclass
class _Entry:
    df: pd.DataFrame
    stored_at: float


_store: Dict[str, _Entry] = {}
_lock = Lock()


def put(df: pd.DataFrame) -> str:
    """Register a DataFrame in the store and return its dataset_id."""
    dataset_id = uuid.uuid4().hex
    with _lock:
        _evict_expired()
        if len(_store) >= _MAX_DATASETS:
            oldest_id = min(_store, key=lambda key: _store[key].stored_at)
            del _store[oldest_id]
        _store[dataset_id] = _Entry(df=df, stored_at=time.time())
    return dataset_id


def get(dataset_id: str) -> Optional[pd.DataFrame]:
    """Fetch a previously stored DataFrame, or None if missing/expired."""
    with _lock:
        entry = _store.get(dataset_id)
        if entry is None:
            return None
        entry.stored_at = time.time()
        return entry.df


def drop(dataset_id: str) -> None:
    """Remove a dataset from the store, if present."""
    with _lock:
        _store.pop(dataset_id, None)


def _evict_expired() -> None:
    now = time.time()
    expired = [key for key, value in _store.items() if now - value.stored_at > _TTL_SECONDS]
    for key in expired:
        del _store[key]