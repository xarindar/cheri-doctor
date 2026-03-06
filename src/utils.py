"""Shared utilities for the Metro Manual pipeline."""

import hashlib
import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: dict | list, path: str | Path, indent: int = 2):
    """Save dict/list as JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> dict | list:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: list[dict], path: str | Path):
    """Save list of dicts as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file (reads in chunks for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB chunks
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def config_hash(config: dict) -> str:
    """Deterministic hash of config dict."""
    raw = json.dumps(config, sort_keys=True)
    return f"sha256:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def generate_run_id() -> str:
    """Generate a unique run ID: timestamp + short UUID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def to_dict(obj) -> dict:
    """Convert a dataclass to a dict, handling nested dataclasses."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


def resolve_path(path: str | Path, base: str | Path) -> Path:
    """Resolve a path relative to a base directory."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(base) / p


class Timer:
    """Simple context-manager timer for pipeline stages."""

    def __init__(self, label: str = ""):
        self.label = label
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.label:
            print(f"  [{self.label}] {self.elapsed:.1f}s")
