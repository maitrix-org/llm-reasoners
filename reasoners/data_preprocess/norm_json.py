#!/usr/bin/env python3
"""
Normalise every .json file in a directory so that each record contains
only the allowed keys; anything else is moved under `extra_info` as a
compressed Base64‐encoded JSON string split into 10 parts.

If a record already has `extra_info` or `reward_model` from a previous run,
we first de-serialise, merge with any new stray keys, then re-serialise,
compress, and Base64-encode.

Canonical keys:
    ability
    apply_chat_template
    data_source
    extra_info         (dict → JSON → gzip → Base64 split into parts)
    prompt
    qwen2.5_7b_pass_rate
    qwen3_30b_pass_rate
    reward_model       (dict → JSON → gzip → Base64)
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import json
import gzip
import base64
import sys
import argparse
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KEEP_KEYS = {
    "ability",
    "apply_chat_template",
    "data_source",
    "extra_info",            # handled specially
    "prompt",
    "qwen2.5_7b_pass_rate",
    "qwen3_30b_pass_rate",
    "reward_model",
}

Json = Union[Dict[str, Any], List[Any]]

# ---------------------------------------------------------------------------
# Serialisation & Compression helpers
# ---------------------------------------------------------------------------

def _serialise_extra(extra: Dict[str, Any]) -> str:
    """Return a compact JSON string for `extra`."""
    return json.dumps(extra, ensure_ascii=False, separators=(',', ':'))

def _deserialise_extra(extra_str: str) -> Dict[str, Any]:
    """Load JSON string if valid, else return empty dict."""
    try:
        val = json.loads(extra_str)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}

def _compress_str(s: str) -> str:
    """Gzip-compress then Base64-encode a JSON string."""
    gz = gzip.compress(s.encode('utf-8'))
    return base64.b64encode(gz).decode('ascii')

def _decompress_str(b64: str) -> str:
    """Base64-decode then gzip-decompress to original JSON string."""
    gz = base64.b64decode(b64.encode('ascii'))
    return gzip.decompress(gz).decode('utf-8')

# ---------------------------------------------------------------------------
# JSON/NDJSON loader
# ---------------------------------------------------------------------------

def load_json_or_ndjson(path: Path) -> Json:
    text = path.read_text(encoding='utf-8').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        ln = line.strip()
        if ln:
            records.append(json.loads(ln))
    return records

# ---------------------------------------------------------------------------
# Record normalisation (with 10-way split for extra_info)
# ---------------------------------------------------------------------------

def normalise_record(record: Dict[str, Any]) -> Dict[str, Any]:
    # 1) extract existing extra_info dict (if any)
    extra_raw = record.get('extra_info', {})
    if isinstance(extra_raw, str):
        nested = json.loads(extra_raw).get('compressed', '')
        extra = _deserialise_extra(_decompress_str(nested))
    elif isinstance(extra_raw, dict) and 'compressed' in extra_raw:
        extra = _deserialise_extra(_decompress_str(extra_raw['compressed']))
    elif isinstance(extra_raw, dict):
        extra = extra_raw.copy()
    else:
        extra = {}

    # 2) extract existing reward_model dict (if any)
    rm_raw = record.get('reward_model', {})
    if isinstance(rm_raw, str):
        reward_model = _deserialise_extra(rm_raw)
    elif isinstance(rm_raw, dict) and 'compressed' in rm_raw:
        reward_model = _deserialise_extra(_decompress_str(rm_raw['compressed']))
    elif isinstance(rm_raw, dict):
        reward_model = rm_raw.copy()
    else:
        reward_model = {}

    # 3) copy canonical keys, gather others into extra
    out: Dict[str, Any] = {}
    for k, v in record.items():
        if k in KEEP_KEYS and k not in ('extra_info', 'reward_model'):
            out[k] = v
        elif k not in KEEP_KEYS:
            extra[k] = v

    # 4) serialise + compress + split extra_info into 10 parts
    if extra:
        ser = _serialise_extra(extra)
        comp = _compress_str(ser)
        total_len = len(comp)
        chunk_size = total_len // 10
        parts: Dict[str, str] = {}
        for i in range(10):
            start = i * chunk_size
            end = start + chunk_size if i < 9 else total_len
            parts[f'part_{i+1}'] = comp[start:end]
        out['extra_info'] = parts

    # 5) serialise + compress reward_model if present
    if reward_model:
        ser_rm = _serialise_extra(reward_model)
        comp_rm = _compress_str(ser_rm)
        out['reward_model'] = {"ground_truth": {"compressed": comp_rm}}

    return out

# ---------------------------------------------------------------------------
# Directory processing (parallel)
# ---------------------------------------------------------------------------

from multiprocessing import Pool, cpu_count

def _process_file(args):
    """
    Worker to normalise one JSON file and write out result.
    args: tuple(src_path_str, dst_dir_str)
    """
    path_str, dst_dir_str = args
    src_path = Path(path_str)
    dst_dir = Path(dst_dir_str)

    try:
        data = load_json_or_ndjson(src_path)
        if isinstance(data, list):
            out_data = [normalise_record(obj) for obj in data]
        else:
            out_data = normalise_record(data)

        out_path = dst_dir / src_path.name
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        return f"✔ {src_path.name} → {out_path.name}"
    except Exception as e:
        return f"✖ {src_path.name} failed: {e}"

def process_dir_parallel(src_dir: Path, dst_dir: Path, workers: int = None) -> None:
    """
    Process all *.json files in src_dir in parallel, writing to dst_dir.
    workers: number of parallel worker processes. Defaults to number of CPU cores minus one.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = max(1, cpu_count() - 1)

    args_list = [(str(path), str(dst_dir)) for path in src_dir.glob('*.json')]
    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_process_file, args_list):
            print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Normalize JSON files by moving non-canonical keys to extra_info")
    parser.add_argument("src", type=str, help="Source directory containing JSON files")
    parser.add_argument("dst", type=str, help="Destination directory for normalized JSON files") 
    parser.add_argument("-w", "--workers", type=int, help="Number of worker processes (default: CPU count - 1)", default=None)
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.is_dir():
        print(f"Error: {src} is not a directory")
        sys.exit(1)

    process_dir_parallel(src, dst, workers=args.workers)
