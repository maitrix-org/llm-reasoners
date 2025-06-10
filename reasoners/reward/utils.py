import json
import base64
import gzip
from typing import Dict, Any

def _deserialise_extra(extra_str: str) -> Dict[str, Any]:
    """Opposite of :func:`_serialise_extra`.  Returns an empty dict if input is
    not valid JSON (so we fail gracefully)."""
    try:
        val = json.loads(extra_str)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}
    
def _decompress_str(b64) -> str:
    """Base64 解码 + gzip 解压，支持 dict 十片合并后的处理"""
    if isinstance(b64, str):
        # 单块字符串，直接解码
        gz = base64.b64decode(b64.encode('ascii'))
        return gzip.decompress(gz).decode('utf-8')

    elif isinstance(b64, dict):
        # 1) 按 part_1, part_2, … part_N 顺序取片段
        total = []
        for i in range(1, len(b64) + 1):
            key = f'part_{i}'
            total.append(b64[key])
        full_b64 = ''.join(total)

        # 2) 一次性 Base64 解码
        gz_bytes = base64.b64decode(full_b64.encode('ascii'))
        # 3) gzip 解压
        return gzip.decompress(gz_bytes).decode('utf-8')

    else:
        raise ValueError(f"Unsupported type for decompress: {type(b64)}")

