"""Token estimation for the OpenAI-compatible usage field.

DeepSeek's web API (chat.deepseek.com) does NOT report per-request token
usage in its SSE stream. To keep downstream consumers (9router dashboards,
Hermes billing/context math) from seeing nonsense, the gateway estimates
usage itself:

- if `tiktoken` happens to be importable, its encoding is used (better
  accuracy);
- otherwise a dependency-free character-class heuristic runs:
  CJK/full-width ~= 1 token per char, latin/other ~= 1 token per 4 chars,
  +4 tokens message framing overhead applied by the caller.

Estimates are approximate by design; they are labeled `estimated: true`
in the usage payload so consumers can tell them apart from billed counts.
"""
from typing import Any, Dict, Optional

try:  # optional, used automatically when present
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None

_CJK_RANGES = (
    (0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF), (0xFF00, 0xFFEF),  # incl. fullwidth pipes U+FF5C
    (0x20000, 0x2A6DF),
)


def _is_cjk_or_fw(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def estimate_text(text: Optional[str]) -> int:
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text, disallowed_special=()))
        except Exception:
            pass
    cjk = sum(1 for ch in text if _is_cjk_or_fw(ch))
    rest = len(text) - cjk
    return cjk + (rest + 3) // 4
