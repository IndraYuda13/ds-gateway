"""
Tool-calling shim for the upstream DeepSeek web chat endpoint.

The upstream web completion API has no native function-calling capability,
but the underlying model emits tool calls from its training data in two
textual formats:

  1. DSML (native preference): a wrapper of invoke/parameter lines whose
     tag names are built from the module constants below (BAR-based).
     The model emits these bars INCONSISTENTLY (0-2 per side, mixed
     within one block, ASCII pipes, and degraded/malformed closes on
     long generations), so parsing is anchor-based and tolerant.
  2. JSON block fallback: a single JSON object with name/arguments keys.

Both are parsed here into OpenAI-format structured tool_calls.
"""

import re
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

BAR = "\uff5c"  # fullwidth vertical bar used by the DSML format
# bars OPTIONAL per side: emission varies from 0 (degraded) to 2
_BARS_OPT = "[" + BAR + "|]*"

TAG_CALLS_OPEN = "<" + BAR * 2 + "DSML" + BAR * 2 + " calls>"
TAG_CALLS_CLOSE = "</" + BAR * 2 + "DSML" + BAR * 2 + " calls>"
TAG_INVOKE_OPEN = "<" + BAR * 2 + "DSML" + BAR * 2 + " invoke "
TAG_INVOKE_CLOSE = "</" + BAR * 2 + "DSML" + BAR * 2 + " invoke>"
TAG_PARAM_OPEN = "<" + BAR * 2 + "DSML" + BAR * 2 + " parameter "
TAG_PARAM_CLOSE = "</" + BAR * 2 + "DSML" + BAR * 2 + " parameter>"

# Any DSML-ish tag: tolerates missing '<', missing bars, malformed closes.
# [^>\n]* keeps the match on one line so a missing '>' cannot swallow a value.
_ANCHOR_RE = re.compile(
    "</?" + _BARS_OPT + r"\s*DSML\s*" + _BARS_OPT + r"[^>\n]*>?",
    re.DOTALL,
)
_NAME_RE = re.compile(r'\bname\s*=\s*"?([^"\s>]+)"?')

J_OPEN = "<" + "tool_call" + ">"
J_CLOSE = "</" + "tool_call" + ">"
JSON_BLOCK_RE = re.compile(
    re.escape(J_OPEN) + r"\s*(.*?)\s*" + re.escape(J_CLOSE),
    re.DOTALL,
)


def _compact_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize OpenAI tools / legacy functions into compact defs."""
    out: List[Dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        elif t.get("name"):
            out.append({
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            })
    return [t for t in out if t.get("name")]


def build_tool_instructions(
    tools: Optional[List[Dict[str, Any]]],
    functions: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> str:
    """Build the prompt block that teaches the model the tool-call protocol."""
    if isinstance(tool_choice, str) and tool_choice.lower() == "none":
        return ""
    compact = _compact_tools(tools) + _compact_tools(functions)
    if not compact:
        return ""

    example = "\n".join([
        TAG_CALLS_OPEN,
        TAG_INVOKE_OPEN + 'name="get_weather">',
        TAG_PARAM_OPEN + 'name="city" string="true">Jakarta' + TAG_PARAM_CLOSE,
        TAG_INVOKE_CLOSE,
        TAG_CALLS_CLOSE,
    ])
    return "\n".join([
        "You have access to the following tools. Their JSON Schemas are listed below.",
        json.dumps(compact, ensure_ascii=False),
        "",
        "When you need to call one or more tools, reply with ONLY the tool call",
        "block(s) in exactly this format and nothing else:",
        "",
        example,
        "",
        "Rules:",
        "- Emit one invoke block per tool call.",
        "- Emit one parameter line per argument.",
        "- Multiple calls go as multiple invoke blocks inside one wrapper.",
        "- Do not write any other text when emitting tool calls.",
        'After a tool result is given back to you as "[Tool result ...]", continue helping.',
        "If no tool is needed, answer normally in plain text.",
    ])


def _coerce_value(raw: str) -> Any:
    """Keep JSON-ish scalars typed; fall back to string."""
    v = raw.strip()
    try:
        return json.loads(v)
    except Exception:
        return v


def _parse_dsml(content: str, fallback_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Walk DSML anchors tolerantly.

    Handles inconsistent bar counts (0-2 per side), ASCII pipes, mixed
    forms inside one block, degraded/malformed closing tags, and a
    MISSING invoke tag (long generations drop it): parameters seen
    before any invoke are kept as orphans and attached to a synthetic
    call named `fallback_name` when provided.
    """
    anchors = list(_ANCHOR_RE.finditer(content))
    if not anchors:
        return []

    calls: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    orphan_args: Dict[str, Any] = {}
    cur_target: Dict[str, Any] = orphan_args
    pname: Optional[str] = None
    pstart: Optional[int] = None

    def close_param(end: int) -> None:
        nonlocal pname, pstart
        if pname is not None and pstart is not None:
            cur_target[pname] = _coerce_value(content[pstart:end])
        pname = pstart = None

    for m in anchors:
        tag = m.group(0)
        low = tag.lower()
        nm = _NAME_RE.search(tag)
        if nm and "invoke" in low:
            close_param(m.start())
            if cur is not None:
                calls.append(cur)
            cur = {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": nm.group(1).strip(), "arguments": "{}"},
                "_args": {},
            }
            cur_target = cur["_args"]
        elif nm and "parameter" in low:
            close_param(m.start())
            if cur is None:
                # orphan parameter (invoke tag lost) — collect anyway
                cur_target = orphan_args
            pname = nm.group(1).strip()
            pstart = m.end()
        else:
            # wrapper open/close, mangled closes, anything else: boundary
            close_param(m.start())
    close_param(len(content))
    if cur is not None:
        calls.append(cur)

    if not calls and orphan_args and fallback_name:
        # degraded emission: parameters without any invoke tag
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": fallback_name,
                         "arguments": json.dumps(orphan_args, ensure_ascii=False)},
        })

    out: List[Dict[str, Any]] = []
    for c in calls:
        if not c["function"]["name"]:
            continue
        if "_args" in c:
            c["function"]["arguments"] = json.dumps(c.pop("_args"), ensure_ascii=False)
        out.append(c)
    return out


def strip_dsml_region(content: str) -> str:
    """Remove the DSML region (first anchor through last) for display.

    Guarantees raw tool-call tags are never shown to the client even when
    no call could be parsed.
    """
    anchors = list(_ANCHOR_RE.finditer(content))
    if not anchors:
        return content
    cleaned = content[:anchors[0].start()] + content[anchors[-1].end():]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def extract_param_names(content: str) -> List[str]:
    """Parameter names declared by DSML parameter anchors."""
    names = []
    for m in _ANCHOR_RE.finditer(content):
        nm = _NAME_RE.search(m.group(0))
        if nm and "parameter" in m.group(0).lower():
            names.append(nm.group(1).strip())
    return names


def infer_fallback_name(
    content: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Best guess for the tool of a degraded (invoke-less) emission.

    1. unique declared tool whose schema properties cover the emitted
       parameter names;
    2. otherwise the last tool used in the conversation history (agent
       loops usually repeat the same tool).
    """
    params = set(extract_param_names(content))
    if params:
        cands = []
        for t in _compact_tools(tools) + _compact_tools(functions):
            props = set(((t.get("parameters") or {}).get("properties") or {}).keys())
            if props and props >= params:
                cands.append(t["name"])
        if len(cands) == 1:
            return cands[0]
    for m in reversed(messages or []):
        if isinstance(m, dict) and (m.get("role") or "").lower() == "tool" and m.get("name"):
            return m["name"]
    for m in reversed(messages or []):
        if isinstance(m, dict):
            for tc in reversed(m.get("tool_calls") or []):
                fn = (tc or {}).get("function") or {}
                if fn.get("name"):
                    return fn["name"]
    return None


def _parse_json_blocks(content: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for m in JSON_BLOCK_RE.finditer(content):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name:
            inner = obj.get("tool_call") or obj.get("function") or {}
            if isinstance(inner, dict) and inner.get("name"):
                name = inner["name"]
                if "arguments" in inner:
                    obj = inner
        if not name:
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": args},
        })

    if not calls:
        # Bare JSON object as the whole content: {"name": ..., "arguments": ...}
        # or {"tool_call"/"function": {...}} (no fence).
        try:
            obj = json.loads(content.strip())
        except Exception:
            obj = None
        if isinstance(obj, dict):
            inner = obj.get("tool_call") or obj.get("function")
            if isinstance(inner, dict) and inner.get("name"):
                obj = inner
            name = obj.get("name")
            args = obj.get("arguments")
            if name and isinstance(args, (dict, str)) and (
                "tool_call" in content or "function" in content
                or set(obj.keys()) <= {"name", "arguments", "id", "type"}
            ):
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                calls.append({
                    "id": "call_" + uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })
    return calls


def parse_tool_calls(
    content: str,
    fallback_name: Optional[str] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Extract tool calls (DSML or JSON blocks).

    Returns (cleaned_content, tool_calls or None). When DSML anchors are
    present but no call could be recovered, the region is still stripped
    so raw tags never reach the client.
    """
    if not content:
        return content, None

    calls = _parse_dsml(content, fallback_name) or _parse_json_blocks(content)
    has_dsml = bool(_ANCHOR_RE.search(content))
    if not calls:
        if has_dsml:
            return strip_dsml_region(content), None
        return content, None

    cleaned = content
    if has_dsml:
        cleaned = strip_dsml_region(content)
    cleaned = JSON_BLOCK_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, calls
