"""
Tool-calling shim for the upstream DeepSeek web chat endpoint.

The upstream web completion API has no native function-calling capability,
but the underlying model emits tool calls from its training data in two
textual formats:

  1. DSML (native preference): a wrapper of invoke/parameter lines whose
     tag names are built from the module constants below (BAR-based).
  2. JSON block fallback: a single JSON object with name/arguments keys.

Both are parsed here into OpenAI-format structured tool_calls.
"""

import re
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

BAR = "\uff5c"  # fullwidth vertical bar used by the DSML format

TAG_CALLS_OPEN = "<" + BAR * 2 + "DSML" + BAR * 2 + " calls>"
TAG_CALLS_CLOSE = "</" + BAR * 2 + "DSML" + BAR * 2 + " calls>"
TAG_INVOKE_OPEN = "<" + BAR * 2 + "DSML" + BAR * 2 + " invoke "
TAG_INVOKE_CLOSE = "</" + BAR * 2 + "DSML" + BAR * 2 + " invoke>"
TAG_PARAM_OPEN = "<" + BAR * 2 + "DSML" + BAR * 2 + " parameter "
TAG_PARAM_CLOSE = "</" + BAR * 2 + "DSML" + BAR * 2 + " parameter>"

DSML_INVOKE_RE = re.compile(
    re.escape(TAG_INVOKE_OPEN) + r'name="([^"]+)"[^>]*>(.*?)' + re.escape(TAG_INVOKE_CLOSE),
    re.DOTALL,
)
DSML_PARAM_RE = re.compile(
    re.escape(TAG_PARAM_OPEN) + r'name="([^"]+)"[^>]*>(.*?)' + re.escape(TAG_PARAM_CLOSE),
    re.DOTALL,
)

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


def _parse_dsml(content: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for m in DSML_INVOKE_RE.finditer(content):
        name = m.group(1).strip()
        if not name:
            continue
        args: Dict[str, Any] = {}
        for pm in DSML_PARAM_RE.finditer(m.group(2)):
            args[pm.group(1).strip()] = _coerce_value(pm.group(2))
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    return calls


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
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    return calls


def parse_tool_calls(content: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Extract tool calls (DSML or JSON blocks).

    Returns (cleaned_content, tool_calls or None).
    """
    if not content:
        return content, None

    calls = _parse_dsml(content) or _parse_json_blocks(content)
    if not calls:
        return content, None

    cleaned = content
    if TAG_CALLS_OPEN in cleaned:
        cleaned = re.sub(
            re.escape(TAG_CALLS_OPEN) + r".*?" + re.escape(TAG_CALLS_CLOSE),
            "",
            cleaned,
            flags=re.DOTALL,
        ).strip()
    cleaned = JSON_BLOCK_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, calls
