"""Regression harness for degraded DeepSeek DSML emissions.

The upstream web model sometimes malforms its native tool-call tags:
inconsistent bar counts (0-2 per side, fullwidth U+FF5C or ASCII pipe),
mangled closing tags, blocks split across the thinking/response stream,
or — on long payloads — an entirely LOST invoke tag (parameters only).

Run: python3 tests/test_degraded_parser.py
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.tools import parse_tool_calls, infer_fallback_name  # noqa: E402

FW = "\uff5c"

def t(s: str) -> str:  # two-bar open helper
    return "<" + FW * 2 + "DSML" + FW * 2 + " " + s + ">"

def c(s: str) -> str:  # close helper
    return "</" + FW * 2 + "DSML" + FW * 2 + " " + s + ">"

# Degraded shape observed in production: wrapper + parameters, NO invoke,
# mixed/mangled closes.
SYNTH_LEAK = (
    t("calls") + "\n"
    + t('parameter name="code" string="true"') + "# Sweep aktivitas\nprint('ok')\n"
    + c("parameter") + "\n"
    + t('parameter name="timeout_s" string="false"') + "240"
    + "</" + FW + "DSML" + FW + ">\n"
    + "</DSML" + FW * 2 + " invoke>\n"
    + "</DSML" + FW * 2 + " calls>"
)

fails = []

def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + str(extra)) if extra and not cond else ""))
    if not cond:
        fails.append(name)

# 1. orphan parameters recovered via history/schema fallback
cleaned, calls = parse_tool_calls(SYNTH_LEAK, fallback_name="browser_exec")
check("degraded block recovered via fallback", calls and len(calls) == 1, calls)
if calls:
    fn = calls[0]["function"]
    args = json.loads(fn["arguments"])
    check("recovered tool name", fn["name"] == "browser_exec", fn)
    check("recovered code param", args.get("code", "").startswith("# Sweep aktivitas"), list(args))
    check("recovered timeout_s=240", args.get("timeout_s") == 240, args.get("timeout_s"))
check("recovered content stripped", "DSML" not in cleaned, cleaned[:60])

# 2. without a fallback the region must still never leak raw tags
cleaned2, calls2 = parse_tool_calls(SYNTH_LEAK)
check("no fallback -> no raw leak", "DSML" not in cleaned2 and not calls2, cleaned2[:80])

# 3. fallback inference: unique declared-schema match
tools = [{"type": "function", "function": {"name": "browser_exec",
           "parameters": {"properties": {"code": {}, "timeout_s": {}}}}}]
check("schema inference", infer_fallback_name(SYNTH_LEAK, tools=tools, messages=[]) == "browser_exec")

# 4. fallback inference: last tool used in history wins
msgs = [{"role": "tool", "name": "browser_exec", "content": "ok"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "terminal"}}]}]
check("history inference", infer_fallback_name(SYNTH_LEAK, messages=msgs) == "browser_exec")
check("history inference none on empty",
      infer_fallback_name(SYNTH_LEAK, messages=[{"role": "user", "content": "hi"}]) is None)

# 5. block split across THINK -> RESPONSE boundary
think = "reasoning text\n" + t("calls") + "\n" + t('invoke name="terminal"')
resp = t('parameter name="command" string="true"') + "ls -la" + c("parameter") + "\n" + c("invoke") + "\n" + c("calls")
_, calls5 = parse_tool_calls(think + "\n" + resp)
check("split block recovers invoke+params", calls5 and calls5[0]["function"]["name"] == "terminal"
      and json.loads(calls5[0]["function"]["arguments"]) == {"command": "ls -la"}, calls5)

# 6. standard two-bar block
std = (t("calls") + "\n" + t('invoke name="get_weather"') + "\n"
       + t('parameter name="city" string="true"') + "Jakarta" + c("parameter") + "\n"
       + c("invoke") + "\n" + c("calls"))
_, calls6 = parse_tool_calls(std)
check("std invoke+param", calls6 and json.loads(calls6[0]["function"]["arguments"]) == {"city": "Jakarta"})

# 7. ascii pipes
ap = ('<|DSML| calls>\n<|DSML| invoke name="t1">\n<|DSML| parameter name="a" string="true">1</|DSML| parameter>\n'
      '</|DSML| invoke>\n</|DSML| calls>')
_, calls7 = parse_tool_calls(ap)
check("ascii pipes", calls7 and json.loads(calls7[0]["function"]["arguments"]) == {"a": 1}, calls7)

# 8. single-bar user paste with two params
up = ('<' + FW + 'DSML' + FW + ' calls>\n<' + FW + 'DSML' + FW + ' invoke name="terminal">\n'
      '<' + FW + 'DSML' + FW + ' parameter name="command" string="true">echo hi</' + FW + 'DSML' + FW + ' parameter>\n'
      '<' + FW + 'DSML' + FW + ' parameter name="brief" string="true">Cek disk</' + FW + 'DSML' + FW + ' parameter>\n'
      '</' + FW + 'DSML' + FW + ' invoke>\n</' + FW + 'DSML' + FW + ' calls>')
_, calls8 = parse_tool_calls(up)
check("1-bar two params", calls8 and calls8[0]["function"]["name"] == "terminal"
      and json.loads(calls8[0]["function"]["arguments"])["brief"] == "Cek disk", calls8)

# 9. mangled closes with invoke present
dg = (t("calls") + "\n" + t('invoke name="x"') + "\n"
      + t('parameter name="a" string="true"') + "v1" + c("parameter") + "\n"
      + t('parameter name="b" string="false"') + "7</" + FW + "DSML" + FW + ">\n"
      + "</DSML" + FW * 2 + " invoke>\n" + "</DSML" + FW * 2 + " calls>")
_, calls9 = parse_tool_calls(dg)
check("mangled closes", calls9 and json.loads(calls9[0]["function"]["arguments"]) == {"a": "v1", "b": 7}, calls9)

# 10. plain text untouched
c10, calls10 = parse_tool_calls("Halo bre, ada yang bisa dibantu?")
check("plain text passthrough", c10 == "Halo bre, ada yang bisa dibantu?" and not calls10)

# 11. surrounding prose preserved when a real call exists
pre = "Oke gw cek dulu ya.\n\n" + std + "\n\nTunggu sebentar."
c11, calls11 = parse_tool_calls(pre)
check("surrounding text kept", calls11 and "Oke gw cek" in c11 and "Tunggu" in c11 and "DSML" not in c11, c11)

# 12. fenced and wrapped JSON fallbacks (built without literal tag words
#     so the file survives serializers that treat them as markup)
TO, TC = "<" + "tool_" + "call" + ">", "</" + "tool_" + "call" + ">"
_, calls12 = parse_tool_calls(TO + json.dumps({"name": "gw", "arguments": {"c": 1}}) + TC)
check("json fenced", calls12 and calls12[0]["function"]["name"] == "gw", calls12)
_, calls13 = parse_tool_calls('{"tool_call": {"name": "gw", "arguments": {"c": 1}}}')
check("json wrapped object", calls13 and calls13[0]["function"]["name"] == "gw", calls13)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
