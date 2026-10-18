"""Regression harness for /v1/models catalog metadata.

Run: python3 tests/test_catalog_metadata.py

Without context_length/vision on these entries, 9router-GATER forwards a bare id
list and the GATER storefront/customer API shows "Belum diketahui" for the
DeepSeek models even though the vendor publishes the numbers.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.catalog import catalog_entry, list_models_response  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" -> {detail}"))
    if not ok:
        fails.append(name)


doc = list_models_response(["ds-chat", "ds-reasoner", "deepseek-chat", "deepseek-reasoner"])
check("list shape", doc.get("object") == "list" and len(doc.get("data", [])) == 4, str(list(doc.keys())))
for m in doc["data"]:
    check(f"{m['id']}: konteks terisi",
          isinstance(m.get("context_length"), int) and m["context_length"] > 0 and m["context_window"] == m["context_length"],
          m.get("context_length"))
    check(f"{m['id']}: max output + type", isinstance(m.get("max_completion_tokens"), int) and m.get("type") == "chat", m)
    check(f"{m['id']}: vision boolean + modalities",
          isinstance(m.get("vision"), bool) and "image" in m["architecture"]["input_modalities"], m.get("vision"))
    check(f"{m['id']}: sumber nilai labelled",
          bool(m["capabilities"].get("context_length_source")), m["capabilities"])

check("reasoning dibedakan per model",
      catalog_entry("ds-reasoner")["capabilities"]["reasoning"] is True
      and catalog_entry("ds-chat")["capabilities"]["reasoning"] is False)

# env override: vendor angka bisa disusul tanpa menyentuh kode
for k, v in (("DS_CONTEXT_LENGTH", "128000"), ("DS_VISION", "false"), ("DS_CONTEXT_SOURCE", "manual")):
    os.environ[k] = v
mod = importlib.reload(importlib.import_module("app.api.catalog"))
entry = mod.catalog_entry("ds-chat")
check("override context", entry["context_length"] == 128000, entry["context_length"])
check("override vision", entry["vision"] is False and entry["architecture"]["input_modalities"] == ["text"], entry["architecture"])
check("override source", entry["capabilities"]["context_length_source"] == "manual", entry["capabilities"])
for k in ("DS_CONTEXT_LENGTH", "DS_VISION", "DS_CONTEXT_SOURCE"):
    os.environ.pop(k, None)
importlib.reload(mod)
check("kembali ke default setelah env dibersihkan", mod.catalog_entry("ds-chat")["vision"] is True)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
