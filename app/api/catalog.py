"""Metadata katalog untuk /v1/models.

Nilai dasar diambil dari halaman resmi DeepSeek "Models & Pricing"
(api-docs.deepseek.com/quick_start/pricing, diakses 18 Okt 2026):
CONTEXT LENGTH 1M · MAX OUTPUT 384K · Vision didukung pada deepseek-flash.

Semua bisa dioverride lewat env tanpa menyentuh kode, karena angka ini pernyataan
vendor: API web DeepSeek tidak mengembalikan metadata konteks, jadi gateway tidak
mengukurnya sendiri. Konsumen (9router -> GATER) bisa melihat asal nilainya lewat
capabilities.context_length_source.
"""
import os
from typing import Any, Dict, List

CATALOG_CONTEXT_LENGTH = int(os.getenv("DS_CONTEXT_LENGTH", "1000000"))
CATALOG_MAX_OUTPUT = int(os.getenv("DS_MAX_OUTPUT", "384000"))
CATALOG_VISION = os.getenv("DS_VISION", "true").strip().lower() in ("1", "true", "yes")
CATALOG_SOURCE = os.getenv("DS_CONTEXT_SOURCE", "deepseek-docs-models-pricing")


def catalog_entry(model_id: str, owned_by: str = "deepseek") -> Dict[str, Any]:
    """Satu entri /v1/models lengkap: konsumen tidak perlu menebak konteks atau
    kemampuan input. Field mengikuti konvensi OpenAI plus ekstensi
    `context_length`/`context_window`/`vision` yang sudah dipakai katalog jerouter."""
    modalities = ["text", "image"] if CATALOG_VISION else ["text"]
    return {
        "id": model_id,
        "object": "model",
        "created": 1700000000,
        "owned_by": owned_by,
        "permission": [],
        "root": model_id,
        "parent": None,
        "type": "chat",
        "context_length": CATALOG_CONTEXT_LENGTH,
        "context_window": CATALOG_CONTEXT_LENGTH,
        "max_completion_tokens": CATALOG_MAX_OUTPUT,
        "vision": CATALOG_VISION,
        "architecture": {
            "modality": "text+image->text" if CATALOG_VISION else "text->text",
            "input_modalities": modalities,
            "output_modalities": ["text"],
        },
        "capabilities": {
            "vision": CATALOG_VISION,
            "tools": True,
            "json_output": True,
            "reasoning": model_id.endswith("reasoner"),
            "context_length_source": CATALOG_SOURCE,
        },
    }


def list_models_response(model_ids: List[str]) -> Dict[str, Any]:
    return {"object": "list", "data": [catalog_entry(m) for m in model_ids]}
