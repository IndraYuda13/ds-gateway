import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SESSION_FILE = DATA_DIR / "session_state.json"
SOLVER_PATH = BASE_DIR / "app" / "solver.js"
WASM_PATH = BASE_DIR / "app" / "sha3_wasm_bg.wasm"

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8550"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

# DeepSeek Upstream Configuration
DEFAULT_TOKEN = os.getenv("DS_TOKEN", os.getenv("DEEPSEEK_TOKEN", "qosOTpuhzASFMfNcNw1rTJhJXaV6CK9icR3C94feTk1ZY7IPBpFI/Ogk9pOp69ek"))
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")  # Optional API key for proxy access

USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
CLIENT_VERSION = os.getenv("CLIENT_VERSION", "2.4.0")
CLIENT_LOCALE = "en_US"
CLIENT_BUNDLE_ID = "com.deepseek.chat"

# Upstream Endpoints
DS_BASE_URL = "https://chat.deepseek.com"
DS_HIF_LEIM_URL = "https://hif-leim.deepseek.com/query"
