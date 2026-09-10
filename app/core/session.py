import json
import time
from pathlib import Path
from typing import Optional

from app.config import SESSION_FILE

class SessionManager:
    """
    Maintains a sticky active chat session on DeepSeek web.
    All consecutive requests continue in the same thread until explicitly reset.
    """
    def __init__(self, file_path: Path = SESSION_FILE):
        self.file_path = Path(file_path)
        self.session_id: Optional[str] = None
        self.parent_message_id: Optional[int] = None
        self.load()

    def load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.session_id = data.get("session_id")
                    self.parent_message_id = data.get("parent_message_id")
            except Exception as e:
                print(f"[SessionManager] Warning: failed to load session file: {e}")

    def save(self):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.file_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": self.session_id,
                    "parent_message_id": self.parent_message_id,
                    "updated_at": time.time()
                }, f, indent=2)
            tmp_path.replace(self.file_path)
        except Exception as e:
            print(f"[SessionManager] Warning: failed to save session file: {e}")

    def reset(self, new_id: str):
        self.session_id = new_id
        self.parent_message_id = None
        self.save()

    def update_parent(self, new_parent_id: Optional[int]):
        if new_parent_id is not None:
            self.parent_message_id = new_parent_id
            self.save()

session_manager = SessionManager()
