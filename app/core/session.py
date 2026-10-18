import os
import time
import json
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple

from app.config import DATA_DIR

POOL_FILE = DATA_DIR / "session_pool.json"
MAX_POOL_SIZE = int(os.getenv("MAX_SESSIONS", "10"))

class SessionEntry:
    def __init__(
        self,
        session_id: str,
        conv_id: str,
        parent_message_id: Optional[int] = None,
        created_at: Optional[float] = None,
        last_active: Optional[float] = None,
        turn_count: int = 1,
        last_chain_hash: str = "",
        last_depth: int = 0
    ):
        self.session_id = session_id
        self.conv_id = conv_id
        self.parent_message_id = parent_message_id
        self.created_at = created_at or time.time()
        self.last_active = last_active or time.time()
        self.turn_count = turn_count
        # Prefix-chain hash of the most recent full `messages` array this
        # conversation was seen with. Continuation is only granted when a
        # client request's message history *equals or extends exactly* that
        # recorded history — never from a lone first-message fingerprint
        # (which let a fresh /new session inherit an old conversation's
        # upstream context when the opening line repeated).
        self.last_chain_hash = last_chain_hash
        # Number of client messages already delivered into this upstream
        # conversation; the next continued turn only sends the delta.
        self.last_depth = last_depth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conv_id": self.conv_id,
            "parent_message_id": self.parent_message_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "turn_count": self.turn_count,
            "last_chain_hash": self.last_chain_hash,
            "last_depth": self.last_depth
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionEntry":
        return cls(
            session_id=d["session_id"],
            conv_id=d["conv_id"],
            parent_message_id=d.get("parent_message_id"),
            created_at=d.get("created_at"),
            last_active=d.get("last_active"),
            turn_count=d.get("turn_count", 1),
            last_chain_hash=d.get("last_chain_hash", ""),
            last_depth=d.get("last_depth", 0)
        )


class SmartSessionPool:
    """
    Intelligent LRU Session Pool for Upstream Conversational Accounts.
    
    Features:
    1. Multi-tenant conversation isolation (no cross-talk between different scripts/users).
    2. Automatic continuity for multi-turn chats via Conversation Fingerprinting.
    3. Strict bounded size (MAX_POOL_SIZE) with automatic upstream deletion of LRU sessions
       so the upstream web sidebar never fills with spam.
    """
    def __init__(self, file_path: Path = POOL_FILE, max_size: int = MAX_POOL_SIZE):
        self.file_path = Path(file_path)
        self.max_size = max(1, max_size)
        self.pool: Dict[str, SessionEntry] = {}  # conv_id -> SessionEntry
        self.load()

    def load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.pool[k] = SessionEntry.from_dict(v)
            except Exception as e:
                print(f"[SessionPool] Warning: Failed to load pool file: {e}")

    def save(self):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.file_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self.pool.items()}, f, indent=2)
            tmp_path.replace(self.file_path)
        except Exception as e:
            print(f"[SessionPool] Warning: Failed to save pool file: {e}")

    def compute_conv_id(
        self,
        messages: List[Dict[str, Any]],
        user: Optional[str] = None,
        explicit_session_id: Optional[str] = None
    ) -> str:
        """
        Derives an immutable conversation identifier:
        1. Explicit session ID passed in payload/header.
        2. Standard OpenAI `user` parameter.
        3. Fingerprint of the root user message (remains constant across turns).
        """
        if explicit_session_id:
            return f"custom_{explicit_session_id.strip()}"
        if user and str(user).strip() not in ("", "none", "null"):
            return f"user_{str(user).strip()}"

        # Fingerprint from the first user prompt in the messages array
        first_user_content = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str) and c.strip():
                    first_user_content = c.strip()
                    break

        if first_user_content:
            h = hashlib.sha256(first_user_content.encode("utf-8")).hexdigest()[:16]
            return f"conv_{h}"

        return f"conv_anon_{int(time.time())}"

    # ------------------------------------------------------------------
    # Prefix-chain conversation binding (anti cross-session contamination)
    # ------------------------------------------------------------------
    RETRY_WINDOW = 60          # seconds a FULL history match may continue
    FULL_TRUST_DEPTH = 6       # long replays are real retries, always continue

    @staticmethod
    def _hash_msg(m: Dict[str, Any]) -> bytes:
        blob = json.dumps(m, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).digest()

    def chain_hashes(self, messages: List[Dict[str, Any]]) -> Dict[int, str]:
        """Rolling hash over message prefixes: depth -> chain hash.

        chain[d] commits to the exact ordered content of messages[:d],
        so equality means byte-identical history (collision-resistant).
        """
        cur = b"\x00ds-chain"
        out: Dict[int, str] = {}
        for i, m in enumerate(messages, 1):
            cur = hashlib.sha256(cur + self._hash_msg(m)).digest()
            out[i] = cur.hex()
        return out

    def find_continuation(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """conv_id of a pooled conversation this request genuinely extends.

        A conversation continues only when its last-seen full message list
        is an exact prefix of the incoming list. An identical full list is
        accepted as a retry when the conversation was active within
        RETRY_WINDOW, or when the history is deep (long agent transcripts
        repeating by chance is not realistic). A shallow identical opener
        (e.g. a fresh /new starting with "hai") does NOT continue — it
        gets its own conversation instead.
        """
        n = len(messages)
        if n == 0:
            return None
        chain = self.chain_hashes(messages)
        by_hash = {v: d for d, v in chain.items()}
        now = time.time()
        best_depth, best_key = 0, None
        for k, e in self.pool.items():
            lh = e.last_chain_hash
            if not lh:
                continue
            depth = by_hash.get(lh)
            if depth is None:
                continue
            if depth < n:
                ok = True                       # exact history extended
            elif n > self.FULL_TRUST_DEPTH:
                ok = True                       # deep identical replay = retry
            else:
                ok = (now - e.last_active) <= self.RETRY_WINDOW  # fresh retry
            if not ok:
                continue
            if (depth > best_depth) or (
                depth == best_depth and best_key is not None
                and e.turn_count > self.pool[best_key].turn_count
            ):
                best_depth, best_key = depth, k
        return best_key

    def route(
        self,
        messages: List[Dict[str, Any]],
        user: Optional[str] = None,
        explicit_session_id: Optional[str] = None,
        force_new: bool = False
    ) -> str:
        return self.route_full(messages, user, explicit_session_id, force_new)[0]

    def route_full(
        self,
        messages: List[Dict[str, Any]],
        user: Optional[str] = None,
        explicit_session_id: Optional[str] = None,
        force_new: bool = False
    ) -> Tuple[str, bool]:
        """(conv_id, continued) for a request.

        continued=True means this request chains onto an existing pooled
        conversation (real next turn or a genuine retry). continued=False
        means a brand-new conversation instance: the client is either at
        its opening turn or has diverging history (edited transcript,
        context compaction) — the caller should hand the whole replay to
        the fresh upstream conversation instead of only the last message.
        """
        if explicit_session_id:
            cid = f"custom_{explicit_session_id.strip()}"
            return cid, cid in self.pool
        if user and str(user).strip() not in ("", "none", "null"):
            cid = f"user_{str(user).strip()}"
            return cid, cid in self.pool
        if not force_new:
            cont = self.find_continuation(messages)
            if cont:
                return cont, True
        chain = self.chain_hashes(messages)
        if chain:
            # unique key per conversation INSTANCE: identical openers from
            # fresh client sessions must NOT map onto a pooled entry by id;
            # continuation happens only through find_continuation's chain
            # matching, never by content-equal ids.
            return f"conv_{chain[len(messages)][:12]}_{uuid.uuid4().hex[:4]}", False
        return f"conv_anon_{int(time.time())}", False

    def note(self, conv_id: str, messages: List[Dict[str, Any]]) -> None:
        """Record that this conversation has now been seen with `messages`."""
        e = self.pool.get(conv_id)
        if e is None or not messages:
            return
        e.last_chain_hash = self.chain_hashes(messages).get(len(messages), "")
        e.last_depth = len(messages)
        e.last_active = time.time()
        self.save()

    def depth_of(self, conv_id: str) -> int:
        e = self.pool.get(conv_id)
        return e.last_depth if e else 0

    def acquire(
        self,
        conv_id: str,
        create_fn: Callable[[], str],
        delete_fn: Callable[[str], bool],
        force_new: bool = False
    ) -> Tuple[str, Optional[int]]:
        """
        Retrieves an active session for the conversation or initializes one.
        Evicts and deletes the oldest LRU session on upstream if pool is full.
        """
        now = time.time()

        # If forced new or already exists
        if not force_new and conv_id in self.pool:
            entry = self.pool[conv_id]
            entry.last_active = now
            entry.turn_count += 1
            self.save()
            return entry.session_id, entry.parent_message_id

        if force_new and conv_id in self.pool:
            # replacing a bound conversation: delete its upstream session
            # so no orphaned conversation remains on the account
            old = self.pool.pop(conv_id)
            try:
                delete_fn(old.session_id)
            except Exception:
                pass

        # If pool is at capacity, evict least recently used (LRU)
        if len(self.pool) >= self.max_size:
            lru_key = min(self.pool.keys(), key=lambda k: self.pool[k].last_active)
            old_entry = self.pool.pop(lru_key)
            print(f"[SessionPool] Pool full ({len(self.pool)+1}/{self.max_size}). Evicting LRU {lru_key} ({old_entry.session_id})")
            try:
                delete_fn(old_entry.session_id)
            except Exception as e:
                print(f"[SessionPool] Upstream delete error for {old_entry.session_id}: {e}")

        # Create fresh upstream session
        new_sid = create_fn()
        self.pool[conv_id] = SessionEntry(
            session_id=new_sid,
            conv_id=conv_id,
            parent_message_id=None,
            created_at=now,
            last_active=now,
            turn_count=1
        )
        self.save()
        return new_sid, None

    def update_parent(self, conv_id: str, new_parent_id: Optional[int]):
        if conv_id in self.pool and new_parent_id is not None:
            self.pool[conv_id].parent_message_id = new_parent_id
            self.pool[conv_id].last_active = time.time()
            self.save()

    def reset_conv(self, conv_id: str, delete_fn: Callable[[str], bool]):
        if conv_id in self.pool:
            old = self.pool.pop(conv_id)
            self.save()
            try:
                delete_fn(old.session_id)
            except:
                pass


smart_pool = SmartSessionPool()
