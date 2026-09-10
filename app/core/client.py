import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Generator, Tuple

from app.config import (
    DS_BASE_URL,
    DS_HIF_LEIM_URL,
    USER_AGENT,
    CLIENT_VERSION,
    CLIENT_LOCALE,
    CLIENT_BUNDLE_ID
)
from app.core.pow import PoWSolver
from app.core.session import session_manager

class DeepSeekUpstreamClient:
    def __init__(self, token: str):
        self.token = token.strip()

    def _headers(self, json_content: bool = True) -> Dict[str, str]:
        h = {
            "authorization": f"Bearer {self.token}",
            "user-agent": USER_AGENT,
            "x-client-platform": "web",
            "x-client-version": CLIENT_VERSION,
            "x-client-locale": CLIENT_LOCALE,
            "x-client-bundle-id": CLIENT_BUNDLE_ID,
            "origin": DS_BASE_URL,
            "referer": f"{DS_BASE_URL}/"
        }
        if json_content:
            h["content-type"] = "application/json"
        return h

    def get_user_profile(self) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{DS_BASE_URL}/api/v0/users/current",
            headers=self._headers(json_content=False)
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("code") == 0:
                return data.get("data", {}).get("biz_data", {})
            raise RuntimeError(f"DeepSeek Upstream Error: {data.get('msg')}")

    def get_hif_leim(self) -> str:
        req = urllib.request.Request(
            DS_HIF_LEIM_URL,
            headers={"user-agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"]["biz_data"]["value"]

    def create_session(self) -> str:
        req = urllib.request.Request(
            f"{DS_BASE_URL}/api/v0/chat_session/create",
            data=b"{}",
            headers=self._headers()
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"]["biz_data"]["chat_session"]["id"]

    def delete_session(self, session_id: str) -> bool:
        req = urllib.request.Request(
            f"{DS_BASE_URL}/api/v0/chat_session/delete",
            data=json.dumps({"chat_session_id": session_id}).encode("utf-8"),
            headers=self._headers()
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("code") == 0

    def create_pow_challenge(self, target_path: str = "/api/v0/chat/completion") -> Dict[str, Any]:
        body = json.dumps({"target_path": target_path}).encode("utf-8")
        req = urllib.request.Request(
            f"{DS_BASE_URL}/api/v0/chat/create_pow_challenge",
            data=body,
            headers=self._headers()
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"]["biz_data"]["challenge"]

    def stream_completion(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        parent_message_id: Optional[int] = None,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        force_new_session: bool = False
    ) -> Generator[Tuple[str, str, str, Optional[int]], None, None]:
        """
        Yields (fragment_type, token_str, session_id, response_message_id)
        """
        # Resolve session
        if session_id:
            active_sid = session_id
            active_parent = parent_message_id
        else:
            if force_new_session or not session_manager.session_id:
                new_sid = self.create_session()
                session_manager.reset(new_sid)
            active_sid = session_manager.session_id
            active_parent = session_manager.parent_message_id

        # Solve anti-bot PoW + get leim token
        leim_val = self.get_hif_leim()
        pow_challenge = self.create_pow_challenge()
        b64_pow = PoWSolver.solve(pow_challenge)

        payload = {
            "chat_session_id": active_sid,
            "parent_message_id": active_parent,
            "model_type": "default",
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "action": None,
            "preempt": False
        }

        headers = self._headers()
        headers["x-ds-pow-response"] = b64_pow
        headers["x-hif-leim"] = leim_val

        req_url = urllib.request.Request(
            f"{DS_BASE_URL}/api/v0/chat/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers
        )

        try:
            resp = urllib.request.urlopen(req_url, timeout=120)
        except urllib.error.HTTPError as e:
            # Upstream 400 or 404 indicates session expired/deleted: auto-heal once
            if e.code in (400, 404) and not force_new_session:
                print(f"[UpstreamClient] Session {active_sid} error HTTP {e.code}. Auto-healing with new session...")
                yield from self.stream_completion(
                    prompt=prompt,
                    session_id=None,
                    parent_message_id=None,
                    thinking_enabled=thinking_enabled,
                    search_enabled=search_enabled,
                    force_new_session=True
                )
                return
            raise

        current_type = "RESPONSE"
        latest_resp_id = None

        with resp:
            for line in resp:
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str.startswith("data: "):
                    continue
                chunk = line_str[6:]
                if not chunk.startswith("{"):
                    continue
                try:
                    obj = json.loads(chunk)
                except:
                    continue

                if "response_message_id" in obj:
                    latest_resp_id = obj["response_message_id"]

                # Detect type transitions
                if "v" in obj and isinstance(obj["v"], dict) and "response" in obj["v"]:
                    frags = obj["v"]["response"].get("fragments", [])
                    if frags:
                        current_type = frags[-1].get("type", "RESPONSE")
                        init_text = frags[-1].get("content", "")
                        if init_text:
                            yield current_type, init_text, active_sid, latest_resp_id
                elif obj.get("p") == "response/fragments" and isinstance(obj.get("v"), list) and obj["v"]:
                    current_type = obj["v"][-1].get("type", "RESPONSE")
                    init_text = obj["v"][-1].get("content", "")
                    if init_text:
                        yield current_type, init_text, active_sid, latest_resp_id

                # Token chunk
                p = obj.get("p")
                v = obj.get("v")
                tok = None
                if p is None and isinstance(v, str):
                    tok = v
                elif p == "response/fragments/-1/content" and isinstance(v, str):
                    tok = v

                if tok is not None:
                    yield current_type, tok, active_sid, latest_resp_id

        # Update parent ID in local session manager if using the active session
        if not session_id and latest_resp_id:
            session_manager.update_parent(latest_resp_id)
