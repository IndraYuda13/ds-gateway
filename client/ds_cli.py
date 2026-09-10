#!/usr/bin/env python3
"""
DeepSeek Terminal Client & Python SDK (Smart Session Pool Edition)
Connects to DeepSeek Web API Proxy via Cloudflare Tunnel or local host.
Zero external dependencies (Python Standard Library only).
"""

import sys
import os
import json
import urllib.request
import urllib.error
import argparse
from typing import Generator, Dict, Any, Tuple, Optional, List

DEFAULT_API_URL = os.getenv("DEEPSEEK_API_BASE", "https://deepseek.indrayuda.my.id")

# ANSI Colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def color(text: str, c: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{c}{text}{RESET}"

class DeepSeekCLIClient:
    def __init__(self, api_url: str = DEFAULT_API_URL, api_key: str = "lemon"):
        self.base_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "User-Agent": "DeepSeekClient/1.2",
            "Authorization": f"Bearer {self.api_key}"
        }

    def check_health(self) -> Dict[str, Any]:
        try:
            req = urllib.request.Request(f"{self.base_url}/health", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e), "status": "offline"}

    def new_session(self, conv_id: Optional[str] = None) -> Optional[str]:
        try:
            url = f"{self.base_url}/chat/new"
            if conv_id:
                url += f"?conv_id={conv_id}"
            req = urllib.request.Request(url, data=b"{}", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("session_id")
        except Exception as e:
            print(color(f"[Error] Failed to reset session: {e}", RED))
            return None

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "ds-chat",
        thinking: Optional[bool] = None,
        search: bool = False,
        new_session: bool = False,
        session_id: Optional[str] = None,
        user: Optional[str] = None
    ) -> Generator[Tuple[Dict[str, Any], Optional[str]], None, None]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "search": search,
            "new_session": new_session
        }
        if session_id:
            payload["session_id"] = session_id
        if user:
            payload["user"] = user
        if thinking is not None:
            payload["thinking"] = thinking

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers()
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str.startswith("data: "):
                    continue
                chunk_raw = line_str[6:]
                if chunk_raw == "[DONE]":
                    break
                if not chunk_raw.startswith("{"):
                    continue
                try:
                    chunk = json.loads(chunk_raw)
                    delta = chunk["choices"][0].get("delta", {})
                    sess_id = chunk.get("session_id")
                    yield delta, sess_id
                except:
                    continue


def main():
    parser = argparse.ArgumentParser(description="DeepSeek Web API Terminal Client (Smart Session Pool)")
    parser.add_argument("prompt", nargs="?", help="Direct prompt. If omitted, opens interactive REPL mode.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API Base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--no-think", action="store_true", help="Disable DeepSeek-R1 reasoning (Instant mode)")
    parser.add_argument("--search", action="store_true", help="Enable live web search")
    parser.add_argument("--new", action="store_true", help="Force create a new conversation thread on DeepSeek")
    parser.add_argument("--user", default=None, help="Explicit conversation/user ID for session isolation")
    args = parser.parse_args()

    client = DeepSeekCLIClient(api_url=args.api_url)

    # 1. Non-interactive single shot / piped input
    if args.prompt or not sys.stdin.isatty():
        prompt = args.prompt
        if not prompt:
            prompt = sys.stdin.read().strip()
        if not prompt:
            print(color("Prompt cannot be empty.", RED))
            sys.exit(1)

        model = "ds-chat" if args.no_think else "ds-reasoner"
        in_think = False

        try:
            msgs = [{"role": "user", "content": prompt}]
            for delta, _ in client.stream_chat(msgs, model=model, search=args.search, new_session=args.new, user=args.user):
                if "reasoning_content" in delta:
                    if not in_think:
                        sys.stderr.write(color("\n--- [Thinking / R1 Reasoning] ---\n", YELLOW))
                        in_think = True
                    sys.stderr.write(color(delta["reasoning_content"], DIM))
                    sys.stderr.flush()
                if "content" in delta:
                    if in_think:
                        sys.stderr.write(color("\n--- [Response] ---\n", YELLOW))
                        in_think = False
                    sys.stdout.write(delta["content"])
                    sys.stdout.flush()
            sys.stdout.write("\n")
        except Exception as e:
            print(color(f"\n[Error] Failed to connect to API: {e}", RED))
            sys.exit(1)
        return

    # 2. Interactive REPL Mode
    health = client.check_health()
    is_online = health.get("status") == "online"
    status_str = color("Online ✓", GREEN) if is_online else color(f"Warning ({health.get('error', 'offline')})", RED)
    user_name = health.get("account", {}).get("name", "User")
    active_pool_count = health.get("pool", {}).get("active_conversations_count", 0)

    print(color("=" * 64, CYAN))
    print(color("       DeepSeek Terminal Client (Smart Session Pool)       ", BOLD + CYAN))
    print(color("=" * 64, CYAN))
    print(f"  {color('API Endpoint  :', BOLD)} {client.base_url}")
    print(f"  {color('Status        :', BOLD)} {status_str} ({user_name})")
    print(f"  {color('Active Pool   :', BOLD)} {active_pool_count} active conversation slots")
    print(f"  {color('Commands      :', BOLD)} /think [on|off], /search [on|off], /new, /exit")
    print(color("-" * 64, CYAN))

    thinking_on = not args.no_think
    search_on = args.search
    force_next_new = args.new
    history: List[Dict[str, str]] = []

    while True:
        try:
            status_tag = f"[{color('Think: ' + ('ON' if thinking_on else 'OFF'), YELLOW)} | {color('Search: ' + ('ON' if search_on else 'OFF'), BLUE)}]"
            prompt = input(f"\n{color('You', BOLD + CYAN)} {status_tag} > ").strip()

            if not prompt:
                continue

            if prompt.lower() in ('/exit', '/quit', 'exit', 'quit'):
                print(color("Sampai jumpa bre! 👋", CYAN))
                break
            elif prompt.lower() in ('/new', '/clear', '/reset'):
                history.clear()
                force_next_new = True
                print(color("[✓] Fresh conversation thread initialized (history cleared).", GREEN))
                continue
            elif prompt.lower() == '/think on':
                thinking_on = True
                print(color("[✓] R1 Deep Reasoning mode ENABLED.", GREEN))
                continue
            elif prompt.lower() == '/think off':
                thinking_on = False
                print(color("[✓] Fast Instant mode ENABLED.", GREEN))
                continue
            elif prompt.lower() == '/search on':
                search_on = True
                print(color("[✓] Live Web Search ENABLED.", GREEN))
                continue
            elif prompt.lower() == '/search off':
                search_on = False
                print(color("[✓] Live Web Search DISABLED.", GREEN))
                continue
            elif prompt.lower() == '/help':
                print("Commands:")
                print("  /think on|off  : Toggle DeepSeek-R1 reasoning")
                print("  /search on|off : Toggle web search integration")
                print("  /new           : Start a fresh chat thread")
                print("  /exit          : Exit client")
                continue

            model = "ds-reasoner" if thinking_on else "ds-chat"
            in_think = False
            first_resp = True
            ans_buf = []

            # Append current turn to client history
            history.append({"role": "user", "content": prompt})

            for delta, sid in client.stream_chat(
                messages=history,
                model=model,
                search=search_on,
                new_session=force_next_new,
                user=args.user
            ):
                force_next_new = False
                if "reasoning_content" in delta:
                    if not in_think:
                        sys.stdout.write(f"\n{color('💭 Thinking:', YELLOW + BOLD)}\n")
                        in_think = True
                    sys.stdout.write(color(delta["reasoning_content"], DIM))
                    sys.stdout.flush()
                if "content" in delta:
                    if in_think:
                        sys.stdout.write("\n")
                        in_think = False
                    if first_resp:
                        sys.stdout.write(f"\n{color('🤖 DeepSeek:', GREEN + BOLD)}\n")
                        first_resp = False
                    tok = delta["content"]
                    ans_buf.append(tok)
                    sys.stdout.write(tok)
                    sys.stdout.flush()
            sys.stdout.write("\n")

            # Save assistant reply to client history for multi-turn continuity
            history.append({"role": "assistant", "content": "".join(ans_buf)})

        except KeyboardInterrupt:
            print(color("\n[!] Type /exit to quit.", YELLOW))
        except Exception as e:
            print(color(f"\n[Error] {e}", RED))

if __name__ == "__main__":
    main()
