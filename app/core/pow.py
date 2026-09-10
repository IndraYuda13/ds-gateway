import json
import base64
import subprocess
import shutil
import time
from typing import Dict, Any, Optional

from app.config import SOLVER_PATH

# ─────────────────────────────────────────────────────────────────────────────
# Pure Python Fallback Keccak-256 for DeepSeek
# ─────────────────────────────────────────────────────────────────────────────
_ROUND_CONSTANTS = [
    0, 1, 0, 32898, 0x80000000, 32906, 0x80000000, 0x80008000, 0, 32907, 0, 0x80000001,
    0x80000000, 0x80008081, 0x80000000, 32777, 0, 138, 0, 136, 0, 0x80008009, 0, 0x8000000a,
    0, 0x8000808b, 0x80000000, 139, 0x80000000, 32905, 0x80000000, 32771, 0x80000000, 32770,
    0x80000000, 128, 0, 32778, 0x80000000, 0x8000000a, 0x80000000, 0x80008081, 0x80000000,
    32896, 0, 0x80000001, 0x80000000, 0x80008008
]
_V = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]
_W = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]

def _keccak_round(state, c, d_arr, w_arr):
    for t in range(5):
        n = 2 * t; i = (t + 5) * 2; o = (t + 10) * 2; f = (t + 15) * 2; s = (t + 20) * 2
        c[n] = state[n] ^ state[i] ^ state[o] ^ state[f] ^ state[s]
        c[n+1] = state[n+1] ^ state[i+1] ^ state[o+1] ^ state[f+1] ^ state[s+1]
    for t in range(5):
        w_arr[0] = c[((t+1)%5)*2]
        w_arr[1] = c[((t+1)%5)*2 + 1]
        o = w_arr[0]; f = w_arr[1]
        w_arr[0] = ((o << 1) & 0xffffffff) | (f >> 31)
        w_arr[1] = ((f << 1) & 0xffffffff) | (o >> 31)
        d_arr[2*t] = c[((t+4)%5)*2] ^ w_arr[0]
        d_arr[2*t+1] = c[((t+4)%5)*2 + 1] ^ w_arr[1]
        for r in range(0, 25, 5):
            state[(r+t)*2] ^= d_arr[2*t]
            state[(r+t)*2+1] ^= d_arr[2*t+1]
    w_arr[0] = state[2]; w_arr[1] = state[3]
    for i in range(24):
        t = _V[i]; a = _W[i]
        c[0] = state[t*2]; c[1] = state[t*2+1]
        o = w_arr[0]; f = w_arr[1]
        shift_a = a & 31
        shift_u = (32 - a) & 31
        s = 0 if a < 32 else 1
        w_arr[s] = ((o << shift_a) & 0xffffffff) | (f >> shift_u)
        w_arr[(s+1)%2] = ((f << shift_a) & 0xffffffff) | (o >> shift_u)
        state[t*2] = w_arr[0]; state[t*2+1] = w_arr[1]
        w_arr[0] = c[0]; w_arr[1] = c[1]
    for t in range(0, 25, 5):
        for n in range(5):
            c[n*2] = state[(t+n)*2]
            c[n*2+1] = state[(t+n)*2+1]
        for n in range(5):
            o = ((n+1)%5)*2; f = ((n+2)%5)*2
            state[(t+n)*2] ^= (~c[o] & 0xffffffff) & c[f]
            state[(t+n)*2+1] ^= (~c[o+1] & 0xffffffff) & c[f+1]

def deepseek_hash_py(msg_bytes: bytes) -> str:
    state = [0] * 50
    queue = bytearray(136)
    q_off = 0
    c = [0] * 10
    d_arr = [0] * 10
    w_arr = [0] * 2

    def run_keccak():
        for rnd in range(1, 24):
            _keccak_round(state, c, d_arr, w_arr)
            state[0] ^= _ROUND_CONSTANTS[2 * rnd]
            state[1] ^= _ROUND_CONSTANTS[2 * rnd + 1]

    for b in msg_bytes:
        queue[q_off] = b
        q_off += 1
        if q_off >= 136:
            for r in range(0, 136, 8):
                n = r // 4
                state[n] ^= (queue[r+7] << 24) | (queue[r+6] << 16) | (queue[r+5] << 8) | queue[r+4]
                state[n+1] ^= (queue[r+3] << 24) | (queue[r+2] << 16) | (queue[r+1] << 8) | queue[r]
            run_keccak()
            q_off = 0

    q_copy = bytearray(queue)
    s_copy = list(state)
    for i in range(q_off, 136):
        q_copy[i] = 0
    q_copy[q_off] |= 6
    q_copy[135] |= 128
    for r in range(0, 136, 8):
        n = r // 4
        s_copy[n] ^= (q_copy[r+7] << 24) | (q_copy[r+6] << 16) | (q_copy[r+5] << 8) | q_copy[r+4]
        s_copy[n+1] ^= (q_copy[r+3] << 24) | (q_copy[r+2] << 16) | (q_copy[r+1] << 8) | q_copy[r]

    for rnd in range(1, 24):
        _keccak_round(s_copy, c, d_arr, w_arr)
        s_copy[0] ^= _ROUND_CONSTANTS[2 * rnd]
        s_copy[1] ^= _ROUND_CONSTANTS[2 * rnd + 1]

    out = bytearray(32)
    for r in range(0, 32, 8):
        n = r // 4
        out[r] = s_copy[n+1] & 0xff
        out[r+1] = (s_copy[n+1] >> 8) & 0xff
        out[r+2] = (s_copy[n+1] >> 16) & 0xff
        out[r+3] = (s_copy[n+1] >> 24) & 0xff
        out[r+4] = s_copy[n] & 0xff
        out[r+5] = (s_copy[n] >> 8) & 0xff
        out[r+6] = (s_copy[n] >> 16) & 0xff
        out[r+7] = (s_copy[n] >> 24) & 0xff
    return out.hex()

def solve_pow_py(cfg: dict) -> int:
    target = cfg["challenge"]
    salt = cfg["salt"]
    expire_at = cfg["expire_at"]
    diff = cfg["difficulty"]
    prefix = f"{salt}_{expire_at}_".encode("utf-8")
    for i in range(diff):
        h = deepseek_hash_py(prefix + str(i).encode("utf-8"))
        if h == target:
            return i
    raise RuntimeError("PoW solution not found within difficulty range (Pure Python)")

class PoWSolver:
    @staticmethod
    def solve(challenge_config: Dict[str, Any], verbose: bool = False) -> str:
        has_node = shutil.which("node") is not None
        answer = None
        t0 = time.time()

        if has_node and SOLVER_PATH.exists():
            try:
                res = subprocess.run(
                    ["node", str(SOLVER_PATH), json.dumps(challenge_config)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True
                )
                answer = int(res.stdout.strip())
                if verbose:
                    print(f"[PoW] Solved via Node/WASM in {time.time()-t0:.3f}s (nonce: {answer})")
            except Exception as e:
                if verbose:
                    print(f"[PoW] Node solver failed ({e}), falling back to Python...")

        if answer is None:
            if verbose:
                print("[PoW] Solving via Pure Python Keccak...")
            answer = solve_pow_py(challenge_config)
            if verbose:
                print(f"[PoW] Solved via Python in {time.time()-t0:.3f}s (nonce: {answer})")

        resp_obj = {
            "algorithm": challenge_config["algorithm"],
            "challenge": challenge_config["challenge"],
            "salt": challenge_config["salt"],
            "answer": answer,
            "signature": challenge_config["signature"],
            "target_path": challenge_config["target_path"]
        }
        return base64.b64encode(json.dumps(resp_obj).encode("utf-8")).decode("ascii")
