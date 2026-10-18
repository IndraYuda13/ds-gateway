"""Conversation-pool isolation unit tests (no upstream calls).

Covers the prefix-chain continuation design: a request only continues a
pooled upstream conversation when its message history EXACTLY extends the
last-seen history. Identical opening lines from fresh client sessions must
never inherit another session's upstream context (the bug that made a
plain "/new" + "hai" continue an old LMS automation conversation).

Run: python3 tests/test_pool_isolation.py
"""
import os
import sys
import time
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.session import SmartSessionPool, SessionEntry  # noqa: E402

fails = []

def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + str(extra)) if extra and not cond else ""))
    if not cond:
        fails.append(name)

tmp = tempfile.mktemp(suffix=".json")
p = SmartSessionPool(file_path=tmp, max_size=10)

M1 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hai"}]
M2 = M1 + [{"role": "assistant", "content": "Hai!"}, {"role": "user", "content": "cek disk"}]
M2b = M1 + [{"role": "assistant", "content": "DIFFERENT"}, {"role": "user", "content": "cek disk"}]

# turn 1: fresh conversation
c1, cont1 = p.route_full(M1)
check("new conv id", c1.startswith("conv_") and not cont1, c1)
p.pool[c1] = SessionEntry(session_id="sess-1", conv_id=c1)
p.note(c1, M1)
check("depth anchored at turn1", p.depth_of(c1) == len(M1), p.depth_of(c1))

# turn 2 extends turn 1 exactly -> same conversation, continued=True
c2, cont2 = p.route_full(M2)
check("continuation on exact prefix", c2 == c1 and cont2, f"{c2} != {c1}")
p.note(c1, M2)
check("depth advanced after turn2", p.depth_of(c1) == len(M2), p.depth_of(c1))

# exact retry (identical bytes, fresh touch) -> same conversation
FRESH = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hai"}]
p.note(c1, FRESH)
c3, cont3 = p.route_full(FRESH)
check("full match within retry window continues (retry semantics)", c3 == c1 and cont3, c3)
p.note(c1, M2)

# same opener but the conversation is stale -> NEW conversation.
# The production incident shape: an old deep session and a fresh client
# session that begin with the same line must never share upstream context.
S1 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "rahasia: 42. balas ACK"}]
cs = p.route(S1)
p.pool[cs] = SessionEntry(session_id="sess-old", conv_id=cs)
p.note(cs, S1)
p.pool[cs].last_active = time.time() - 3600
c4 = p.route(S1)
check("stale full match gets NEW conv (/new + same opener)", c4 != cs, f"{c4} == {cs}")

# divergent history (assistant text differs) -> not a continuation
p.note(c1, M2)
c5 = p.route(M2b)
check("divergent history does not continue c1", c5 != c1, c5)

# deep identical replay (> FULL_TRUST_DEPTH) continues even when stale
DEEP = M1 + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u"},
             {"role": "assistant", "content": "b"}, {"role": "user", "content": "v"},
             {"role": "assistant", "content": "c"}, {"role": "user", "content": "w"}]
c6 = p.route(DEEP)
p.pool[c6] = SessionEntry(session_id="sess-3", conv_id=c6)
p.note(c6, DEEP)
p.pool[c6].last_active = time.time() - 7200
c7 = p.route(DEEP)
check("deep identical replay continues (real retry)", c7 == c6, c7)

# explicit session_id and user keep precedence
check("explicit sid wins", p.route(M2, explicit_session_id="abc") == "custom_abc")
check("user wins", p.route(M2, user="bob") == "user_bob")

# force_new bypasses continuation
check("force_new gives fresh id", p.route(M2, force_new=True) != c1)

# persistence round-trip
p.save()
p2 = SmartSessionPool(file_path=tmp, max_size=10)
check("chain state persists",
      p2.pool[c6].last_chain_hash == p.pool[c6].last_chain_hash
      and p2.pool[c6].last_depth == p.pool[c6].last_depth)

# two different bots (different system prompt) with same opener never collide
A = [{"role": "system", "content": "BOT-A"}, {"role": "user", "content": "hai"}]
B = [{"role": "system", "content": "BOT-B"}, {"role": "user", "content": "hai"}]
ca, cb = p.route(A), p.route(B)
check("different system prompts isolated", ca != cb, f"{ca} == {cb}")

# conv-id format per instance: identical stale openers get distinct ids
check("fresh instances never share ids", p.route(S1) != p.route(S1))

try:
    os.remove(tmp)
except OSError:
    pass
print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
