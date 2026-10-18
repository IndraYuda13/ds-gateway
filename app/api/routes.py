import time
import uuid
from typing import List
import json
from typing import Optional, Any
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import DEFAULT_TOKEN, PROXY_API_KEY
from app.core.session import smart_pool
from app.core.client import DeepSeekUpstreamClient
from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    SimpleChatRequest
)
from app.core.tools import (
    build_tool_instructions,
    parse_tool_calls,
    infer_fallback_name,
    strip_dsml_region,
)
from app.core.tokens import estimate_text
from app.core.vision import extract_images

router = APIRouter()

def authenticate(authorization: Optional[str] = None) -> str:
    token = DEFAULT_TOKEN
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        bearer = authorization.split("Bearer ", 1)[1].strip()
        if PROXY_API_KEY and bearer != PROXY_API_KEY and not bearer.startswith("qos"):
            raise HTTPException(status_code=401, detail="Invalid Proxy API Key")
        if bearer and bearer not in ("lemon", "default", "sk-123", "none", PROXY_API_KEY):
            token = bearer
    return token

def _format_msg(m) -> str:
    """Render one client message into the upstream prompt form (role-aware)."""
    role = (m.role or "user").lower()
    content = m.content or ""
    if isinstance(content, list):
        # vision-style content parts: keep the text parts for the prompt
        texts = [
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        content = "\n".join(t for t in texts if t).strip()
    if role == "tool":
        label = m.name or m.tool_call_id or "tool"
        return f"[Tool result for {label}]:\n{content}"
    if role == "assistant" and m.tool_calls:
        parts = [content] if content else []
        for tc in m.tool_calls:
            fn = (tc or {}).get("function") or {}
            parts.append(json.dumps(
                {"name": fn.get("name", ""), "arguments": fn.get("arguments", "")},
                ensure_ascii=False,
            ))
        return "[Assistant tool call]:\n" + "\n".join(parts)
    return content

def build_last_prompt(req: ChatCompletionRequest) -> str:
    """Build the upstream prompt from the last message (role-aware)."""
    if not req.messages:
        return ""
    return _format_msg(req.messages[-1])

def build_messages_prompt(msgs) -> str:
    """Render a delta of messages (e.g. several parallel tool results) as
    one upstream turn. Continuations must deliver EVERY message the
    upstream conversation has not seen yet — not just the final one."""
    chunks = [_format_msg(m) for m in msgs]
    chunks = [c for c in chunks if str(c).strip()]
    return "\n\n".join(chunks) if chunks else ""


def estimate_request_tokens(req: ChatCompletionRequest) -> int:
    """OpenAI-convention prompt estimate: all request messages + tool
    schemas, plus per-message framing overhead."""
    total = 0
    for m in req.messages:
        total += estimate_text(_format_msg(m)) + 4
    if req.tools:
        total += estimate_text(json.dumps(req.tools, ensure_ascii=False))
    if req.functions:
        total += estimate_text(json.dumps(req.functions, ensure_ascii=False))
    return total


def build_system_prefix(req: ChatCompletionRequest) -> str:
    """System messages + tool instructions; injected on fresh sessions only."""
    parts = []
    sys_text = "\n\n".join(
        (m.content or "") for m in req.messages
        if m.role == "system" and m.content
    ).strip()
    if sys_text:
        parts.append("[System instructions]\n" + sys_text)
    tool_text = build_tool_instructions(req.tools, req.functions, req.tool_choice)
    if tool_text:
        parts.append(tool_text)
    return "\n\n".join(parts)


@router.get("/health")
@router.get("/")
def health(authorization: Optional[str] = Header(default=None)):
    token = authenticate(authorization)
    client = DeepSeekUpstreamClient(token)
    user_info = {}
    try:
        user_info = client.get_user_profile()
    except Exception as e:
        user_info = {"error": str(e)}

    active_convs = [
        {
            "conv_id": k,
            "session_id": v.session_id,
            "turn_count": v.turn_count,
            "last_active": v.last_active
        }
        for k, v in smart_pool.pool.items()
    ]

    return {
        "status": "online",
        "service": "ds-gateway",
        "version": "1.2.0",
        "models": ["ds-chat", "ds-reasoner", "deepseek-chat", "deepseek-reasoner"],
        "pool": {
            "active_conversations_count": len(smart_pool.pool),
            "max_pool_size": smart_pool.max_size,
            "conversations": active_convs
        },
        "account": {
            "name": user_info.get("name", "Unknown"),
            "email": user_info.get("email", "Unknown")
        }
    }

@router.get("/v1/models")
def list_models():
    models = [
        {"id": "ds-chat", "root": "ds-chat"},
        {"id": "ds-reasoner", "root": "ds-reasoner"},
        {"id": "deepseek-chat", "root": "deepseek-chat"},
        {"id": "deepseek-reasoner", "root": "deepseek-reasoner"},
    ]
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek",
                "permission": [],
                "root": m["root"],
                "parent": None
            }
            for m in models
        ]
    }

@router.post("/v1/chat/sessions/new")
@router.post("/chat/new")
def new_session(
    conv_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None)
):
    token = authenticate(authorization)
    client = DeepSeekUpstreamClient(token)
    
    target_conv = conv_id or f"manual_{int(time.time())}"
    new_sid, _ = smart_pool.acquire(
        conv_id=target_conv,
        create_fn=client.create_session,
        delete_fn=client.delete_session,
        force_new=True
    )
    return {
        "status": "ok",
        "message": "New session created in pool successfully.",
        "conv_id": target_conv,
        "session_id": new_sid
    }

def build_transcript_prompt(req: ChatCompletionRequest, max_chars: int = 14000) -> str:
    """Render the whole client history into one prompt.

    Used when a request opens a FRESH upstream conversation but carries
    existing history (context compaction, edited/replayed transcript,
    mid-session fork): the new DeepSeek conversation otherwise starts
    blank, so the replayed history must be handed to it in full.
    """
    lines: List[str] = []
    for m in req.messages:
        role = (m.role or "user").lower()
        if role == "system":
            continue
        content = m.content or ""
        if isinstance(content, list):
            texts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "\n".join(t for t in texts if t).strip()
        extra = ""
        if m.tool_calls:
            names = [
                (tc or {}).get("function", {}).get("name", "?")
                for tc in m.tool_calls
            ]
            extra = f" [called tools: {', '.join(names)}]"
        if not str(content).strip() and not extra:
            continue
        if role == "tool":
            label = m.name or m.tool_call_id or "tool"
            lines.append(f"[Tool result {label}]: {str(content)[:1500]}")
        elif role == "assistant":
            lines.append(f"Assistant: {str(content)[:1500]}{extra}")
        else:
            lines.append(f"User: {str(content)[:1500]}")
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
        nl = joined.find("\n")
        if nl != -1:
            joined = joined[nl + 1:]
    return (
        "Below is a conversation between a User and the Assistant that "
        "already happened. Continue as the Assistant from exactly where "
        "it left off and respond to the last message. Do not restart the "
        "conversation.\n\n" + joined
    )

def sse_event_stream(req: ChatCompletionRequest, token: str, force_new: bool = False):
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    client = DeepSeekUpstreamClient(token)

    thinking_enabled = False
    if req.thinking is not None:
        thinking_enabled = req.thinking
    elif req.model and ("reasoner" in req.model.lower() or "r1" in req.model.lower()):
        thinking_enabled = True

    # Compute conversation ID from messages/user/session_id.
    # Continuation requires the incoming history to exactly extend the
    # pooled conversation's last-seen history (prefix-chain); a repeated
    # opening line from a fresh client session no longer inherits an old
    # upstream conversation.
    msgs_dict = [m.dict() for m in req.messages]
    conv_id, continued = smart_pool.route_full(
        msgs_dict,
        user=req.user,
        explicit_session_id=req.session_id,
        force_new=force_new
    )

    # Fresh conversation that arrives with existing history (compaction,
    # edited/replayed transcript): the blank upstream conversation must be
    # seeded with the whole replay, not just the final message.
    # Continued conversation: send only the messages the upstream has not
    # seen yet (a parallel tool round adds several tool results at once).
    if continued:
        d0 = smart_pool.depth_of(conv_id)
        delta = req.messages[d0:]
        prompt = build_messages_prompt(delta) if delta else build_last_prompt(req)
    elif len(req.messages) > 2:
        prompt = build_transcript_prompt(req)
    else:
        prompt = build_last_prompt(req)
    system_prefix = build_system_prefix(req)

    # Vision: upload images from the last user message, get ref_file_ids
    ref_file_ids = []
    try:
        images = extract_images(req.messages)
        for img_bytes, filename, content_type in images:
            fid = client.upload_file(img_bytes, filename, content_type)
            ref_file_ids.append(fid)
            print(f"[Vision] uploaded {filename} -> {fid}")
    except Exception as e:
        print(f"[Vision] image handling failed: {e}")

    # When tools are declared, buffer the stream so tool-call blocks can be
    # parsed and emitted as structured tool_calls deltas at the end.
    # Also enable when the message history shows an agent loop (tool results
    # or prior assistant tool_calls): clients often omit `tools` on follow-up
    # turns, but the model still holds the protocol from session context.
    tools_mode = bool(req.tools or req.functions)
    if not tools_mode:
        for m in req.messages:
            if (m.role or "").lower() == "tool" or m.tool_calls:
                tools_mode = True
                break
    think_buf, content_buf = [], []
    out_parts: List[str] = []
    active_sess_id = None

    def _chunk(delta, finish_reason=None):
        return {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "session_id": active_sess_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        }

    for frag_type, tok, sid, resp_id in client.stream_completion(
        prompt=prompt,
        conv_id=conv_id,
        thinking_enabled=thinking_enabled,
        search_enabled=bool(req.search),
        force_new_session=force_new,
        inject_prefix=system_prefix,
        ref_file_ids=ref_file_ids
    ):
        active_sess_id = sid
        if tools_mode:
            (think_buf if frag_type == "THINK" else content_buf).append(tok)
            continue
        delta = {"reasoning_content": tok} if frag_type == "THINK" else {"content": tok}
        out_parts.append(tok)
        yield f"data: {json.dumps(_chunk(delta), ensure_ascii=False)}\n\n"

    finish_reason = "stop"
    reasoning_str, cleaned, tool_calls = "", "", None
    if tools_mode:
        reasoning_str = "".join(think_buf)
        content_str = "".join(content_buf)
        # Degraded emissions (e.g. the invoke tag lost on long payloads) are
        # recovered by inferring the tool from declared schemas or the last
        # tool used in this conversation.
        fallback = infer_fallback_name(content_str, req.tools, req.functions, msgs_dict)
        cleaned, tool_calls = parse_tool_calls(content_str, fallback_name=fallback)
        if not tool_calls and reasoning_str:
            # block split across the THINK -> RESPONSE boundary
            fb2 = infer_fallback_name(reasoning_str + "\n" + content_str,
                                      req.tools, req.functions, msgs_dict)
            _, c_calls = parse_tool_calls(reasoning_str + "\n" + content_str,
                                          fallback_name=fb2)
            if c_calls:
                tool_calls = c_calls
                cleaned = ""
                reasoning_str = strip_dsml_region(reasoning_str)
        if reasoning_str:
            d = _chunk({"reasoning_content": reasoning_str})
            yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
        if tool_calls:
            d = _chunk({"tool_calls": tool_calls})
            yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
            finish_reason = "tool_calls"
        elif cleaned:
            d = _chunk({"content": cleaned})
            yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

    # Response fully generated: anchor the conversation at this history so
    # the next turn (which replays it as a prefix) continues this conversation.
    smart_pool.note(conv_id, msgs_dict)

    d = _chunk({}, finish_reason)
    yield f"data: {json.dumps(d)}\n\n"

    if req.stream_options and req.stream_options.get("include_usage"):
        if tools_mode:
            comp_text = reasoning_str + cleaned + (json.dumps(tool_calls) if tool_calls else "")
        else:
            comp_text = "".join(out_parts)
        pt_est = estimate_request_tokens(req)
        ct_est = estimate_text(comp_text)
        usage_evt = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [],
            "usage": {"prompt_tokens": pt_est, "completion_tokens": ct_est,
                      "total_tokens": pt_est + ct_est, "estimated": True}
        }
        yield f"data: {json.dumps(usage_evt, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"

@router.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
    x_new_session: Optional[str] = Header(default=None),
    x_session_id: Optional[str] = Header(default=None)
):
    token = authenticate(authorization)
    if x_session_id and not req.session_id:
        req.session_id = x_session_id

    force_new = (isinstance(x_new_session, str) and x_new_session.lower() in ("true", "1")) or bool(req.new_session)

    if req.stream:
        return StreamingResponse(
            sse_event_stream(req, token, force_new=force_new),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming aggregation
    thinking_buf = []
    content_buf = []
    tool_buf = []
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    active_sid = None

    for chunk_line in sse_event_stream(req, token, force_new=force_new):
        if chunk_line.startswith("data: ") and not chunk_line.startswith("data: [DONE]"):
            try:
                data = json.loads(chunk_line[6:].strip())
                active_sid = data.get("session_id", active_sid)
                delta = data["choices"][0]["delta"]
                if "reasoning_content" in delta:
                    thinking_buf.append(delta["reasoning_content"])
                if "content" in delta:
                    content_buf.append(delta["content"])
                if "tool_calls" in delta:
                    tool_buf.extend(delta["tool_calls"])
            except:
                pass

    content_str = "".join(content_buf)
    reasoning_str = "".join(thinking_buf) if thinking_buf else None

    # Catch spontaneous tool-call blocks even when tools were not declared
    if not tool_buf:
        fb = infer_fallback_name(content_str, req.tools, req.functions,
                                 [m.dict() for m in req.messages])
        content_str, tool_buf = parse_tool_calls(content_str, fallback_name=fb)

    msg = {"role": "assistant", "content": content_str if content_str else None}
    if reasoning_str:
        msg["reasoning_content"] = reasoning_str
    if tool_buf:
        msg["tool_calls"] = tool_buf

    pt_est = estimate_request_tokens(req)
    ct_est = (estimate_text(content_str)
              + estimate_text(reasoning_str or "")
              + (estimate_text(json.dumps(tool_buf, ensure_ascii=False)) if tool_buf else 0))

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "session_id": active_sid,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": "tool_calls" if tool_buf else "stop"
            }
        ],
        "usage": {
            "prompt_tokens": pt_est,
            "completion_tokens": ct_est,
            "total_tokens": pt_est + ct_est,
            "estimated": True
        }
    }

@router.post("/chat")
def simple_chat(req: SimpleChatRequest, authorization: Optional[str] = Header(default=None)):
    from app.api.schemas import MessageItem
    chat_req = ChatCompletionRequest(
        model="ds-reasoner" if req.thinking else "ds-chat",
        messages=[MessageItem(role="user", content=req.prompt)],
        stream=req.stream,
        thinking=req.thinking,
        search=req.search,
        new_session=req.new_session,
        user=req.user,
        session_id=req.session_id
    )
    if req.stream:
        return chat_completions(chat_req, authorization=authorization)
    res = chat_completions(chat_req, authorization=authorization)
    return {
        "response": res["choices"][0]["message"]["content"],
        "reasoning": res["choices"][0]["message"].get("reasoning_content"),
        "session_id": res.get("session_id")
    }
