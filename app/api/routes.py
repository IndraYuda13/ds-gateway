import time
import uuid
import json
from typing import Optional, Any
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import DEFAULT_TOKEN, PROXY_API_KEY
from app.core.session import session_manager
from app.core.client import DeepSeekUpstreamClient
from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    SimpleChatRequest
)

router = APIRouter()

def authenticate(authorization: Optional[str] = None) -> str:
    token = DEFAULT_TOKEN
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        bearer = authorization.split("Bearer ", 1)[1].strip()
        # If proxy has a custom API key protection configured
        if PROXY_API_KEY and bearer != PROXY_API_KEY and not bearer.startswith("qos"):
            raise HTTPException(status_code=401, detail="Invalid Proxy API Key")
        if bearer and bearer not in ("lemon", "default", "sk-123", "none", PROXY_API_KEY):
            token = bearer
    return token

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

    return {
        "status": "online",
        "service": "DeepSeek Web API Proxy",
        "version": "1.1.0",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "active_session": {
            "id": session_manager.session_id,
            "parent_message_id": session_manager.parent_message_id
        },
        "account": {
            "name": user_info.get("name", "Unknown"),
            "email": user_info.get("email", "Unknown")
        }
    }

@router.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek",
                "permission": [],
                "root": "deepseek-chat",
                "parent": None
            },
            {
                "id": "deepseek-reasoner",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek",
                "permission": [],
                "root": "deepseek-reasoner",
                "parent": None
            }
        ]
    }

@router.post("/v1/chat/sessions/new")
@router.post("/chat/new")
def new_session(authorization: Optional[str] = Header(default=None)):
    token = authenticate(authorization)
    client = DeepSeekUpstreamClient(token)
    new_id = client.create_session()
    session_manager.reset(new_id)
    return {
        "status": "ok",
        "message": "New session created successfully.",
        "session_id": new_id
    }

def sse_event_stream(req: ChatCompletionRequest, token: str, force_new: bool = False):
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    client = DeepSeekUpstreamClient(token)

    thinking_enabled = False
    if req.thinking is not None:
        thinking_enabled = req.thinking
    elif req.model and ("reasoner" in req.model.lower() or "r1" in req.model.lower()):
        thinking_enabled = True

    # Build prompt from messages
    if len(req.messages) == 1:
        prompt = req.messages[0].content
    else:
        prompt = req.messages[-1].content

    active_sess_id = session_manager.session_id

    for frag_type, tok, sid, resp_id in client.stream_completion(
        prompt=prompt,
        session_id=req.session_id,
        thinking_enabled=thinking_enabled,
        search_enabled=bool(req.search),
        force_new_session=force_new
    ):
        active_sess_id = sid
        delta = {"reasoning_content": tok} if frag_type == "THINK" else {"content": tok}
        chunk_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "session_id": active_sess_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "session_id": active_sess_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@router.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
    x_new_session: Optional[str] = Header(default=None)
):
    token = authenticate(authorization)
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
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    active_sid = session_manager.session_id

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
            except:
                pass

    content_str = "".join(content_buf)
    reasoning_str = "".join(thinking_buf) if thinking_buf else None

    msg = {"role": "assistant", "content": content_str}
    if reasoning_str:
        msg["reasoning_content"] = reasoning_str

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
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(content_str),
            "total_tokens": len(content_str)
        }
    }

@router.post("/chat")
def simple_chat(req: SimpleChatRequest, authorization: Optional[str] = Header(default=None)):
    from app.api.schemas import MessageItem
    chat_req = ChatCompletionRequest(
        model="deepseek-reasoner" if req.thinking else "deepseek-chat",
        messages=[MessageItem(role="user", content=req.prompt)],
        stream=req.stream,
        thinking=req.thinking,
        search=req.search,
        new_session=req.new_session
    )
    if req.stream:
        return chat_completions(chat_req, authorization=authorization)
    res = chat_completions(chat_req, authorization=authorization)
    return {
        "response": res["choices"][0]["message"]["content"],
        "reasoning": res["choices"][0]["message"].get("reasoning_content"),
        "session_id": res.get("session_id")
    }
