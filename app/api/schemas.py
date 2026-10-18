from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class MessageItem(BaseModel):
    role: str
    # str for plain text, or a list of content parts (text + image_url) for vision
    content: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "deepseek-chat"
    messages: List[MessageItem]
    stream: Optional[bool] = False
    # bool, or Anthropic-style {"type": "enabled"/"disabled"} (coerced below)
    thinking: Optional[Any] = None
    search: Optional[bool] = False
    session_id: Optional[str] = None
    new_session: Optional[bool] = False
    user: Optional[str] = None
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    functions: Optional[List[Dict[str, Any]]] = None  # legacy function-calling format
    stream_options: Optional[Dict[str, Any]] = None   # OpenAI-style {include_usage}

    @field_validator("thinking", mode="before")
    @classmethod
    def _coerce_thinking(cls, v):
        """Normalize thinking to bool/None: accept bool, {"type": "enabled"},
        {"type": "disabled", "budget_tokens": N}, or plain strings."""
        if v is None or isinstance(v, bool):
            return v
        if isinstance(v, dict):
            t = v.get("type")
            if isinstance(t, str):
                return t.strip().lower() in ("enabled", "on", "true", "yes", "1")
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("enabled", "on", "true", "yes", "1")
        return bool(v)

class SimpleChatRequest(BaseModel):
    prompt: str
    thinking: Optional[bool] = False
    search: Optional[bool] = False
    stream: Optional[bool] = False
    new_session: Optional[bool] = False
    user: Optional[str] = None
    session_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChoiceItem(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # upstream DeepSeek web gives no billed counts; these are gateway estimates
    estimated: Optional[bool] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    session_id: Optional[str] = None
    choices: List[ChoiceItem]
    usage: UsageInfo
