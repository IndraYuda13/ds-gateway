from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

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
    thinking: Optional[bool] = None
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

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    session_id: Optional[str] = None
    choices: List[ChoiceItem]
    usage: UsageInfo
