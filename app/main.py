from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.config import DEBUG

app = FastAPI(
    title="DeepSeek Unofficial Web API",
    description="Production-grade OpenAI-compatible API proxy for chat.deepseek.com with anti-bot WASM bypass & sticky session management.",
    version="1.1.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
