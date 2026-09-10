#!/usr/bin/env python3
"""
Entrypoint for DeepSeek Web API Proxy Server
Usage:
    python main.py
    uvicorn app.main:app --host 0.0.0.0 --port 8550
"""
import uvicorn
from app.config import HOST, PORT, DEBUG

if __name__ == "__main__":
    print(f"[*] Launching DeepSeek Web API Server on http://{HOST}:{PORT}")
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        access_log=True
    )
