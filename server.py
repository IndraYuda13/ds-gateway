#!/usr/bin/env python3
"""
Backward compatibility launcher for deepseek-api.service.
Delegates directly to main.py
"""
import sys
import uvicorn
from app.main import app
from app.config import HOST, PORT

if __name__ == "__main__":
    print(f"Starting DeepSeek API server on port {PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)
