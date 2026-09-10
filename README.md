# ⚡ ds-gateway

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)

High-performance, lightweight OpenAI-compatible reverse gateway and terminal client for upstream DS conversational endpoints. 

Engineered with a sub-100ms WebAssembly challenge solver, persistent session state orchestration (preventing session spamming on upstream accounts), and full streaming support for both fast responses and deep reasoning tokens.

---

## 🌟 Highlights

- ⚡ **High-Speed Challenge Resolution**: Resolves cryptographic anti-bot proof-of-work (`DeepSeekHashV1`) in ~50–100ms using native WebAssembly via Node.js, with an autonomous pure Python Keccak-256 fallback when Node is unavailable.
- 🔒 **Sticky Session Orchestrator**:
  - Automatically binds consecutive requests to a **single persistent conversation thread**.
  - **Zero Workspace Clutter**: Eliminates the issue of creating dozens of orphaned chat sessions on the upstream web account.
  - Auto-healing resilience: Transparently initializes a replacement thread if the active remote session is archived or deleted.
- 🧠 **Dual Engine Support**:
  - `ds-chat` / `deepseek-chat` (Fast instant inference)
  - `ds-reasoner` / `deepseek-reasoner` (Chain-of-thought deep reasoning stream)
- 🔌 **Drop-in OpenAI Compatibility**: Works out of the box with official OpenAI SDKs (`openai-python`, `openai-node`), LangChain, LobeChat, LibreChat, NextChat, and OpenWebUI.
- 💻 **Zero-Dependency CLI**: Includes a self-contained interactive terminal client powered entirely by the Python Standard Library.
- 🐳 **Production Packaging**: Pre-configured Docker, Docker Compose, and Systemd deployment templates.

---

## 📐 Architecture

```
[ Client Applications / SDK / CLI ]
                │
                ▼  (Standard OpenAI Format)
    [ Reverse Proxy / CF Tunnel ]
                │
                ▼  (Port :8550)
      [ DS-Gateway Server ]
                │
                ├── Session State Store (data/session_state.json)
                ├── WASM Challenge Solver (solver.js + sha3_wasm_bg.wasm)
                └── Upstream Temporal Handshake Poller
                │
                ▼  (HTTP/2 Server-Sent Events)
       [ Upstream Endpoint ]
```

---

## 🚀 Quickstart

### 1. Terminal Client (Zero External Dependencies)
Connect directly to an active gateway instance:

```bash
# Clone the repository
git clone https://github.com/IndraYuda13/ds-gateway.git
cd ds-gateway

# Launch interactive terminal REPL
python3 run_cli.py

# Or execute a single prompt directly
python3 run_cli.py --no-think "Explain event-driven architecture in two sentences."
```

---

### 2. Self-Hosting (Local Machine or VPS)

#### Prerequisites
- Python 3.10+
- Node.js (recommended for ~50ms WASM challenge execution)

#### Installation
```bash
# 1. Clone repository
git clone https://github.com/IndraYuda13/ds-gateway.git
cd ds-gateway

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Open .env and set your DS_TOKEN (see Token Extraction below)

# 4. Start the gateway server
python3 main.py
```
The gateway will be operational on `http://0.0.0.0:8550`.

---

### 3. Docker Deployment

```bash
# Build and run container in detached mode
docker compose up -d --build

# View runtime logs
docker compose logs -f
```

---

### 4. Production Systemd Service

Deploy as a background daemon managed by systemd:

```bash
sudo cp systemd/deepseek-api.service /etc/systemd/system/ds-gateway.service
sudo systemctl daemon-reload
sudo systemctl enable --now ds-gateway

# Check service health
sudo systemctl status ds-gateway
```

---

## 🔑 Upstream Token Acquisition

1. Navigate to the upstream chat portal in your desktop browser.
2. Open Developer Tools (`F12` or `Ctrl + Shift + I`) and switch to the **Network** tab.
3. Send any prompt in the chat.
4. Locate the network request named `completion` or `users/current`.
5. Under **Request Headers**, copy the value of the `authorization` header:
   ```text
   Bearer qosOTpuhzASFMfNcNw...69ek
   ```
6. Paste the token into your `.env` file under `DS_TOKEN`.

---

## 📡 API Reference

### Health & Runtime State
```bash
curl -s http://localhost:8550/health
```

**Response:**
```json
{
  "status": "online",
  "service": "ds-gateway",
  "version": "1.1.0",
  "models": ["ds-chat", "ds-reasoner"],
  "active_session": {
    "id": "85ac9330-ba67-4d71-923e-3aad6b2f6e45",
    "parent_message_id": 4
  }
}
```

---

### OpenAI Chat Completions (`/v1/chat/completions`)

#### Standard Request (Non-Streaming)
```bash
curl -s -X POST http://localhost:8550/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds-reasoner",
    "messages": [
      {"role": "user", "content": "What is 17 * 23? Return answer only."}
    ]
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-9d81d6e4dac3",
  "object": "chat.completion",
  "created": 1789031609,
  "model": "ds-reasoner",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "391",
        "reasoning_content": "Compute 17 * 23: 17 * 20 = 340, 17 * 3 = 51. Sum is 391."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 3,
    "total_tokens": 3
  }
}
```

---

### Resetting Conversation Thread (`/chat/new`)
By default, the gateway maintains thread continuity across all incoming calls. To explicitly initialize a brand-new conversation thread:

```bash
curl -s -X POST http://localhost:8550/chat/new
```
Alternatively, include the header `X-New-Session: true` or pass `"new_session": true` in the JSON completion payload.

---

## 💻 SDK Integration Examples

### Python (Official `openai` Library)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8550/v1",
    api_key="none"  # Any non-empty string
)

# Deep reasoning model with streaming
stream = client.chat.completions.create(
    model="ds-reasoner",
    messages=[{"role": "user", "content": "Design a distributed rate-limiter architecture."}],
    stream=True
)

for chunk in stream:
    # Reasoning / thinking delta
    if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    # Output response delta
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Node.js (Official `openai` Library)
```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8550/v1",
  apiKey: "none"
});

async function main() {
  const stream = await client.chat.completions.create({
    model: "ds-chat",
    messages: [{ role: "user", content: "Summarize CAP theorem in 3 bullet points." }],
    stream: true
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
  console.log();
}

main();
```

---

## ⚙️ Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `DS_TOKEN` | Bearer authorization token from upstream web account | *(Required)* |
| `HOST` | Gateway listener host | `0.0.0.0` |
| `PORT` | Gateway listener port | `8550` |
| `PROXY_API_KEY` | *(Optional)* Access key for authenticating incoming API requests | `""` (Open) |
| `DATA_DIR` | Directory for persistent thread state storage | `./data` |

---

## 🛡️ Protocol Mechanics: Proof-of-Work Challenge

The upstream service protects sensitive completion routes with a proof-of-work challenge:
- Challenge parameters: `algorithm` (`DeepSeekHashV1`), `challenge` (hex), `salt`, `difficulty` (~144,000), and `expire_at`.
- Solving condition: Identify integer `nonce` satisfying:
  $$\text{DeepSeekHashV1}(\text{salt} + \text{"\_"} + \text{expire\_at} + \text{"\_"} + \text{nonce}) = \text{challenge}$$
- `DeepSeekHashV1` operates on a customized 64-bit Keccak-256 permutation with reversed 32-bit word ordering during absorb and squeeze phases.
- The included WASM solver compiles directly into memory and executes via V8/Node.js, achieving sub-100ms nonce discovery.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Developed for protocol interoperability research, developer productivity, and local pipeline integration.
