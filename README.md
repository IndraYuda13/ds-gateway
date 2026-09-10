# 🚀 DeepSeek Web API Proxy & CLI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)

Proxy API berkinerja tinggi yang mem-bypass proteksi bot **chat.deepseek.com** dan mengubah akun web gratisan DeepSeek menjadi **API standar OpenAI-Compatible** (`/v1/chat/completions`). 

Dilengkapi dengan **WebAssembly PoW Solver** super cepat (~50–100ms), **Persistent Session State** (mencegah spam ratusan sesi chat di akun DeepSeek), serta dukungan **Real-Time Streaming** untuk token penalaran (*DeepSeek-R1 Thinking*) dan jawaban (*V3 Instant*).

---

## ✨ Fitur Unggulan

- ⚡ **Sub-100ms Anti-Bot Bypass**: Menyelesaikan challenge proof-of-work (`DeepSeekHashV1`) menggunakan modul WebAssembly native via Node.js bridge (dengan fallback murni Pure Python Keccak-256 tanpa dependensi luar).
- 🧠 **Dukungan Penuh DeepSeek-R1 & V3**:
  - `deepseek-chat` (Mode Instant respons cepat)
  - `deepseek-reasoner` (Mode Deep Thinking / Chain-of-Thought)
- 🔒 **Persistent Session Sticky Tracker**:
  - Semua percakapan otomatis diikat ke dalam **satu sesi thread yang sama** secara berkelanjutan (multi-turn memory).
  - **Zero Session Spam**: Tidak membuat puluhan chat baru di sidebar web DeepSeek kecuali dipicu secara eksplisit (`/new` atau `new_session=True`).
  - Auto-healing: Jika sesi di web terhapus/kadaluarsa, proxy otomatis membuat sesi pengganti tanpa memutus request.
- 🔌 **Drop-in Replacement OpenAI SDK**: Kompatibel 100% dengan library resmi `openai-python`, `openai-node`, LangChain, LibreChat, LobeChat, NextChat, maupun OpenWebUI.
- 🌐 **Cloudflare Tunnel Ready**: Siap diexpose ke domain publik via Cloudflare Tunnel tanpa perlu IP publik statis atau port forwarding.
- 💻 **Standalone Zero-Dependency CLI**: Disertai client terminal interaktif bertenaga Python Standard Library murni.

---

## 🏛️ Arsitektur Sistem

```
[ Client / App / CLI ] 
        │  (HTTP / OpenAI format)
        ▼
[ Cloudflare Tunnel / Reverse Proxy ]
        │  (https://deepseek.indrayuda.my.id)
        ▼
[ FastAPI Proxy Server (:8550) ]
        │
        ├── Session State Manager (session_state.json)
        ├── Dynamic Anti-Bot Solver (solver.js + sha3_wasm_bg.wasm)
        └── Token Poller (x-hif-leim query)
        │
        ▼  (HTTP/2 SSE Stream)
[ chat.deepseek.com Upstream ]
```

---

## 🚀 Quickstart

### 1. Cara Termudah: Gunakan Python CLI Langsung
Jika server proxy sudah online di server/tunnel, kamu cukup menjalankan script client standalone (tanpa install library apa pun):

```bash
# Clone repo
git clone https://github.com/IndraYuda13/deepseek-web-api.git
cd deepseek-web-api

# Masuk ke Terminal Chat Interaktif
python3 run_cli.py

# Atau kirim 1 pertanyaan langsung (Single-shot)
python3 run_cli.py --no-think "Jelaskan apa itu Docker dalam 2 kalimat."
```

---

### 2. Self-Hosting: Menjalankan Server Lokal / VPS

#### Prasyarat
- Python 3.10+
- Node.js (direkomendasikan untuk WASM PoW ~100ms)

#### Langkah Instalasi
```bash
# 1. Clone repository
git clone https://github.com/IndraYuda13/deepseek-web-api.git
cd deepseek-web-api

# 2. Buat virtual environment & install dependensi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Konfigurasi Token DeepSeek (.env)
cp .env.example .env
# Edit .env dan masukkan DEEPSEEK_TOKEN akunmu (lihat panduan ambil token di bawah)

# 4. Jalankan Server
python3 main.py
```
Server akan berjalan di `http://0.0.0.0:8550`.

---

### 3. Menjalankan via Docker & Docker Compose

```bash
# Build dan jalankan container di background
docker compose up -d --build

# Cek status log
docker compose logs -f
```

---

### 4. Menjalankan sebagai Daemon 24/7 (Systemd)

Salin file service yang disediakan:
```bash
sudo cp systemd/deepseek-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deepseek-api

# Cek status
sudo systemctl status deepseek-api
```

---

## 🔑 Cara Mengambil Token DeepSeek

1. Buka browser dan login ke **[chat.deepseek.com](https://chat.deepseek.com)**.
2. Buka Developer Tools (`F12` atau `Ctrl + Shift + I`) ➔ Masuk ke tab **Network**.
3. Kirim sembarang pesan di chat.
4. Cari request bernama `completion` atau `users/current`.
5. Di bagian **Request Headers**, salin isi header `authorization`:
   ```
   Bearer qosOTpuhzASFMfNcNw...69ek
   ```
6. Masukkan token tersebut ke file `.env` pada variabel `DEEPSEEK_TOKEN`.

---

## 📡 Dokumentasi Endpoint API

### 1. Health & Status Check
```bash
curl -s https://deepseek.indrayuda.my.id/health
```
**Response:**
```json
{
  "status": "online",
  "service": "DeepSeek Web API Proxy",
  "version": "1.1.0",
  "models": ["deepseek-chat", "deepseek-reasoner"],
  "active_session": {
    "id": "85ac9330-ba67-4d71-923e-3aad6b2f6e45",
    "parent_message_id": 4
  },
  "account": {
    "name": "Indra yuda adi saputra",
    "email": "lvt*****re@gmail.com"
  }
}
```

### 2. OpenAI Chat Completions (`/v1/chat/completions`)
Mendukung format standar OpenAI baik streaming (SSE) maupun non-streaming:

```bash
curl -s -X POST https://deepseek.indrayuda.my.id/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-reasoner",
    "messages": [
      {"role": "user", "content": "Berapa hasil 17 * 23?"}
    ]
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-9d81d6e4dac3",
  "object": "chat.completion",
  "created": 1789031609,
  "model": "deepseek-reasoner",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "391",
        "reasoning_content": "User asks 17 * 23. 17 * 20 = 340, 17 * 3 = 51. 340 + 51 = 391."
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

### 3. Reset / Buat Sesi Baru (`/chat/new`)
Secara default, proxy akan **tetap melanjutkan obrolan di sesi yang sama**. Jika kamu ingin membuka topik/sesi baru di web DeepSeek:

```bash
curl -s -X POST https://deepseek.indrayuda.my.id/chat/new
```
Atau tambahkan header `"X-New-Session: true"` / payload `"new_session": true` pada request completion berikutnya.

---

## 💻 Contoh Integrasi Code

### Python (OpenAI SDK Resmi)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://deepseek.indrayuda.my.id/v1",
    api_key="lemon"  # Isi sembarang string
)

# DeepSeek-R1 (Deep Reasoning)
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": "Rancang arsitektur microservices untuk e-commerce!"}
    ],
    stream=True
)

for chunk in response:
    # Token reasoning/thinking
    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    # Token jawaban final
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Node.js (OpenAI SDK)
```javascript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://deepseek.indrayuda.my.id/v1",
  apiKey: "lemon"
});

async function main() {
  const stream = await openai.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: "Halo dari Node.js!" }],
    stream: true,
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
  console.log();
}

main();
```

---

## ⚙️ Variabel Lingkungan (`.env`)

| Variabel | Deskripsi | Default |
| :--- | :--- | :--- |
| `DEEPSEEK_TOKEN` | Bearer token dari akun web chat.deepseek.com | *(Wajib diisi)* |
| `HOST` | Bind host listener | `0.0.0.0` |
| `PORT` | Bind port listener | `8550` |
| `PROXY_API_KEY` | *(Opsional)* Password API untuk membatasi akses publik | `""` (Terbuka) |
| `DATA_DIR` | Folder tempat menyimpan state sesi aktif (`session_state.json`) | `./data` |

---

## 🛡️ Reverse-Engineering Notes: Anti-Bot & PoW Challenge

DeepSeek Web menggunakan proteksi multi-lapis:
1. **Dynamic Challenge Verification (`x-hif-leim`)**: Token handshake temporal dari endpoint `https://hif-leim.deepseek.com/query` dengan TTL 600 detik.
2. **Proof-of-Work Challenge (`DeepSeekHashV1`)**:
   - Server memberikan difficulty ~144,000 dengan payload `challenge`, `salt`, `signature`, dan `expire_at`.
   - Formula: Cari integer `nonce` sehingga `DeepSeekHashV1(salt + "_" + expire_at + "_" + nonce) == challenge`.
   - Algoritma hashing adalah varian Keccak-256 (SHA3) dengan modifikasi word endian swap pada memory sponge. Modul WASM bawaan diekstraksi langsung dari bundle web DeepSeek untuk performa native maksimal.

---

## 📄 Lisensi

Didistribusikan di bawah Lisensi [MIT](LICENSE). Dibuat untuk tujuan edukasi, penelitian interoperabilitas protokol, dan otomatisasi produktivitas pribadi.
