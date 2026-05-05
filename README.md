# ai-cache

> One line of code. Zero config. Caches every AI call automatically.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/ai-cache)](https://pypi.org/project/ai-cache/)
[![Downloads](https://img.shields.io/pypi/dm/ai-cache)](https://pypi.org/project/ai-cache/)
[![Stars](https://img.shields.io/github/stars/my-ai-stack/ai-cache)](https://github.com/my-ai-stack/ai-cache/stargazers)

## 🚡 Features

| Feature | Description |
|---------|-------------|
| 🔒 **Zero Config** | One decorator, works immediately |
| 💾 **Multi-Provider** | OpenAI, Anthropic, Groq, Together AI |
| 📊 **Smart Keys** | Content-addressed, param-sensitive |
| 🚡 **Cost Savings** | 90%+ reduction on repeated calls |
| 🧠 **LRU + TTL** | Automatic cache invalidation |
| 🌐 **Team Ready** | Redis backend for shared cache |

## 🚀 Quick Start

### Installation
```bash
pip install ai-cache
```

### Usage
```python
from ai_cache import cached

@cached(provider="openai", model="gpt-4o")
def summarize(messages, temperature=0.7, max_tokens=512):
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# First call: $0.02 → API called, result cached
# Second call: $0.00 → instant cache hit
```

## 🏗️ Storage Backends

| Mode | How | Use Case |
|------|-----|----------|
| Solo | `~/.ai-cache/cache.db` (SQLite) | Default, zero setup |
| Team | `redis://team-server:6379/0` | Shared, one env var |
| CI | `/tmp/ai-cache` (memory) | Ephemeral, per-run |

## 📦 Architecture

```
Cache Key = SHA256(provider:model:sorted_request_json)
```

- **Hash Collision Protection**: SHA-256 collision resistance
- **Parameter Bleed Protection**: Temp, max_tokens, top_p in key
- **Message Ordering Protection**: Sorted by (role, content) before hash
- **Stale Model Protection**: Model version in key prefix

## 🔧 Cache Management

```python
from ai_cache import stats, purge

# View stats
stats()
# {'hits': 142, 'misses': 23, 'hit_rate': 0.86, 'db_size_mb': 12.4}

# Purge all OpenAI entries
purge(provider="openai")

# Purge specific model
purge(provider="openai", model="gpt-4o")
```

## 📁 Project Structure

```
ai-cache/
├── ai_cache/           # Core cache library
│   ├── __init__.py
│   └── cache.py
├── tests/
├── README.md
├── pyproject.toml
└── LICENSE
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## ⭐ Support

Star the repo if you find it useful!

---

**Built with ❤️ by [my-ai-stack](https://github.com/my-ai-stack)**
