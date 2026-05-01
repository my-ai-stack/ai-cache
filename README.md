# ai-cache

> One line of code. Zero config. Caches every AI call automatically.

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

# First call:  $0.02 → API called, result cached
# Second call: $0.00 → instant cache hit
```

**90%+ cost savings** on repeated AI calls. CI pipelines, eval suites, RAG pipelines, team workflows — all benefit.

---

## The Hard Problems (Solved)

### 1. Hash Collision — Content-Addressed Keys

```
cache key = sha256(provider:model:sorted_request_json)
```

Every field that affects model output is in the key. SHA-256 collision resistance means identical content always maps to the same key, and different content (even by one character) maps to a different key.

### 2. Parameter Bleed — Param-Sensitive Keys

```python
# Same prompt, different temperature → two DIFFERENT cache entries
summarize(messages, temperature=0.7)  → key: openai:gpt-4o:a1b2c3...
summarize(messages, temperature=1.0)  → key: openai:gpt-4o:d4e5f6...

# Same prompt, different model → two DIFFERENT cache entries
summarize(messages, model="gpt-4o")      → key: openai:gpt-4o:a1b2c3...
summarize(messages, model="gpt-4o-mini") → key: openai:gpt-4o-mini:e5f6a7...
```

If `temperature`, `max_tokens`, `top_p`, `seed`, or `stream` change, the cache key changes. No silent corruption.

### 3. Message Ordering — Order-Invariant Keys

Messages are sorted by `(role, content)` before hashing, so the same conversation in different order produces the same key:

```python
# Both produce the SAME cache key
[{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}]
[{"role": "user", "content": "hi"}, {"role": "system", "content": "be helpful"}]
```

### 4. Stale Model Cache — Model Version in Key

Model is part of the cache key prefix. Switch models → different key space. No cross-contamination.

### 5. Cache Invalidation — TTL + LRU + Manual Purge

```python
@cached(ttl=3600, max_entries=100_000, max_size_mb=512)

# Explicit purge
cache.purge(provider="openai", model="gpt-4o")

# Pattern-based purge (SQLite backend)
from ai_cache import purge
purge(provider="openai", model="gpt-4o")
```

Eviction: oldest entries (by `created_at`) are deleted when limits are hit.

---

## Storage Backends

| Mode | How | Use Case |
|---|---|---|
| Solo | `~/.ai-cache/cache.db` (SQLite) | Default, zero setup |
| Team | `redis://team-server:6379/0` | Shared, one env var |
| CI | `/tmp/ai-cache` (memory) | Ephemeral, per-run |

```python
# Solo — just works, SQLite in ~/.ai-cache/
@cached(provider="openai", model="gpt-4o")

# Team — Redis
@cached(provider="openai", model="gpt-4o", backend="redis", redis_url="redis://cache.myteam.io:6379/0")

# CI — ephemeral memory, per-run
@cached(provider="openai", model="gpt-4o", backend="memory")
```

---

## Multi-Provider Support

```python
from ai_cache import CacheManager

cache = CacheManager()

@cache.openai(model="gpt-4o")
def openai_call(messages, ...):
    return openai.chat.completions.create(...)

@cache.anthropic(model="claude-sonnet-4-20250514")
def anthropic_call(messages, ...):
    return anthropic.messages.create(...)

@cache.groq(model="llama-3.3-70b-versatile")
def groq_call(messages, ...):
    return groq.chat.completions.create(...)
```

---

## Cache Management

```python
from ai_cache import stats, purge

# View stats
stats()
# {'hits': 142, 'misses': 23, 'hit_rate': 0.86, 'db_size_mb': 12.4}

# Purge all OpenAI entries
purge(provider="openai")

# Purge specific model
purge(provider="openai", model="gpt-4o")

# Explicit invalidation on a function
@cached(provider="openai", model="gpt-4o")
def summarize(...):
    ...

summarize.invalidate(messages=[{"role": "user", "content": "old context"}])
```

---

## Installation

```bash
pip install ai-cache
```

Or from source:

```bash
git clone https://github.com/my-ai-stack/ai-cache
cd ai-cache
pip install -e .
```

---

## The Math

| Scenario | Before | After | Saved |
|---|---|---|---|
| 100 eval runs, same prompts | $200 | $2 | 99% |
| 5-person team, shared prompts | $500/mo | $50/mo | 90% |
| CI daily builds (10 builds × 50 calls) | $1,500/mo | $150/mo | 90% |
| RAG pipeline, 10k doc chunks, high repeat | $2,000/mo | $200/mo | 90% |

---

## Compared to Alternatives

| Approach | Problem | ai-cache |
|---|---|---|
| API proxy server | Extra process, extra latency, data leaves your machine | Zero overhead, local by default |
| Cloud cache | Data leaves your infrastructure | SQLite never leaves `~/.ai-cache/` |
| Custom Redis | Build it yourself | `pip install ai-cache`, done |
| No cache | Pay full price every time | 90% cheaper on repeat calls |
| LangChain cache | Heavy, tied to LangChain ecosystem | Drop-in decorator, any codebase |
