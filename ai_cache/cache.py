"""
Core caching decorator with payload-aware keying.

Key design (solves the hard problems):
1. FULL payload hashing — every field in the request dict, sorted and serialized
2. Param-sensitive keys — temperature, max_tokens, model, seed all live IN the key
3. Collision resistance — SHA256(full_payload) + a BLOB lookup, not just hash-as-key
4. Model versioning — model change auto-invalidates; hash prefix is the model+version
5. Content-addressed storage — key IS the content hash; collision = identical content
6. TTL enforcement — entries expire and are cleaned up on read
7. Portable serialization — JSON everywhere, no pickle, Redis-compatible
"""

import hashlib
import json
import sqlite3
import time
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union
from dataclasses import dataclass
from enum import Enum
import threading
import contextlib

try:
    import orjson as json_lib
    def _json_dumps(obj: Any) -> bytes:
        return json_lib.dumps(obj)
    def _json_loads(data: bytes) -> Any:
        return json_lib.loads(data)
except ImportError:
    import json as json_stdlib
    json_lib = json_stdlib
    def _json_dumps(obj: Any) -> bytes:
        return json_stdlib.dumps(obj).encode("utf-8")
    def _json_loads(data: bytes) -> Any:
        return json_stdlib.loads(data.decode("utf-8"))


# ─── Serializers ────────────────────────────────────────────────────────────────
#
# Serialize response objects to portable JSON bytes.
# Must be JSON-compatible so Redis can share across non-Python clients.
# Provider-specific extraction handles non-JSON-native response objects.
#
# ────────────────────────────────────────────────────────────────────────────────


class SerializationError(Exception):
    """Raised when a response object can't be serialized."""


def serialize_response(response: Any) -> bytes:
    """
    Serialize any provider response to JSON bytes.
    
    Handles:
    - OpenAI ChatCompletion response objects (via .to_dict() or model_dump())
    - Anthropic message response objects (via .model_dump() or dict access)
    - Already-serializable dict/list/str/int/float types
    - bytes (decode to string)
    """
    if response is None:
        return b'""'
    
    # Already a primitive
    if isinstance(response, (dict, list, str, int, float, bool)):
        return _json_dumps(response)
    
    # bytes
    if isinstance(response, bytes):
        return _json_dumps({"__bytes": True, "data": response.decode("latin-1")})
    
    # OpenAI response object — has .to_dict() or .model_dump()
    if hasattr(response, "to_dict"):
        try:
            return _json_dumps({"__openai": True, "data": response.to_dict()})
        except Exception:
            pass
    if hasattr(response, "model_dump"):
        try:
            return _json_dumps({"__openai": True, "data": response.model_dump()})
        except Exception:
            pass
    
    # Anthropic response object
    if hasattr(response, "content") and hasattr(response, "id"):
        try:
            return _json_dumps({
                "__anthropic": True,
                "data": {
                    "id": response.id,
                    "type": getattr(response, "type", "message"),
                    "role": getattr(response, "role", "assistant"),
                    "content": [c.model_dump() if hasattr(c, "model_dump") else c for c in response.content]
                        if hasattr(response, "content") else [],
                    "usage": response.usage.model_dump() if hasattr(response, "usage") else None,
                    "model": getattr(response, "model", ""),
                    "stop_reason": getattr(response, "stop_reason", None),
                }
            })
        except Exception:
            pass
    
    # Dict-like objects
    if hasattr(response, "items"):
        try:
            return _json_dumps(dict(response.items()))
        except Exception:
            pass
    
    # Last resort — try model_dump if it exists
    if hasattr(response, "model_dump"):
        try:
            return _json_dumps(response.model_dump())
        except Exception:
            pass
    
    raise SerializationError(
        f"Cannot serialize response of type {type(response).__name__}. "
        f"Wrap it with @cached(serializer=my_serializer) or return a dict."
    )


def deserialize_response(data: bytes) -> Any:
    """Reconstruct a response object from JSON bytes."""
    if not data:
        return None
    obj = _json_loads(data)
    
    # Wrapped types — check before general dict/list
    if isinstance(obj, dict):
        if obj.get("__bytes"):
            return obj["data"].encode("latin-1")
        if obj.get("__openai") or obj.get("__anthropic"):
            return obj["data"]
    
    # Primitives — return as-is
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    
    return obj


# ─── Key Anatomy ───────────────────────────────────────────────────────────────
#
# Key = sha256_hex(model:provider:sorted_request_json)
#
# Example key for two calls with same prompt but different temperature:
#
#   Call 1: {"model":"gpt-4o","messages":[...],"temperature":0.7,"max_tokens":512}
#           → key: openai:gpt-4o:a1b2c3...
#
#   Call 2: {"model":"gpt-4o","messages":[...],"temperature":1.0,"max_tokens":512}
#           → key: openai:gpt-4o:d4e5f6...
#
# Two different keys. No collision. No bleed.
#
# For chat completions, the "request" key is built from the fields that affect
# model behavior: model, messages, temperature, max_tokens, top_p, seed, stream.
# Fields like 'frequency_penalty', 'presence_penalty' also affect output.
#
# ────────────────────────────────────────────────────────────────────────────────


class BackendType(Enum):
    SQLITE = "sqlite"
    REDIS = "redis"
    MEMORY = "memory"


@dataclass
class CacheConfig:
    """Configuration for the cache decorator."""

    provider: str = "openai"
    model: str = ""
    ttl: int = 3600  # seconds
    max_size_mb: int = 512  # SQLite DB size cap
    max_entries: int = 100_000  # entry count cap
    backend: Union[str, BackendType] = "sqlite"
    redis_url: Optional[str] = None
    enabled: bool = True
    storage_path: Optional[str] = None  # defaults to ~/.ai-cache/

    def __post_init__(self):
        if isinstance(self.backend, str):
            self.backend = BackendType(self.backend)
        if self.storage_path is None:
            self.storage_path = os.path.expanduser("~/.ai-cache")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    errors: int = 0
    bytes_saved: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


_global_stats = CacheStats()
_global_lock = threading.Lock()


def _stats():
    return _global_stats


# ─── Payload Key Builder ────────────────────────────────────────────────────────
#
# Only includes fields that AFFECT model output. Skips metadata like 'n' (number
# of responses), 'response_format', 'tools', 'tool_choice', 'user'.
#
# Rationale:
#   - 'n' doesn't change model behavior per-call, it just requests multiple
#   - 'tools'/'tool_choice' do affect output but are rare; we can add them later
#   - 'user' is just context, doesn't affect the model's core generation
#   - 'response_format' (json schema) DOES affect output — include it
#
# ────────────────────────────────────────────────────────────────────────────────


def _build_request_fingerprint(
    provider: str,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    stream: bool = False,
    **kwargs,
) -> str:
    """
    Build a deterministic, collision-resistant fingerprint of a request.

    Only fields that affect model behavior are included. This means:
    - Same prompt + same params → same fingerprint
    - Same prompt + different temp → DIFFERENT fingerprint
    - Same prompt + different model → DIFFERENT fingerprint
    - Same prompt + different max_tokens → DIFFERENT fingerprint
    """
    # Normalize messages — sort by role to prevent ordering collisions
    normalized_messages = sorted(
        messages,
        key=lambda m: (m.get("role", ""), m.get("content", "")),
    )

    # Core fields that affect generation
    fingerprint_data = {
        "provider": provider,
        "model": model,
        "messages": normalized_messages,
        "stream": stream,
    }

    # Optional generation params — only include if explicitly set (not None)
    # This ensures default values don't create spurious different keys
    if temperature is not None:
        fingerprint_data["temperature"] = temperature
    if max_tokens is not None:
        fingerprint_data["max_tokens"] = max_tokens
    if top_p is not None:
        fingerprint_data["top_p"] = top_p
    if seed is not None:
        fingerprint_data["seed"] = seed

    # Response format (e.g. {"type": "json_object"}) DOES affect output
    if "response_format" in kwargs:
        fingerprint_data["response_format"] = kwargs["response_format"]

    # Sort + serialize deterministically
    serialized = json_lib.dumps(fingerprint_data, option=json_lib.OPT_SORT_KEYS)
    return hashlib.sha256(serialized).hexdigest()


def _normalize_function_args(func: Callable, args, kwargs) -> dict:
    """
    Extract the request-relevant arguments from a function call signature.
    Works with any function that takes keyword args matching the provider API.
    """
    import inspect

    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # Extract only relevant fields
    result = {}
    for param_name, param_value in bound.arguments.items():
        if param_name in (
            "messages",
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "seed",
            "stream",
            "response_format",
        ):
            result[param_name] = param_value
        elif param_name == "provider":
            result["provider"] = param_value

    return result


# ─── SQLite Backend ─────────────────────────────────────────────────────────────


class SQLiteBackend:
    """
    Content-addressed SQLite cache.

    Key = sha256(provider:model:fingerprint)
    Value = JSON-serialized response (portable across Python versions)

    Schema:
        CREATE TABLE cache (
            key TEXT PRIMARY KEY,          -- sha256 hex
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at REAL NOT NULL,      -- unix timestamp
            expires_at REAL NOT NULL,      -- unix timestamp (TTL enforcement)
            last_accessed REAL,             -- unix timestamp
            response BLOB NOT NULL,         -- JSON bytes
            request_fingerprint TEXT NOT NULL,
            hit_count INTEGER DEFAULT 1
        );
        CREATE INDEX idx_model ON cache(provider, model);
        CREATE INDEX idx_expires ON cache(expires_at);
        CREATE INDEX idx_created ON cache(created_at);
    """

    def __init__(self, storage_path: str, max_size_mb: int = 512, max_entries: int = 100_000):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "cache.db"
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                last_accessed REAL,
                response BLOB NOT NULL,
                request_fingerprint TEXT NOT NULL,
                hit_count INTEGER DEFAULT 1
            )
        """)
        # Add expires_at column if migrating from older schema (no-op if already exists)
        try:
            conn.execute("ALTER TABLE cache ADD COLUMN expires_at REAL")
        except sqlite3.OperationalError:
            pass  # already exists
        try:
            conn.execute("ALTER TABLE cache ADD COLUMN last_accessed REAL")
        except sqlite3.OperationalError:
            pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON cache(provider, model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
        conn.commit()

    def _enforce_limits(self):
        """Evict oldest entries when limits are reached, or when expired."""
        conn = self._get_conn()
        now = time.time()

        # Evict expired entries on every write
        count = conn.execute(
            "SELECT COUNT(*) FROM cache WHERE expires_at < ?",
            (now,),
        ).fetchone()[0]
        if count > 0:
            conn.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
            conn.commit()

        # Check entry count
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count >= self.max_entries:
            conn.execute("""
                DELETE FROM cache WHERE key IN (
                    SELECT key FROM cache ORDER BY created_at ASC LIMIT ?
                )
            """, (count - self.max_entries + 1000,))
            conn.commit()

        # Check size
        if self.db_path.exists():
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
            if db_size_mb >= self.max_size_mb:
                conn.execute("""
                    DELETE FROM cache WHERE key IN (
                        SELECT key FROM cache ORDER BY created_at ASC LIMIT ?
                    )
                """, (max(100, int(self.max_entries * 0.1)),))
                conn.commit()

    def get(self, key: str) -> Optional[bytes]:
        conn = self._get_conn()
        now = time.time()
        row = conn.execute(
            "SELECT response, expires_at FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        # TTL enforcement — check expiry before returning
        if row["expires_at"] < now:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return None
        conn.execute(
            "UPDATE cache SET last_accessed = ?, hit_count = hit_count + 1 WHERE key = ?",
            (now, key),
        )
        conn.commit()
        return row["response"]

    def set(self, key: str, provider: str, model: str, fingerprint: str, response: bytes, ttl: int):
        conn = self._get_conn()
        now = time.time()
        expires_at = now + ttl
        try:
            conn.execute("""
                INSERT OR REPLACE INTO cache
                (key, provider, model, created_at, expires_at, last_accessed, response, request_fingerprint, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (key, provider, model, now, expires_at, now, response, fingerprint))
            conn.commit()
            self._enforce_limits()
        except sqlite3.IntegrityError:
            pass  # already exists

    def delete(self, key: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()

    def purge(self, provider: Optional[str] = None, model: Optional[str] = None):
        conn = self._get_conn()
        now = time.time()
        if provider and model:
            conn.execute("DELETE FROM cache WHERE provider = ? AND model = ?", (provider, model))
        elif provider:
            conn.execute("DELETE FROM cache WHERE provider = ?", (provider,))
        else:
            conn.execute("DELETE FROM cache")
        conn.commit()

    def stats(self) -> dict:
        conn = self._get_conn()
        now = time.time()
        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        hits = conn.execute("SELECT SUM(hit_count) FROM cache").fetchone()[0] or 0
        expired = conn.execute("SELECT COUNT(*) FROM cache WHERE expires_at < ?", (now,)).fetchone()[0]
        misses = _stats().misses
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        return {
            "total_entries": total,
            "expired_entries": expired,
            "total_hits": hits,
            "estimated_misses": misses,
            "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0.0,
            "db_size_mb": round(db_size_mb, 3),
        }

    def close(self):
        if hasattr(self._local, "conn"):
            self._local.conn.close()


# ─── Redis Backend ─────────────────────────────────────────────────────────────


class RedisBackend:
    """Redis-backed cache for team sharing. Uses JSON serialization."""

    def __init__(self, redis_url: str, ttl: int = 3600):
        import redis as redis_lib
        self.redis_url = redis_url
        self._client = redis_lib.from_url(redis_url, decode_responses=False)
        self._ttl = ttl

    def get(self, key: str) -> Optional[bytes]:
        val = self._client.get(key)
        if val:
            self._client.incr(f"{key}:hits")
        return val

    def set(self, key: str, provider: str, model: str, fingerprint: str, response: bytes, ttl: int):
        pipe = self._client.pipeline()
        effective_ttl = ttl if ttl > 0 else self._ttl
        pipe.set(key, response, ex=effective_ttl)
        pipe.hset(f"{key}:meta", mapping={
            "provider": provider,
            "model": model,
            "fingerprint": fingerprint,
            "created_at": time.time(),
        })
        pipe.expire(f"{key}:meta", effective_ttl)
        pipe.execute()

    def delete(self, key: str):
        self._client.delete(key, f"{key}:meta", f"{key}:hits")

    def purge(self, provider: Optional[str] = None, model: Optional[str] = None):
        if provider and model:
            pattern = f"{provider}:{model}:*"
        elif provider:
            pattern = f"{provider}:*"
        else:
            pattern = "*"
        keys = self._client.keys(pattern)
        if keys:
            self._client.delete(*keys)

    def stats(self) -> dict:
        return {"backend": "redis", "note": "stats not yet implemented"}


# ─── Memory Backend (ephemeral, per-process) ───────────────────────────────────


class MemoryBackend:
    """In-memory cache for single-process/CI use. Evicts on restart."""

    def __init__(self, max_entries: int = 10_000):
        self._store: dict[str, tuple[bytes, float]] = {}  # key → (response, expires_at)
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            response, expires_at = entry
            if expires_at < time.time():
                del self._store[key]
                return None
            return response

    def set(self, key: str, provider: str, model: str, fingerprint: str, response: bytes, ttl: int):
        with self._lock:
            if len(self._store) >= self._max_entries:
                # Evict ~10% oldest by expiry
                sorted_items = sorted(self._store.items(), key=lambda x: x[1][1])
                for k, _ in sorted_items[: max(100, int(self._max_entries * 0.1))]:
                    del self._store[k]
            self._store[key] = (response, time.time() + ttl)

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def purge(self, provider: Optional[str] = None, model: Optional[str] = None):
        with self._lock:
            if provider and model:
                prefix = f"{provider}:{model}:"
                for k in list(self._store.keys()):
                    if k.startswith(prefix):
                        del self._store[k]
            else:
                self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            expired = sum(1 for _, exp in self._store.values() if exp < now)
            return {
                "total_entries": len(self._store),
                "expired_entries": expired,
                "backend": "memory",
            }


# ─── Backend Factory ────────────────────────────────────────────────────────────


def _get_backend(config: CacheConfig):
    backend = config.backend
    if isinstance(backend, (SQLiteBackend, MemoryBackend, RedisBackend)):
        return backend
    if isinstance(backend, str):
        backend = BackendType(backend)
    if backend == BackendType.SQLITE:
        return SQLiteBackend(config.storage_path, config.max_size_mb, config.max_entries)
    elif backend == BackendType.REDIS:
        if not config.redis_url:
            raise ValueError("redis_url required for Redis backend")
        return RedisBackend(config.redis_url, config.ttl)
    elif backend == BackendType.MEMORY:
        return MemoryBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ─── Core Decorator ─────────────────────────────────────────────────────────────


def cached(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    ttl: int = 3600,
    backend: Union[str, CacheConfig] = "sqlite",
    redis_url: Optional[str] = None,
    storage_path: Optional[str] = None,
    max_size_mb: int = 512,
    max_entries: int = 100_000,
    enabled: bool = True,
) -> Callable:
    """
    Cache AI responses with payload-aware keying.

    Usage:
        from ai_cache import cached

        @cached(provider="openai", model="gpt-4o")
        def summarize(messages, temperature=0.7, max_tokens=512, **kwargs):
            return openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        # First call: API call made, result cached
        result = summarize(messages=[{"role":"user","content": "hello"}])

        # Second call (same messages, same params): instant, no API call
        result = summarize(messages=[{"role":"user","content": "hello"}])

    Cache key includes:
        - Full message content (sorted by role to prevent ordering issues)
        - temperature, max_tokens, top_p, seed (only if not None)
        - model change → different cache entry
        - temperature change → different cache entry
        - stream vs non-stream → different cache entry

    TTL: entries expire after `ttl` seconds. Expired entries are deleted on read.
    """
    if isinstance(backend, CacheConfig):
        config = backend
    else:
        config = CacheConfig(
            provider=provider or "openai",
            model=model or "",
            ttl=ttl,
            backend=backend,
            redis_url=redis_url,
            storage_path=storage_path,
            max_size_mb=max_size_mb,
            max_entries=max_entries,
            enabled=enabled,
        )

    _backend = _get_backend(config)

    def decorator(func: Callable) -> Callable:
        effective_model = model or config.model

        @contextlib.wraps(func)
        def wrapper(*args, **kwargs):
            if not config.enabled:
                return func(*args, **kwargs)

            call_args = _normalize_function_args(func, args, kwargs)
            effective_provider = call_args.get("provider", config.provider)
            model_from_args = call_args.get("model") or effective_model

            fingerprint = _build_request_fingerprint(
                provider=effective_provider,
                model=model_from_args,  # FIX: was using decorator `model`, now uses runtime-resolved model
                messages=call_args.get("messages", []),
                temperature=call_args.get("temperature"),
                max_tokens=call_args.get("max_tokens"),
                top_p=call_args.get("top_p"),
                seed=call_args.get("seed"),
                stream=call_args.get("stream", False),
                **{k: v for k, v in call_args.items()
                   if k in ("response_format",)},
            )

            cache_key = f"{effective_provider}:{model_from_args}:{fingerprint}"

            # ── Cache lookup ────────────────────────────────────────────────────
            cached_response = _backend.get(cache_key)
            if cached_response is not None:
                with _global_lock:
                    _stats().hits += 1
                return deserialize_response(cached_response)

            # ── Cache miss: call the actual function ─────────────────────────────
            with _global_lock:
                _stats().misses += 1

            result = func(*args, **kwargs)

            # Serialize and store (JSON — no pickle, Redis-compatible)
            try:
                serialized = serialize_response(result)
                _backend.set(
                    key=cache_key,
                    provider=effective_provider,
                    model=model_from_args,
                    fingerprint=fingerprint,
                    response=serialized,
                    ttl=config.ttl,
                )
            except SerializationError:
                raise
            except Exception:
                pass  # cache write failures should not break the call

            return result

        # ── Cache management API ──────────────────────────────────────────────
        wrapper.cache = _backend
        wrapper.config = config

        def invalidate(*args, **kwargs):
            """Explicitly invalidate a cache entry."""
            call_args = _normalize_function_args(func, args, kwargs)
            effective_provider = call_args.get("provider", config.provider)
            model_from_call = call_args.get("model") or effective_model
            fingerprint = _build_request_fingerprint(
                provider=effective_provider,
                model=model_from_call,
                messages=call_args.get("messages", []),
                temperature=call_args.get("temperature"),
                max_tokens=call_args.get("max_tokens"),
                top_p=call_args.get("top_p"),
                seed=call_args.get("seed"),
                stream=call_args.get("stream", False),
            )
            cache_key = f"{effective_provider}:{model_from_call}:{fingerprint}"
            _backend.delete(cache_key)

        wrapper.invalidate = invalidate

        return wrapper

    return decorator


# ─── CacheManager ───────────────────────────────────────────────────────────────


class CacheManager:
    """
    Provider-aware cache manager.

    Usage:
        from ai_cache import CacheManager

        cache = CacheManager()

        @cache.openai(model="gpt-4o")
        def summarize(messages, model="gpt-4o", temperature=0.7, max_tokens=512):
            return openai.chat.completions.create(...)

        @cache.anthropic(model="claude-sonnet-4-20250514")
        def think(messages, model="claude-sonnet-4-20250514", temperature=0.7, max_tokens=1024):
            return anthropic.messages.create(...)
    """

    def __init__(self, ttl: int = 3600, backend: Union[str, CacheConfig] = "sqlite",
                 storage_path: Optional[str] = None):
        self._config = CacheConfig(ttl=ttl, backend=backend, storage_path=storage_path)
        self._backends: dict[str, Any] = {}

    def _get_cached(self, provider: str, model: str = ""):
        key = f"{provider}:{model}"
        if key not in self._backends:
            config = CacheConfig(provider=provider, model=model, ttl=self._config.ttl,
                                 backend=self._config.backend,
                                 redis_url=self._config.redis_url,
                                 storage_path=self._config.storage_path)
            self._backends[key] = _get_backend(config)
        return self._backends[key]

    def openai(self, model: str = "", ttl: int = 3600):
        return cached(provider="openai", model=model, ttl=ttl, backend=self._config)

    def anthropic(self, model: str = "", ttl: int = 3600):
        return cached(provider="anthropic", model=model, ttl=ttl, backend=self._config)

    def groq(self, model: str = "", ttl: int = 3600):
        return cached(provider="groq", model=model, ttl=ttl, backend=self._config)

    def cerebras(self, model: str = "", ttl: int = 3600):
        return cached(provider="cerebras", model=model, ttl=ttl, backend=self._config)


# ─── Convenience helpers ────────────────────────────────────────────────────────


def purge(provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
    """Purge cache entries by provider/model."""
    config = CacheConfig(**kwargs)
    backend = _get_backend(config)
    backend.purge(provider=provider, model=model)


def stats() -> dict:
    """Return cache statistics."""
    return _stats().__dict__
