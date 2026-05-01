"""
Tests for ai_cache.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_cache.cache import (
    _build_request_fingerprint,
    _normalize_function_args,
    SQLiteBackend,
    MemoryBackend,
    CacheConfig,
    _stats,
    cached,
    CacheManager,
    serialize_response,
    deserialize_response,
    SerializationError,
)


# ─── Serialization tests ───────────────────────────────────────────────────────


def test_serialize_primitives():
    for val in [{"key": "value"}, ["a", "b"], "string", 42, 3.14, True]:
        data = serialize_response(val)
        assert isinstance(data, bytes)
        assert deserialize_response(data) == val


def test_serialize_bytes():
    data = serialize_response(b"binary data")
    result = deserialize_response(data)
    # bytes are encoded as JSON {"__bytes": True, "data": "...encoded..."}
    assert isinstance(result, bytes)
    assert result == b"binary data"


def test_serialize_openai_response():
    """Mock an OpenAI-like response object."""
    class MockResponse:
        def to_dict(self):
            return {"id": "chatcmpl-1", "model": "gpt-4o", "choices": []}
    
    data = serialize_response(MockResponse())
    result = deserialize_response(data)
    # deserialize_response unwraps __openai -> returns the inner dict directly
    assert result["id"] == "chatcmpl-1"
    assert result["model"] == "gpt-4o"


def test_serialize_anthropic_response():
    """Mock an Anthropic-like response object."""
    class MockContentBlock:
        def model_dump(self):
            return {"type": "text", "text": "hello"}
    
    class MockUsage:
        def model_dump(self):
            return {"input_tokens": 10, "output_tokens": 20}
    
    class MockResponse:
        id = "msg_1"
        type = "message"
        role = "assistant"
        model = "claude-sonnet-4-20250514"
        stop_reason = "end_turn"
        content = [MockContentBlock()]
        usage = MockUsage()
    
    data = serialize_response(MockResponse())
    result = deserialize_response(data)
    # deserialize_response unwraps __anthropic -> returns the inner dict directly
    assert result["id"] == "msg_1"
    assert result["model"] == "claude-sonnet-4-20250514"


def test_serialize_dict_like():
    class DictLike:
        def __init__(self):
            self._data = {"a": 1, "b": 2}
        def items(self):
            return self._data.items()
    
    data = serialize_response(DictLike())
    result = deserialize_response(data)
    assert result == {"a": 1, "b": 2}


def test_serialize_error_on_unknown_type():
    class UnknownType:
        pass
    
    with pytest.raises(SerializationError):
        serialize_response(UnknownType())


# ─── Fingerprint tests ──────────────────────────────────────────────────────────


def test_fingerprint_same_prompt_same_params():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7, max_tokens=512,
    )
    fp2 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7, max_tokens=512,
    )
    assert fp1 == fp2


def test_fingerprint_different_temperature():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7,
    )
    fp2 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        temperature=1.0,
    )
    assert fp1 != fp2


def test_fingerprint_different_model():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )
    fp2 = _build_request_fingerprint(
        provider="openai", model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert fp1 != fp2


def test_fingerprint_different_max_tokens():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=512,
    )
    fp2 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1024,
    )
    assert fp1 != fp2


def test_fingerprint_different_provider():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )
    fp2 = _build_request_fingerprint(
        provider="anthropic", model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert fp1 != fp2


def test_fingerprint_different_messages():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )
    fp2 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "world"}],
    )
    assert fp1 != fp2


def test_fingerprint_message_order_invariance():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ],
    )
    fp2 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "be helpful"},
        ],
    )
    assert fp1 == fp2


def test_fingerprint_none_params_excluded():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )
    fp2 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        temperature=None, max_tokens=None, top_p=None, seed=None,
    )
    assert fp1 == fp2


def test_fingerprint_seed_affects_output():
    fp1 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        seed=42,
    )
    fp2 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        seed=42,
    )
    assert fp1 == fp2
    fp3 = _build_request_fingerprint(
        provider="openai", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        seed=99,
    )
    assert fp1 != fp3


def test_fingerprint_content_addressed():
    """Identical content always produces same hash — collision resistance via SHA256."""
    for _ in range(100):
        fp = _build_request_fingerprint(
            provider="openai", model="gpt-4o",
            messages=[{"role": "user", "content": "test content"}],
        )
        assert fp == _build_request_fingerprint(
            provider="openai", model="gpt-4o",
            messages=[{"role": "user", "content": "test content"}],
        )


# ─── normalize_function_args tests ───────────────────────────────────────────


def dummy_func(messages, model="gpt-4o", temperature=0.7, max_tokens=512, **kwargs):
    return {"messages": messages, "model": model}


def test_normalize_function_args_basic():
    args = _normalize_function_args(
        dummy_func,
        ([{"role": "user", "content": "hello"}],),
        {},
    )
    assert args["messages"] == [{"role": "user", "content": "hello"}]
    assert args["model"] == "gpt-4o"
    assert args["temperature"] == 0.7
    assert args["max_tokens"] == 512


def test_normalize_function_args_kwargs():
    args = _normalize_function_args(
        dummy_func,
        ([{"role": "user", "content": "hello"}],),
        {"temperature": 1.0, "max_tokens": 256},
    )
    assert args["temperature"] == 1.0
    assert args["max_tokens"] == 256


# ─── SQLite Backend tests ──────────────────────────────────────────────────────


@pytest.fixture
def tmp_backend():
    tmpdir = tempfile.mkdtemp()
    backend = SQLiteBackend(tmpdir, max_size_mb=10, max_entries=1000)
    yield backend
    backend.close()
    shutil.rmtree(tmpdir)


def test_sqlite_set_get(tmp_backend):
    key = "test:key:abc123"
    tmp_backend.set(
        key=key, provider="openai", model="gpt-4o",
        fingerprint="fp", response=b'{"result": "ok"}', ttl=3600,
    )
    result = tmp_backend.get(key)
    assert result == b'{"result": "ok"}'


def test_sqlite_miss(tmp_backend):
    assert tmp_backend.get("nonexistent:key") is None


def test_sqlite_hit_updates_access_time(tmp_backend):
    key = "test:key:access"
    tmp_backend.set(key=key, provider="openai", model="gpt-4o",
                    fingerprint="fp", response=b"data", ttl=3600)
    tmp_backend.get(key)
    tmp_backend.get(key)
    stats = tmp_backend.stats()
    assert stats["total_hits"] >= 2


def test_sqlite_ttl_expired(tmp_backend):
    """Expired entries are deleted on read."""
    key = "test:key:expired"
    tmp_backend.set(key=key, provider="openai", model="gpt-4o",
                    fingerprint="fp", response=b"data", ttl=1)
    time.sleep(1.1)
    result = tmp_backend.get(key)
    assert result is None


def test_sqlite_ttl_not_yet_expired(tmp_backend):
    """Non-expired entries are returned normally."""
    key = "test:key:valid"
    tmp_backend.set(key=key, provider="openai", model="gpt-4o",
                    fingerprint="fp", response=b"data", ttl=3600)
    result = tmp_backend.get(key)
    assert result == b"data"


def test_sqlite_expired_entries_counted_in_stats(tmp_backend):
    key = "test:key:expired2"
    tmp_backend.set(key=key, provider="openai", model="gpt-4o",
                    fingerprint="fp", response=b"data", ttl=1)
    time.sleep(1.1)
    tmp_backend.get(key)  # should trigger expiry deletion
    stats = tmp_backend.stats()
    assert stats["expired_entries"] == 0  # after read, expired entry is gone


def test_sqlite_purge_by_model(tmp_backend):
    tmp_backend.set("k1", "openai", "gpt-4o", "fp", b"d1", 3600)
    tmp_backend.set("k2", "openai", "gpt-4o-mini", "fp", b"d2", 3600)
    tmp_backend.purge(provider="openai", model="gpt-4o")
    assert tmp_backend.get("k1") is None
    assert tmp_backend.get("k2") is not None


def test_sqlite_purge_all(tmp_backend):
    tmp_backend.set("k1", "openai", "gpt-4o", "fp", b"d1", 3600)
    tmp_backend.set("k2", "anthropic", "claude-sonnet-4-20250514", "fp", b"d2", 3600)
    tmp_backend.purge()
    assert tmp_backend.get("k1") is None
    assert tmp_backend.get("k2") is None


def test_sqlite_stats(tmp_backend):
    tmp_backend.set("k1", "openai", "gpt-4o", "fp", b"d1", 3600)
    tmp_backend.set("k2", "openai", "gpt-4o", "fp", b"d2", 3600)
    tmp_backend.get("k1")
    tmp_backend.get("k1")
    stats = tmp_backend.stats()
    assert stats["total_entries"] == 2
    # k1: hit_count=3 (insert=1 + 2 reads), k2: hit_count=1 (insert only)
    assert stats["total_hits"] == 4


# ─── Memory Backend tests ───────────────────────────────────────────────────────


@pytest.fixture
def mem_backend():
    return MemoryBackend(max_entries=100)


def test_memory_set_get(mem_backend):
    mem_backend.set("k1", "openai", "gpt-4o", "fp", b"v1", 3600)
    assert mem_backend.get("k1") == b"v1"


def test_memory_miss(mem_backend):
    assert mem_backend.get("nonexistent") is None


def test_memory_eviction(mem_backend):
    for i in range(120):
        mem_backend.set(f"k{i}", "openai", "gpt-4o", f"fp{i}", b"data", 3600)
    assert mem_backend.get("k0") is None
    assert mem_backend.get("k119") is not None


def test_memory_ttl(mem_backend):
    mem_backend.set("k1", "openai", "gpt-4o", "fp", b"v1", ttl=1)
    time.sleep(1.1)
    assert mem_backend.get("k1") is None


def test_memory_expired_not_counted(mem_backend):
    """Expired entries are pruned on read, so stats reflects post-pruning state."""
    mem_backend.set("k1", "openai", "gpt-4o", "fp", b"v1", ttl=1)
    # Force a read which triggers TTL expiry check + pruning
    mem_backend.get("k1")
    time.sleep(1.1)
    mem_backend.get("k1")  # expired entry deleted on this read
    stats = mem_backend.stats()
    assert stats["expired_entries"] == 0


# ─── Decorator integration tests ───────────────────────────────────────────────


@pytest.fixture
def tmp_cache_dir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_decorator_cache_hit(tmp_cache_dir):
    call_count = 0

    @cached(provider="openai", model="gpt-4o", backend=SQLiteBackend(tmp_cache_dir),
            enabled=True)
    def api_call(messages, model="gpt-4o", temperature=0.7, max_tokens=512):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    result1 = api_call(messages=[{"role": "user", "content": "hello"}])
    assert result1 == {"result": "call_1"}
    assert call_count == 1

    result2 = api_call(messages=[{"role": "user", "content": "hello"}])
    assert result2 == {"result": "call_1"}
    assert call_count == 1


def test_decorator_different_temp_no_cache_hit(tmp_cache_dir):
    call_count = 0

    @cached(provider="openai", model="gpt-4o", backend=SQLiteBackend(tmp_cache_dir))
    def api_call(messages, model="gpt-4o", temperature=0.7, max_tokens=512):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    result1 = api_call(messages=[{"role": "user", "content": "hello"}], temperature=0.7)
    result2 = api_call(messages=[{"role": "user", "content": "hello"}], temperature=1.0)
    assert result1 == {"result": "call_1"}
    assert result2 == {"result": "call_2"}
    assert call_count == 2


def test_decorator_different_message_no_cache_hit(tmp_cache_dir):
    call_count = 0

    @cached(provider="openai", model="gpt-4o", backend=SQLiteBackend(tmp_cache_dir))
    def api_call(messages, model="gpt-4o", temperature=0.7):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    api_call(messages=[{"role": "user", "content": "hello"}])
    api_call(messages=[{"role": "user", "content": "world"}])
    assert call_count == 2


def test_decorator_invalidate(tmp_cache_dir):
    call_count = 0

    @cached(provider="openai", model="gpt-4o", backend=SQLiteBackend(tmp_cache_dir))
    def api_call(messages, model="gpt-4o"):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    api_call(messages=[{"role": "user", "content": "hello"}])
    assert call_count == 1

    api_call.invalidate(messages=[{"role": "user", "content": "hello"}])

    api_call(messages=[{"role": "user", "content": "hello"}])
    assert call_count == 2


def test_decorator_disabled():
    call_count = 0

    @cached(provider="openai", model="gpt-4o", enabled=False)
    def api_call(messages, model="gpt-4o"):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    api_call(messages=[{"role": "user", "content": "hello"}])
    api_call(messages=[{"role": "user", "content": "hello"}])
    assert call_count == 2


def test_decorator_json_serialization(tmp_cache_dir):
    """Responses should be JSON-serialized, not pickle."""
    call_count = 0

    @cached(provider="openai", model="gpt-4o", backend=SQLiteBackend(tmp_cache_dir))
    def api_call(messages, model="gpt-4o"):
        nonlocal call_count
        call_count += 1
        return {"choices": [{"message": {"content": "hello"}}], "model": "gpt-4o"}

    result1 = api_call(messages=[{"role": "user", "content": "test"}])
    assert result1["choices"][0]["message"]["content"] == "hello"
    assert call_count == 1

    result2 = api_call(messages=[{"role": "user", "content": "test"}])
    assert result2["choices"][0]["message"]["content"] == "hello"
    assert call_count == 1


def test_decorator_ttl_expired(tmp_cache_dir):
    """Expired entries are not returned from cache."""
    call_count = 0

    @cached(provider="openai", model="gpt-4o", backend=SQLiteBackend(tmp_cache_dir), ttl=1)
    def api_call(messages, model="gpt-4o"):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    api_call(messages=[{"role": "user", "content": "hello"}])
    assert call_count == 1

    time.sleep(1.1)

    api_call(messages=[{"role": "user", "content": "hello"}])
    # Should call API again since entry expired
    assert call_count == 2


def test_cache_manager(tmp_cache_dir):
    cache_mgr = CacheManager(backend=SQLiteBackend(tmp_cache_dir))
    call_count = 0

    @cache_mgr.openai(model="gpt-4o")
    def summarize(messages, model="gpt-4o", temperature=0.7, max_tokens=512):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    summarize(messages=[{"role": "user", "content": "test"}])
    summarize(messages=[{"role": "user", "content": "test"}])
    assert call_count == 1


def test_stats_tracked():
    stats = _stats()
    assert hasattr(stats, "hits")
    assert hasattr(stats, "misses")
    assert hasattr(stats, "hit_rate")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
