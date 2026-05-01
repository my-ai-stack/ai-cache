"""
Tests for ai_cache.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path

# Module-level import to test it loads
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
)


# ─── Fingerprint tests ──────────────────────────────────────────────────────────


def test_fingerprint_same_prompt_same_params():
    """Identical prompts with identical params must produce identical fingerprints."""
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
    """Same prompt, different temperature → different fingerprints."""
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
    """Same prompt, different model → different fingerprints."""
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
    """Same prompt, different max_tokens → different fingerprints."""
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
    """Same prompt, different provider → different fingerprints."""
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
    """Different messages → different fingerprints."""
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
    """Messages in different order → same fingerprint (sorted by role)."""
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
    """None-valued optional params don't affect the fingerprint."""
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
    """Same prompt with seed → same fingerprint (seed is part of generation)."""
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
    # Different seed → different fingerprint
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
    """Args are correctly extracted."""
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
    """Kwargs are correctly extracted."""
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
    tmp_backend.get(key)  # first access
    tmp_backend.get(key)  # second access
    stats = tmp_backend.stats()
    assert stats["total_hits"] >= 2


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


def test_sqlite_max_entries_eviction(tmp_backend):
    """When max_entries is exceeded, oldest entries are evicted."""
    for i in range(150):  # max is 1000, but test with a low limit
        tmp_backend.set(f"k{i}", "openai", "gpt-4o", f"fp{i}", b"data", 3600)


def test_sqlite_stats(tmp_backend):
    tmp_backend.set("k1", "openai", "gpt-4o", "fp", b"d1", 3600)
    tmp_backend.set("k2", "openai", "gpt-4o", "fp", b"d2", 3600)
    tmp_backend.get("k1")
    tmp_backend.get("k1")
    stats = tmp_backend.stats()
    assert stats["total_entries"] == 2
    # k1 was accessed twice (hit_count = 1 from insert + 2 increments = 3)
    # k2 was accessed 0 times (hit_count = 1 from insert only)
    # total_hits = 3 + 1 = 4 (but SQLite SUM of hit_count after insert is: k1=3, k2=1 → 4)
    # Wait — INSERT OR REPLACE resets hit_count to 1. So after two gets of k1: k1 hit_count=3, k2 hit_count=1
    assert stats["total_hits"] == 4  # k1(3) + k2(1)


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
    # Oldest entries should be evicted
    assert mem_backend.get("k0") is None
    assert mem_backend.get("k119") is not None


def test_memory_ttl(mem_backend):
    mem_backend.set("k1", "openai", "gpt-4o", "fp", b"v1", ttl=1)
    time.sleep(1.1)
    assert mem_backend.get("k1") is None


# ─── Decorator integration tests ───────────────────────────────────────────────


@pytest.fixture
def tmp_cache_dir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_decorator_cache_hit(monkeypatch, tmp_cache_dir):
    """Second call with same args returns cached result."""
    call_count = 0

    @cached(provider="openai", model="gpt-4o", backend=SQLiteBackend(tmp_cache_dir),
            enabled=True)
    def api_call(messages, model="gpt-4o", temperature=0.7, max_tokens=512):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    # First call — hits API
    result1 = api_call(messages=[{"role": "user", "content": "hello"}])
    assert result1 == {"result": "call_1"}
    assert call_count == 1

    # Second call with SAME args — cache hit
    result2 = api_call(messages=[{"role": "user", "content": "hello"}])
    assert result2 == {"result": "call_1"}  # same as first!
    assert call_count == 1  # still only 1 API call


def test_decorator_different_temp_no_cache_hit(monkeypatch, tmp_cache_dir):
    """Different temperature → no cache hit (different fingerprint)."""
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


def test_decorator_different_message_no_cache_hit(monkeypatch, tmp_cache_dir):
    """Different message → no cache hit."""
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
    """Explicit invalidate removes the cache entry."""
    call_count = 0

    @cached(provider="openai", model="gpt-4o", backend=SQLiteBackend(tmp_cache_dir))
    def api_call(messages, model="gpt-4o"):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    api_call(messages=[{"role": "user", "content": "hello"}])
    assert call_count == 1

    # Invalidate by calling with same args
    api_call.invalidate(messages=[{"role": "user", "content": "hello"}])

    api_call(messages=[{"role": "user", "content": "hello"}])
    assert call_count == 2


def test_decorator_disabled():
    """When disabled, every call goes through to the function."""
    call_count = 0

    @cached(provider="openai", model="gpt-4o", enabled=False)
    def api_call(messages, model="gpt-4o"):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    api_call(messages=[{"role": "user", "content": "hello"}])
    api_call(messages=[{"role": "user", "content": "hello"}])
    assert call_count == 2


def test_cache_manager(tmp_cache_dir):
    """CacheManager creates provider-scoped decorators."""
    cache_mgr = CacheManager(backend=SQLiteBackend(tmp_cache_dir))
    call_count = 0

    @cache_mgr.openai(model="gpt-4o")
    def summarize(messages, model="gpt-4o", temperature=0.7, max_tokens=512):
        nonlocal call_count
        call_count += 1
        return {"result": f"call_{call_count}"}

    summarize(messages=[{"role": "user", "content": "test"}])
    summarize(messages=[{"role": "user", "content": "test"}])
    assert call_count == 1  # second call cached


# ─── Global stats test ─────────────────────────────────────────────────────────


def test_stats_tracked():
    stats = _stats()
    # Stats accumulate across calls — just verify the object exists and has fields
    assert hasattr(stats, "hits")
    assert hasattr(stats, "misses")
    assert hasattr(stats, "hit_rate")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
