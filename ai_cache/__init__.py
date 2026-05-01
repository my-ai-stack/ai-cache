"""
ai_cache — zero-overhead AI response caching with payload-aware keying.
One decorator. Any provider. No proxy.
"""

from .cache import cached, CacheConfig

__version__ = "0.1.0"
__all__ = ["cached", "CacheConfig"]
