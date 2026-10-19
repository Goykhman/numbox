# Installs numbox's part of numba's on-disk cache key before any numbox module compiles; see cache_key.
from numbox.core import cache_key as _cache_key  # noqa: F401
