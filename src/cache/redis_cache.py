"""
Semantic cache layer for the BIM-Graph pipeline.
Key = SHA-256( query.lower() + "|" + floor.lower() )
TTL = 1 hour (configurable)
Falls back to fakeredis if the real Redis server is unavailable.
"""
import json
import hashlib
import logging
import redis
import fakeredis
from config import settings

logger = logging.getLogger("bim_graph.cache")

_TTL_SECONDS  = settings.cache_ttl_seconds
_REDIS_HOST   = settings.redis_host
_REDIS_PORT   = settings.redis_port


def _get_client():
    """Return a real Redis client, or fakeredis if Redis is unavailable."""
    try:
        client = redis.Redis(host=_REDIS_HOST, port=_REDIS_PORT, decode_responses=True)
        client.ping()
        logger.info("  [Cache] Connected to Redis at %s:%d", _REDIS_HOST, _REDIS_PORT)
        return client
    except (redis.ConnectionError, redis.TimeoutError):
        logger.warning("  [Cache] Redis unavailable — using fakeredis (in-memory).")
        return fakeredis.FakeRedis(decode_responses=True)


_client = _get_client()   # singleton


def _make_key(query: str) -> str:
    raw = query.lower().strip()
    return "bim-graph:" + hashlib.sha256(raw.encode()).hexdigest()


def cache_get(query: str) -> dict | None:
    """
    Look up a cached result.
    Returns {"answer": str, "correction_log": list} or None on miss.
    """
    key  = _make_key(query)
    data = _client.get(key)
    if data:
        logger.info("  [Cache] ⚡ CACHE HIT  key=%s…", key[-8:])
        return json.loads(data)
    logger.info("  [Cache] MISS  key=%s…", key[-8:])
    return None


def cache_set(query: str, answer: str, correction_log: list) -> None:
    """Store a successful result with TTL."""
    key     = _make_key(query)
    payload = json.dumps({"answer": answer, "correction_log": correction_log})
    _client.setex(key, _TTL_SECONDS, payload)
    logger.info("  [Cache] Stored  key=%s…  TTL=%ds", key[-8:], _TTL_SECONDS)
