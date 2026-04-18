from cache.redis_cache import cache_get, cache_set


def test_cache_roundtrip():
    """Data stored must be retrievable with the same query."""
    query  = "test query for cache roundtrip"
    answer = "The answer is 42"
    log    = [{"attempt": 1, "search_strategy": "ast",
               "failure_reason": "mismatch", "action_taken": "ast traversal"}]

    cache_set(query, answer, log)
    result = cache_get(query)

    assert result is not None
    assert result["answer"] == answer
    assert result["correction_log"] == log


def test_cache_miss_returns_none():
    result = cache_get("this query was never stored xyz987")
    assert result is None