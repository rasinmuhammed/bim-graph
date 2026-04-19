"""
agent/token_stream.py
─────────────────────
Thread-local queue for forwarding LLM tokens to the SSE transport layer.

The generate node runs inside a background thread (launched by main.py's
SSE endpoint). To stream individual tokens to the client without coupling
the node to FastAPI, we use a thread-local queue:

  - main.py creates a queue, stores it via set_token_queue(), then starts
    the LangGraph thread.
  - The generate node calls get_token_queue() to find it. If the pipeline
    is invoked outside of HTTP (e.g. benchmark, tests), get_token_queue()
    returns None and the node skips emission — no side effects.
  - main.py drains the queue alongside the normal node-event queue.

This keeps the node pure (no FastAPI import, no transport knowledge) while
giving the UI real token-by-token streaming.
"""

import queue
import threading

_local = threading.local()


def set_token_queue(q: "queue.Queue[str] | None") -> None:
    """Register a token queue for the current thread."""
    _local.queue = q


def get_token_queue() -> "queue.Queue[str] | None":
    """Return the token queue for the current thread, or None."""
    return getattr(_local, "queue", None)
