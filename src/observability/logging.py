"""
observability/logging.py
────────────────────────
Structured JSON logging for BIM-Graph.

Every log line is a JSON object, which means you can query the log file
directly with jq:

    # trace a single request end-to-end
    jq 'select(.request_id == "a3b2c1d0")' logs/bim_graph.log

    # find all self-heal events
    jq 'select(.msg | contains("spatial_ast"))' logs/bim_graph.log

    # average node timings
    jq 'select(.event == "node_end") | .elapsed_ms' logs/bim_graph.log

Threading model
───────────────
The LangGraph pipeline runs in a background thread (see main.py). Python's
contextvars don't cross thread boundaries by default, so we use threading.local
to carry the request_id. Each node calls set_request_id() from the state at
entry, which is cheap and keeps the correlation correct even under concurrent
requests.
"""

import json
import logging
import pathlib
import threading
from logging.handlers import RotatingFileHandler

# ── Thread-local request ID ────────────────────────────────────────────────────
_local = threading.local()


def set_request_id(rid: str) -> None:
    """Set the request ID for the current thread. Call once per node entry."""
    _local.request_id = rid


def get_request_id() -> str:
    return getattr(_local, "request_id", "-")


# ── JSON formatter ─────────────────────────────────────────────────────────────
class _JsonFormatter(logging.Formatter):
    """
    Emits each log record as a single JSON line.

    Output shape:
      {"ts": "...", "level": "INFO", "logger": "...", "request_id": "...", "msg": "..."}

    Exceptions are serialized into an "exc" key so they don't break JSON parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts":         self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":      record.levelname,
            "logger":     record.name,
            "request_id": get_request_id(),
            "msg":        record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)


# ── Public setup function ──────────────────────────────────────────────────────
def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    """
    Configure the root logger to emit JSON to both stdout and a rotating file.

    Call this once at application startup (e.g. in FastAPI's lifespan event).
    The rotating file handler caps each file at 10 MB and keeps 5 backups,
    so the logs directory never grows unbounded.

    Args:
        log_dir: Directory where bim_graph.log will be written.
        level:   Minimum log level (default INFO).
    """
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file  = str(pathlib.Path(log_dir) / "bim_graph.log")
    formatter = _JsonFormatter()

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes   = 10 * 1024 * 1024,  # 10 MB per file
        backupCount= 5,
        encoding   = "utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("bim_graph").info(
        "Logging initialized — file=%s level=%s",
        log_file,
        logging.getLevelName(level),
    )
