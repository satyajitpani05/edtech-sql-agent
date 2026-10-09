"""OBSERVABILITY and FEEDBACK storage.

Every agent run is saved as a trace (one JSON line): the question, each step, tokens,
cost and latency. User feedback is saved the same way. On Streamlit Cloud these files
reset when the app restarts; in production they would go to a database.
"""
import json
import os
from datetime import datetime, timezone

from config import TRACES_PATH, FEEDBACK_PATH, VERIFIED_ADDITIONS_PATH, PRICES, MODEL


def _append(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def estimate_cost(usage: dict, model: str = MODEL) -> float:
    """USD cost from token counts. Cache writes cost 1.25x input; cache reads 0.05x (Sonnet/Opus 5.5)."""
    in_price, out_price = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    read_mult = 0.1 if "haiku" in model else 0.05
    cost = (usage.get("input_tokens", 0) * in_price
            + usage.get("cache_creation_input_tokens", 0) * in_price * 1.25
            + usage.get("cache_read_input_tokens", 0) * in_price * read_mult
            + usage.get("output_tokens", 0) * out_price)
    return round(cost / 1_000_000, 5)


def log_trace(record: dict) -> None:
    _append(TRACES_PATH, record)


def log_feedback(run_id: str, question: str, rating: str, comment: str = "",
                 corrected_sql: str = "") -> None:
    _append(FEEDBACK_PATH, {"time": now_iso(), "run_id": run_id, "question": question,
                            "rating": rating, "comment": comment, "corrected_sql": corrected_sql})


def add_verified_query(question: str, sql: str) -> None:
    """Feedback loop: an analyst-approved query becomes retrievable context for next time."""
    _append(VERIFIED_ADDITIONS_PATH, {"question": question, "sql": sql, "added": now_iso()})
