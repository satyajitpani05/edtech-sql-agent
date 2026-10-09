"""Central settings for the agent. Change values here, not inside the other files."""
import os
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "edtech.duckdb")
CONTEXT_PATH = os.path.join(BASE_DIR, "context.yaml")
GOLDEN_PATH = os.path.join(BASE_DIR, "eval", "golden.yaml")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "eval", "eval_results.csv")
EVAL_SUMMARY_PATH = os.path.join(BASE_DIR, "eval", "eval_summary.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
TRACES_PATH = os.path.join(LOG_DIR, "traces.jsonl")
FEEDBACK_PATH = os.path.join(LOG_DIR, "feedback.jsonl")
VERIFIED_ADDITIONS_PATH = os.path.join(LOG_DIR, "verified_additions.jsonl")

# The model that writes SQL. Sonnet is the balance of quality, speed and cost.
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

# USD per million tokens (input, output), from the Claude pricing page.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Agent limits (guardrails on the loop itself)
MAX_STEPS = 8            # most model calls allowed for one question
MAX_SQL_RETRIES = 3      # failed run_sql calls allowed before giving up
MAX_ROWS_RETURNED = 200  # rows kept from any query
ROWS_SHOWN_TO_MODEL = 30 # rows the model sees (keeps the prompt small)
HISTORY_TURNS = 3        # earlier questions kept for follow-ups
RETRIEVAL_TOP_K = 6      # context documents returned by search_context

# Data window of the synthetic dataset
DATA_START = date(2025, 4, 1)
DATA_END = date(2026, 9, 30)

COMPANY = "BrightPath Learning"
