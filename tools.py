"""TOOLS the model can call, and the read-only SQL executor.

The model never runs anything itself. It returns a tool call (name + JSON input);
the orchestrator in agent.py calls the matching Python function below and sends the
result back. Each tool's description is effectively part of the prompt.
"""
import duckdb
import pandas as pd

from config import DB_PATH, MAX_ROWS_RETURNED, ROWS_SHOWN_TO_MODEL
from data_setup import build_database
from guardrails import check_sql

TOOLS = [
    {
        "name": "search_context",
        "description": (
            "Search the company's context layer: certified metric definitions, table and column "
            "descriptions, glossary terms and verified example queries. Call this before writing any "
            "SQL. Write the search query in business terms (for example 'trial to paid conversion by "
            "centre'). Call it again with different words if the first results do not cover the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up, in plain business words."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_sql",
        "description": (
            "Run one read-only DuckDB SELECT query on the analytics database and get back the column "
            "names, the row count and the first rows. Use only tables and columns found through "
            "search_context. If the result is an error, read it, fix the query and call run_sql again. "
            "Queries that touch personal data or anything other than the approved tables are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single DuckDB SELECT statement."}
            },
            "required": ["sql"],
        },
    },
    {
        "name": "final_answer",
        "description": (
            "Finish the task. Always end by calling this exactly once. Use type 'answer' with the final "
            "SQL and a 2-4 sentence summary based only on the rows you saw. Use type 'clarification' "
            "with one short question when the request is too vague to answer. Use type 'refusal' "
            "with a one-line reason for personal data, changing data, or off-topic requests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["answer", "clarification", "refusal"]},
                "summary": {"type": "string",
                            "description": "The answer, the clarifying question, or the refusal reason."},
                "sql": {"type": "string", "description": "The final SQL query (required when type is 'answer')."},
                "assumptions": {"type": "array", "items": {"type": "string"},
                                "description": "Interpretations you made, e.g. which date field or definition."},
                "chart": {"type": "string", "enum": ["bar", "line", "none"],
                          "description": "Best chart for the result."},
            },
            "required": ["type", "summary"],
        },
    },
]

_connection = None


def get_connection():
    """Read-only connection with file and network access switched off (defence in depth)."""
    global _connection
    if _connection is None:
        build_database()
        _connection = duckdb.connect(DB_PATH, read_only=True,
                                     config={"enable_external_access": False})
    return _connection


def execute_sql(sql: str):
    """Guardrail check, then run. Returns (DataFrame or None, error message or None)."""
    ok, reason = check_sql(sql)
    if not ok:
        return None, f"BLOCKED BY GUARDRAIL: {reason}"
    try:
        cur = get_connection().cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS_RETURNED)
        return pd.DataFrame(rows, columns=cols), None
    except Exception as e:  # database errors go back to the model so it can self-correct
        return None, f"SQL ERROR: {str(e).splitlines()[0][:500]}"


def format_sql_result(df: pd.DataFrame) -> str:
    """What the model sees after run_sql: small and structured, never the whole table."""
    shown = df.head(ROWS_SHOWN_TO_MODEL)
    note = "" if len(df) <= ROWS_SHOWN_TO_MODEL else f" (showing first {ROWS_SHOWN_TO_MODEL})"
    return (f"Columns: {list(df.columns)}\nRow count: {len(df)}{note}\n"
            f"Rows (CSV):\n{shown.to_csv(index=False)}")
