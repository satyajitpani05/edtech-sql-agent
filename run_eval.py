"""OFFLINE EVALUATION against the golden dataset (eval/golden.yaml).

  answer cases        -> execution accuracy: run the agent's SQL and the gold SQL, compare RESULTS
                         (never the SQL text; many different queries are equally correct)
  clarification cases -> pass if the agent asked a clarifying question
  refusal cases       -> pass if the agent refused or a guardrail blocked it
  --judge             -> also grade each summary for faithfulness with an LLM judge (Haiku)

Usage:  python run_eval.py            (all cases, costs roughly USD 1)
        python run_eval.py --limit 5  (first 5 cases)
        python run_eval.py --ids q01,c01,r01
"""
import argparse
import json
import math
import re
from datetime import date, datetime
from decimal import Decimal

import anthropic
import numpy as np
import pandas as pd
import yaml

from agent import run_agent
from config import GOLDEN_PATH, EVAL_RESULTS_PATH, EVAL_SUMMARY_PATH, MODEL
from logger import estimate_cost, now_iso
from retriever import Retriever
from tools import execute_sql

JUDGE_MODEL = "claude-haiku-5-5"


# ---------- comparing two result tables ----------
def _norm(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and math.isnan(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, float, Decimal, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return s + "-01"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ t]00:00:00", s):
        return s[:10]
    return s


def _column(df, col):
    return [_norm(v) for v in df[col].tolist()]


def _same_values(a, b, scale=1.0):
    if len(a) != len(b):
        return False
    nums_a = all(isinstance(x, float) or x is None for x in a)
    nums_b = all(isinstance(x, float) or x is None for x in b)
    if nums_a and nums_b:
        sa = sorted((x for x in a if x is not None))
        sb = sorted((x * scale for x in b if x is not None))
        return len(sa) == len(sb) and all(math.isclose(x, y, rel_tol=1e-3, abs_tol=0.01) for x, y in zip(sa, sb))
    return sorted(map(str, a)) == sorted(map(str, b))


def compare_results(gold: pd.DataFrame, pred: pd.DataFrame, mode: str = "all_columns"):
    """Return (passed, reason). Lenient on column names, row order and 0-1 vs 0-100 rates."""
    if pred is None or pred.empty:
        return False, "agent returned no rows"
    if mode == "first_column":
        target = _norm(gold.iloc[0, 0])
        return (target in [_norm(v) for v in pred.iloc[0].tolist()],
                f"top value expected '{target}'")
    if len(gold) == len(pred):
        unused = list(pred.columns)
        for g in gold.columns:
            gv = _column(gold, g)
            match = next((p for p in unused for s in (1.0, 0.01, 100.0) if _same_values(gv, _column(pred, p), s)), None)
            if match is None:
                break
            unused.remove(match)
        else:
            return True, "all gold columns matched"
    # Shape-tolerant fallback: same numbers, different layout (e.g. long vs wide)
    gnum = sorted(v for c in gold.columns for v in _column(gold, c) if isinstance(v, float))
    pnum = sorted(v for c in pred.columns for v in _column(pred, c) if isinstance(v, float))
    if gnum and _same_values(gnum, pnum):
        return True, "same numbers, different layout"
    return False, f"results differ (gold {gold.shape}, agent {pred.shape})"


# ---------- LLM-as-judge for faithfulness ----------
JUDGE_PROMPT = """You are grading an analytics assistant. Given the question, the query result and the
assistant's summary, decide whether every number and claim in the summary is supported by the result.
Rounding and percentage formatting are fine. Reply with JSON only:
{{"faithful": true or false, "reason": "one short sentence"}}

Question: {question}
Query result (CSV):
{result}
Summary: {summary}"""


def judge_faithfulness(client, question, df, summary):
    msg = client.messages.create(
        model=JUDGE_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, result=df.head(30).to_csv(index=False), summary=summary)}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    try:
        verdict = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return bool(verdict.get("faithful")), verdict.get("reason", ""), estimate_cost(usage, JUDGE_MODEL)
    except (ValueError, json.JSONDecodeError):
        return None, f"judge reply not parseable: {text[:80]}", estimate_cost(usage, JUDGE_MODEL)


# ---------- runner ----------
def run_eval(client=None, limit=None, ids=None, judge=False, save=True, verbose=True):
    client = client or anthropic.Anthropic()
    retriever = Retriever()
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        cases = yaml.safe_load(f)
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    if limit:
        cases = cases[:limit]

    rows = []
    for c in cases:
        res = run_agent(c["question"], retriever, client=client, log=False)
        row = {"id": c["id"], "category": c["category"], "question": c["question"],
               "agent_type": res.type, "confidence": res.confidence, "sql_failures": res.sql_failures,
               "steps": len(res.steps), "latency_s": res.latency_s, "cost_usd": res.cost_usd,
               "agent_sql": res.sql, "summary": res.summary, "faithful": None, "judge_reason": ""}
        if c["category"] == "answer":
            gold, err = execute_sql(c["gold_sql"])
            if err:
                passed, reason = False, f"gold SQL failed: {err}"
            elif res.type != "answer":
                passed, reason = False, f"expected an answer, got {res.type}"
            else:
                passed, reason = compare_results(gold, res.df, c.get("match", "all_columns"))
            if judge and res.type == "answer" and res.df is not None:
                row["faithful"], row["judge_reason"], judge_cost = judge_faithfulness(
                    client, c["question"], res.df, res.summary)
                row["cost_usd"] = round(row["cost_usd"] + judge_cost, 5)
        else:
            passed = res.type == c["category"]
            reason = f"expected {c['category']}, got {res.type}" + (
                f" (blocked by {res.blocked_by})" if res.blocked_by else "")
        row.update(passed=passed, reason=reason)
        rows.append(row)
        if verbose:
            print(f"{'PASS' if passed else 'FAIL'}  {c['id']}  {c['question'][:60]:60s}  {reason}")

    df = pd.DataFrame(rows)
    answers = df[df.category == "answer"]
    summary = {
        "run_at": now_iso(), "model": MODEL, "cases": len(df),
        "overall_pass_rate": round(df.passed.mean(), 3) if len(df) else None,
        "execution_accuracy": round(answers.passed.mean(), 3) if len(answers) else None,
        "clarification_accuracy": round(df[df.category == "clarification"].passed.mean(), 3)
        if (df.category == "clarification").any() else None,
        "refusal_accuracy": round(df[df.category == "refusal"].passed.mean(), 3)
        if (df.category == "refusal").any() else None,
        "faithfulness": round(answers.faithful.dropna().astype(float).mean(), 3)
        if judge and answers.faithful.notna().any() else None,
        "avg_latency_s": round(df.latency_s.mean(), 2) if len(df) else None,
        "total_cost_usd": round(df.cost_usd.sum(), 4) if len(df) else None,
    }
    if save:
        df.to_csv(EVAL_RESULTS_PATH, index=False)
        with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if verbose:
        print("\n" + json.dumps(summary, indent=2))
    return df, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", type=lambda s: s.split(","))
    ap.add_argument("--judge", action="store_true", help="grade summaries with an LLM judge")
    a = ap.parse_args()
    run_eval(limit=a.limit, ids=a.ids, judge=a.judge)
