"""A scripted stand-in for the Claude client, so the loop, tools, evals and UI can be
tested without an API key or cost. It follows the golden dataset: for answer cases it
searches, sends one broken query first (to exercise self-correction), then the gold SQL."""
import itertools
from types import SimpleNamespace

import yaml

from config import GOLDEN_PATH

_ids = itertools.count(1)


def _tool(name, **inp):
    return SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids):04d}", name=name, input=inp)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _resp(blocks, stop="tool_use"):
    return SimpleNamespace(content=blocks, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=1800, output_tokens=220,
                                                 cache_creation_input_tokens=0,
                                                 cache_read_input_tokens=900))


class _Messages:
    def __init__(self, golden):
        self.golden = {c["question"]: c for c in golden}
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        msgs = kw["messages"]
        if "tools" not in kw:  # LLM-as-judge call
            return _resp([_text('{"faithful": true, "reason": "numbers match the result"}')], "end_turn")
        question = next(m["content"] for m in reversed(msgs)
                        if m["role"] == "user" and isinstance(m["content"], str))
        step = sum(1 for m in msgs if m["role"] == "assistant" and not isinstance(m["content"], str))
        case = self.golden.get(question, {"category": "answer",
                                          "gold_sql": "SELECT COUNT(*) AS enrolments FROM enrolments"})
        if case["category"] == "clarification":
            return _resp([_tool("final_answer", type="clarification",
                                summary="Which metric do you mean: enrolments, conversion or revenue?")])
        if case["category"] == "refusal":
            return _resp([_tool("final_answer", type="refusal",
                                summary="I can't share personal data or help with off-topic requests.")])
        if step == 0:
            return _resp([_text("Let me find the right definitions."), _tool("search_context", query=question)])
        if step == 1:
            return _resp([_tool("run_sql", sql="SELECT not_a_column FROM enrolments")])
        if step == 2:
            return _resp([_tool("run_sql", sql=case["gold_sql"])])
        return _resp([_tool("final_answer", type="answer", sql=case["gold_sql"],
                            summary="Here is the result based on the query.", chart="bar",
                            assumptions=["Used the certified definition."])])


class FakeClaude:
    def __init__(self):
        with open(GOLDEN_PATH, encoding="utf-8") as f:
            self.messages = _Messages(yaml.safe_load(f))
