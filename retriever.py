"""RETRIEVAL (the R in RAG).

Turns the context layer (context.yaml) into searchable documents and finds the ones most
relevant to a question. Uses TF-IDF + cosine similarity: at ~40 documents this is fast,
free and deterministic. In production you would swap in embeddings + hybrid search +
reranking behind the same search() method.
"""
import json
import os
import re

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CONTEXT_PATH, VERIFIED_ADDITIONS_PATH, RETRIEVAL_TOP_K


def _normalise(text: str) -> str:
    """Split snake_case so 'demo_sessions' also matches the word 'demo'."""
    return text.replace("_", " ").replace("-", " ").lower()


def load_context(path: str = CONTEXT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tables_in_sql(sql: str, table_names: list) -> list:
    return [t for t in table_names if re.search(rf"\b{t}\b", sql, flags=re.IGNORECASE)]


def build_documents(ctx: dict, additions_path: str = VERIFIED_ADDITIONS_PATH) -> list:
    """One document per table, metric, glossary term and verified query."""
    docs = []
    table_names = [t["name"] for t in ctx["tables"]]

    for t in ctx["tables"]:
        cols = "\n".join(f"  - {c}: {d}" for c, d in t["columns"].items())
        content = (f"TABLE {t['name']} - {t['description']} Grain: {t['grain']}.\n"
                   f"Columns:\n{cols}\nJoins to: {t['joins']}")
        docs.append({"id": f"table:{t['name']}", "kind": "table", "title": t["name"],
                     "tables": [t["name"]], "content": content,
                     "search_text": content})

    for m in ctx["metrics"]:
        synonyms = ", ".join(m.get("synonyms", []))
        content = (f"METRIC {m['name']} ({m['label']}). {m['description'].strip()}\n"
                   f"Synonyms: {synonyms}\nCertified SQL logic: {m['sql'].strip()}\n"
                   f"Tables: {', '.join(m['tables'])}")
        docs.append({"id": f"metric:{m['name']}", "kind": "metric", "title": m["label"],
                     "tables": m["tables"], "content": content,
                     "search_text": f"{m['name']} {m['label']} {synonyms} {m['description']}"})

    for g in ctx["glossary"]:
        content = f"GLOSSARY '{g['term']}': {g['meaning']}"
        docs.append({"id": f"glossary:{g['term']}", "kind": "glossary", "title": g["term"],
                     "tables": [], "content": content, "search_text": content})

    verified = list(ctx["verified_queries"])
    if os.path.exists(additions_path):  # feedback loop: analyst-corrected queries
        with open(additions_path, encoding="utf-8") as f:
            verified += [json.loads(line) for line in f if line.strip()]
    for i, v in enumerate(verified):
        content = f"VERIFIED QUERY\nQuestion: {v['question']}\nSQL:\n{v['sql'].strip()}"
        docs.append({"id": f"verified:{i}", "kind": "verified", "title": v["question"],
                     "tables": _tables_in_sql(v["sql"], table_names), "content": content,
                     "search_text": v["question"]})
    return docs


class Retriever:
    def __init__(self, context_path: str = CONTEXT_PATH, additions_path: str = VERIFIED_ADDITIONS_PATH):
        self.ctx = load_context(context_path)
        self.docs = build_documents(self.ctx, additions_path)
        self.by_id = {d["id"]: d for d in self.docs}
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform([_normalise(d["search_text"]) for d in self.docs])

    def search(self, query: str, k: int = RETRIEVAL_TOP_K, expand_tables: bool = True) -> list:
        """Return the top-k documents with similarity scores.

        expand_tables: also attach the table descriptions that the retrieved metrics and
        verified queries depend on (a simple form of schema linking).
        """
        scores = cosine_similarity(self.vectorizer.transform([_normalise(query)]), self.matrix)[0]
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        results = [dict(self.docs[i], score=round(float(scores[i]), 3))
                   for i in ranked[:k] if scores[i] > 0]
        if expand_tables:
            have = {r["id"] for r in results}
            needed = [t for r in results if r["kind"] in ("metric", "verified") for t in r["tables"]]
            for t in dict.fromkeys(needed):
                tid = f"table:{t}"
                if tid not in have:
                    results.append(dict(self.by_id[tid], score=0.0, linked=True))
                    have.add(tid)
        return results


def format_results(results: list) -> str:
    """The text the model actually sees as the search_context tool result."""
    if not results:
        return "No matching context found. Ask the user to clarify, or refuse if the question is out of scope."
    parts = []
    for r in results:
        tag = "linked table" if r.get("linked") else f"relevance {r['score']}"
        parts.append(f"[{r['id']} | {tag}]\n{r['content']}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    r = Retriever()
    for q in ["trial conversion by centre last month", "dropout by course", "how much money did we make"]:
        print(f"\n=== {q}")
        for d in r.search(q):
            print(f"  {d['score']:.3f}  {d['id']}{'  (linked)' if d.get('linked') else ''}")
