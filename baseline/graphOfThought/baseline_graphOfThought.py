#!/usr/bin/env python3
"""
baseline_graph_of_thoughts.py  –  minimal Graph‑of‑Thoughts (GoT) baseline
=======================================================================
This script upgrades the earlier *ultra‑robust* pipeline by swapping the
single‑draft + critic loop for a **Graph‑of‑Thoughts** reasoning strategy
(Pandey et al., 2025).  At test‑time the model:

1. **Expands** K diverse candidate "thoughts" (draft notes) for each
   transcript.
2. **Scores** every thought with a value function agent that predicts how
   accurate / persuasive it is.
3. **Selects** the top‑scoring thought as the final note.

The rest of the robustness tweaks (fresh state, triple key search, truncated
transcripts, detailed error logging, etc.) are retained.
"""
from __future__ import annotations

import os
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# ────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────
CREDENTIALS_FILE        = "credentials.json"
DATA_PATH               = "transcripts_nyse.csv"
LOG_PATH                = "baseline_nyse_got.csv"
MODEL_NAME              = "gpt-4o-mini"
RETURN_THRESHOLD        = 0.05
MAX_TRANSCRIPT_CHARS    = 15_000
NUM_THOUGHT_SAMPLES     = 3      # K in the GoT paper (small for a baseline)
THOUGHT_TEMP            = 0.7    # encourage diversity across samples
VALUE_TEMP              = 0.0    # deterministic value estimation
TRANSCRIPT_COL_CANDIDATES: List[str] = [
    "transcript", "clean_transcript", "call_text", "text"
]

creds           = json.loads(Path(CREDENTIALS_FILE).read_text())
OPENAI_API_KEY  = creds["openai_api_key"]

llm_diverse = ChatOpenAI(
    model         = MODEL_NAME,
    temperature   = THOUGHT_TEMP,
    openai_api_key= OPENAI_API_KEY,
)
llm_value = ChatOpenAI(
    model         = MODEL_NAME,
    temperature   = VALUE_TEMP,
    openai_api_key= OPENAI_API_KEY,
)

SYSTEM_PROMPT = (
    "You are a portfolio manager reading an earnings‑call transcript. Decide "
    "whether the stock price is likely to go Up or Down the next trading day, "
    "and assign a Direction score 0–10 (0 = sure Down, 5 = neutral, 10 = sure Up).\n\n"
    "Return *exactly*: \n"
    "<two‑sentence explanation>\nDirection : <0-10>"
)

THOUGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nThink step‑by‑step, reasoning carefully before answering."),
    ("human", "{transcript}")
])

VALUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a rigorous evaluator of analyst notes.  Rate each note 0–10 for how correct and convincing it is given the transcript."),
    ("human", "Transcript (truncated):\n" + "{transcript}\n\n" +
               "Candidate note:\n{note}\n\nReply with a single integer 0–10 only.")
])

# ────────────────────────────────────────────────────────────────────────
# LangGraph state helpers
# ────────────────────────────────────────────────────────────────────────
class CallState(dict):
    """Simple dict‑based state, as required by LangGraph."""

ALT_KEYS = ["text", "call_text", "clean_transcript"]

def _extract_transcript(state: Dict[str, Any]) -> str | None:
    if "transcript" in state:
        return state["transcript"]
    for k in ALT_KEYS:
        if k in state:
            return state[k]
    for v in state.values():
        if isinstance(v, str) and len(v) > 20:
            return v
    return None

# ────────────────────────────────────────────────────────────────────────
# 1. Thought expansion
# ────────────────────────────────────────────────────────────────────────

def expand_thoughts(state: CallState) -> CallState:
    """Generate K diverse candidate notes."""
    txt = _extract_transcript(state)
    if txt is None:
        state["error"] = "MISSING_TRANSCRIPT_KEY"
        print("🚨 expand_thoughts: no transcript; keys:", list(state.keys()))
        return state

    if len(txt) > MAX_TRANSCRIPT_CHARS:
        txt = txt[:MAX_TRANSCRIPT_CHARS]
        state["truncated"] = True

    drafts = []
    errors = []
    for _ in range(NUM_THOUGHT_SAMPLES):
        try:
            draft = llm_diverse(THOUGHT_PROMPT.format_messages(transcript=txt)).content
            drafts.append(draft.strip())
        except Exception as exc:
            errors.append(f"LLM_ERR:{exc}")
            traceback.print_exc()
            drafts.append("")
    state.update({"drafts": drafts, "expand_errors": errors})
    return state

# ────────────────────────────────────────────────────────────────────────
# 2. Value scoring
# ────────────────────────────────────────────────────────────────────────

def score_thoughts(state: CallState) -> CallState:
    drafts = state.get("drafts", [])
    txt    = _extract_transcript(state) or ""
    scores: List[float] = []
    for note in drafts:
        if not note:
            scores.append(-np.inf)
            continue
        try:
            raw = llm_value(VALUE_PROMPT.format_messages(transcript=txt[:2000], note=note)).content.strip()
            score = float(raw) if raw.replace('.', '', 1).isdigit() else -np.inf
        except Exception as exc:
            traceback.print_exc()
            score = -np.inf
        scores.append(score)
    state["scores"] = scores
    return state

# ────────────────────────────────────────────────────────────────────────
# 3. Selection
# ────────────────────────────────────────────────────────────────────────

def select_best(state: CallState) -> CallState:
    drafts = state.get("drafts", [])
    scores = state.get("scores", [])
    if not drafts or not scores:
        state["final"] = ""
        state["error"] = state.get("error", "NO_DRAFTS_OR_SCORES")
        return state
    idx = int(np.argmax(scores))
    state["final"] = drafts[idx]
    state["selected_score"] = scores[idx]
    return state

# ────────────────────────────────────────────────────────────────────────
# Build the Graph‑of‑Thoughts pipeline
# ────────────────────────────────────────────────────────────────────────
wg = StateGraph(dict, unsafe_copy=False)
wg.add_node("expand",  expand_thoughts)
wg.add_node("score",   score_thoughts)
wg.add_node("select",  select_best)
wg.set_entry_point("expand")
wg.add_edge("expand", "score")
wg.add_edge("score",  "select")
wg.add_edge("select", END)
chain = wg.compile()

# ────────────────────────────────────────────────────────────────────────
# Utility helpers – unchanged
# ────────────────────────────────────────────────────────────────────────

def detect_col(df):
    for c in TRANSCRIPT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("No transcript column found!")

def load_log(path):
    return (
        pd.read_csv(path)
        if os.path.exists(path)
        else pd.DataFrame(columns=["ticker","q","analysis","error","actual_return"]).to_csv(path, index=False) or pd.read_csv(path)
    )

def upsert(df, entry):
    m = (df.ticker == entry["ticker"]) & (df.q == entry["q"])
    if df[m].empty:
        df.loc[len(df)] = entry
    else:
        for k,v in entry.items():
            df.loc[m,k] = v
    df.to_csv(LOG_PATH, index=False)

# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    meta = pd.read_csv(DATA_PATH)
    meta = meta[meta["future_3bday_cum_return"].abs() > 0.05]         # keep |r| > 5

    col  = detect_col(meta)
    log  = load_log(LOG_PATH)

    for _, r in meta.iterrows():
        tkr, q = r.ticker, r.q
        ret    = r.get("future_3bday_cum_return", 0.0)
        txt    = r.get(col, "")
        txt    = "" if pd.isna(txt) else str(txt)
        if not txt.strip():
            upsert(log, {"ticker": tkr, "q": q, "analysis":"", "error":"NO_TEXT", "actual_return": ret})
            continue
        try:
            st = chain.invoke({"transcript": txt})
            upsert(log, {"ticker": tkr, "q": q, "analysis": st.get("final",""), "error": st.get("error",""), "actual_return": ret})
        except Exception as exc:
            traceback.print_exc()
            upsert(log, {"ticker": tkr, "q": q, "analysis":"", "error": str(exc), "actual_return": ret})

    print("✅ All rows processed")
