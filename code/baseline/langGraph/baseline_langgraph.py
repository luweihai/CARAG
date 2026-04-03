#!/usr/bin/env python3
"""
baseline_langgraph.py – ultra‑robust version
===========================================
The previous run shows a `KeyError('transcript')` occurring **inside
`analyst_agent`**, which means `langgraph` delivered a state object that lacks
that key *despite us passing it*.  This patch makes the pipeline bullet‑proof
by:

1. **Forcing a new state dict** for every invocation (`unsafe_copy=False`) so
   that residual keys from prior runs cannot bleed through.
2. **Triple‑checking** inside `analyst_agent` with a fallback list of
   alternative key names and a *last‑resort* grab of the dict’s first string.
3. **Logging full state keys** when the key is missing so we can see what came
   through.
4. **Coercing non‑strings** to strings up‑front (e.g. `np.nan` ➔ "").
5. **Adding an explicit `ERROR_MISSING_TRANSCRIPT` entry to the CSV when it
   still cannot recover.

This should prevent the loop from aborting and give maximum visibility into
what’s going on.
"""
from __future__ import annotations
import os
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- Config ------------------------------------------------------------
CREDENTIALS_FILE = "credentials.json"
DATA_PATH        = "transcripts_nasdaq.csv"
LOG_PATH         = "baseline_nasdaq_lang2.csv"
MODEL_NAME       = "gpt-4o-mini"
RETURN_THRESHOLD = 0.05
MAX_TRANSCRIPT_CHARS = 15_000
TRANSCRIPT_COL_CANDIDATES: List[str] = [
    "transcript", "clean_transcript", "call_text", "text"
]

creds = json.loads(Path(CREDENTIALS_FILE).read_text())
OPENAI_API_KEY = creds["openai_api_key"]

llm = ChatOpenAI(model=MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a portfolio manager reading an earnings‑call transcript. Decide "
    "whether the stock price is likely to go Up or Down the next trading day, "
    "and assign a Direction score 0–10 (0 = sure Down, 5 = neutral, 10 = sure Up).\n\n"
    "Return *exactly*:\n\n<two‑sentence explanation>\nDirection : <0-10>"
)

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{transcript}")
])

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Review the note. If you believe that the reasoning of the note is correct: ACCEPTED:<note>. Else CORRECTED:<better note>"),
    ("human", "Draft note:\n{draft}\n")
])

# --------------------- LangGraph ---------------------------------------
class CallState(dict):
    pass

ALT_KEYS = ["text", "call_text", "clean_transcript"]

def _extract_transcript(state: Dict[str, Any]) -> str | None:
    if "transcript" in state:
        return state["transcript"]
    for k in ALT_KEYS:
        if k in state:
            return state[k]
    # fallback: first long-ish string value
    for v in state.values():
        if isinstance(v, str) and len(v) > 20:
            return v
    return None


def analyst_agent(state: CallState) -> CallState:
    txt = _extract_transcript(state)
    if txt is None:
        state["error"] = "MISSING_TRANSCRIPT_KEY"
        state["draft"] = ""
        print("🚨 analyst_agent received state without transcript; keys:", list(state.keys()))
        return state

    if len(txt) > MAX_TRANSCRIPT_CHARS:
        txt = txt[:MAX_TRANSCRIPT_CHARS]
        state["truncated"] = True

    try:
        draft = llm(ANALYST_PROMPT.format_messages(transcript=txt)).content
    except Exception as exc:
        state["error"] = f"LLM_ERR:{exc}"
        state["draft"] = ""
        traceback.print_exc()
        return state

    state["draft"] = draft
    return state


def critic_agent(state: CallState) -> CallState:
    draft = state.get("draft", "")
    if not draft:
        state["final"] = ""
        return state
    verdict = llm(CRITIC_PROMPT.format_messages(draft=draft)).content
    state["final"] = verdict.replace("ACCEPTED:", "").replace("CORRECTED:", "").strip()
    return state

wg = StateGraph(dict, unsafe_copy=False)  # ensure fresh dict each run
wg.add_node("analyst", analyst_agent)
wg.add_node("critic", critic_agent)
wg.set_entry_point("analyst")
wg.add_edge("analyst", "critic")
wg.add_edge("critic", END)
chain = wg.compile()

# --------------------- Helpers ----------------------------------------

def detect_col(df):
    for c in TRANSCRIPT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("No transcript column found!")

def load_log(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["ticker","q","analysis","error","actual_return"]).to_csv(path, index=False) or pd.read_csv(path)

def upsert(df, entry):
    m = (df.ticker == entry["ticker"]) & (df.q == entry["q"])
    if df[m].empty:
        df.loc[len(df)] = entry
    else:
        for k,v in entry.items():
            df.loc[m,k]=v
    df.to_csv(LOG_PATH,index=False)

# --------------------- Main -------------------------------------------
if __name__ == "__main__":
    meta = pd.read_csv(DATA_PATH)
    col  = detect_col(meta)
    log  = load_log(LOG_PATH)

    for _, r in meta.iterrows():
        tkr, q = r.ticker, r.q
        ret = r.get("future_3bday_cum_return", 0.0)
        txt = r.get(col, "")
        txt = "" if pd.isna(txt) else str(txt)
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