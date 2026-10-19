"""Live view of the crew. Run: streamlit run ui/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.config import get_settings
from app.deps import build_deps
from app.llm import BudgetExceeded, LLMError
from app.main import parse_sites
from app.runner import stream_crew

KEYS = ("report", "final_citations", "metrics", "gaps", "table", "evidence_table", "sentiment_table",
        "themes", "opinion_summaries", "unaddressed")


def frame(table) -> pd.DataFrame:
    return pd.DataFrame(table.rows, columns=table.columns)


st.set_page_config(page_title="Market Research Crew", layout="wide")
st.title("Market Research Crew")
st.caption(
    "Planner → official-site finder → web researchers (facts, official sites only) + App Store lookup (apps) + community "
    "researchers (opinions) → Opinion Processor → Analyst → Writer"
)

with st.sidebar:
    provider = st.selectbox("LLM provider", ["gemini", "grok"])
    apps = st.selectbox("App Store data", ["auto", "on", "off"], help="auto = only when the planner sees that the products are apps.")
    community = st.toggle("Community opinions (Hacker News, Reddit)", value=True, help="Off = facts only.")
    sentiment = st.selectbox("Sentiment scorer", ["hf", "llm"], help="hf = local Hugging Face model (falls back to llm if not installed).")
    st.caption("Keys come from your .env file (GEMINI_API_KEY / XAI_API_KEY, TAVILY_API_KEY).")

question = st.text_area("Research question", "Compare Notion, Coda and ClickUp for small teams", height=80)
dims_text = st.text_input("Dimensions you insist on (optional, comma-separated)", placeholder="pricing, security")
sites_text = st.text_input("Official sites override (optional)", placeholder="notion=notion.com, coda=coda.io",
                           help="Only needed if the automatic lookup picked the wrong site (see Run info).")

if st.button("Run research", type="primary") and question.strip():
    settings = get_settings(
        provider=provider, app_mode=apps,
        enable_community=community, sentiment_backend=sentiment,
    )
    user_dims = [d.strip() for d in dims_text.split(",") if d.strip()] or None
    try:
        deps = build_deps(settings)
    except LLMError as exc:
        st.error(str(exc))
        st.stop()

    final: dict = {"claims": {}, "trace": []}
    with st.status("Crew is working…", expanded=True) as status:
        try:
            for node, update in stream_crew(question, deps, user_dims, parse_sites(sites_text)):
                if not update:
                    continue
                for line in update.get("trace", []):
                    st.write(f"**{node}**: {line}")
                    final["trace"].append(line)
                for c in update.get("claims", []):
                    final["claims"][c.id] = c  # later updates (fact-check verdicts) replace earlier ones
                for key in KEYS:
                    if key in update:
                        final[key] = update[key]
            status.update(label="Done", state="complete")
        except BudgetExceeded as exc:
            status.update(label="Stopped: call budget reached", state="error")
            st.error(f"{exc}. Raise MAX_LLM_CALLS in .env, or ask for fewer dimensions.")
            st.stop()
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Run stopped: {exc}")
            st.stop()

    st.subheader("Report")
    st.markdown(final.get("report", "_No report produced._"))
    st.download_button("Download report (.md)", final.get("report", ""), file_name="report.md")

    facts_tab, users_tab, evidence_tab, run_tab = st.tabs(
        ["Claims and verdicts", "What users say", "Claims vs user evidence", "Run info"]
    )
    with facts_tab:
        rows = [
            {"status": c.status.value, "entity": c.entity, "dimension": c.dimension, "claim": c.text,
             "source": c.source_url, "note": c.check_note}
            for c in final["claims"].values()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
        if "table" in final:
            st.caption("Comparison table (verified facts only)")
            st.dataframe(frame(final["table"]), width="stretch")
    with users_tab:
        if "sentiment_table" in final:
            st.caption("Public sentiment, counted per discussion thread (sampled threads only, a small self-selected sample)")
            st.dataframe(frame(final["sentiment_table"]), width="stretch")
        themes = final.get("themes", [])
        if themes:
            st.dataframe(pd.DataFrame([
                {"entity": t.entity, "dimension": t.dimension, "theme": t.label, "stance": t.stance, "threads": t.n_threads,
                 "confidence": t.confidence, "example": t.examples[0].quote if t.examples else "",
                 "link": t.examples[0].url if t.examples else ""}
                for t in themes
            ]), width="stretch")
        else:
            st.info("No recurring user themes passed the minimum-threads bar.")
    with evidence_tab:
        if "evidence_table" in final:
            st.dataframe(frame(final["evidence_table"]), width="stretch")
        else:
            st.info("No fact-to-evidence mapping (community track off or no verified facts).")
    with run_tab:
        st.json({**final.get("metrics", {}), **deps.llm.usage()})
        if final.get("gaps"):
            st.subheader("Gaps")
            st.write("\n".join(f"- {g}" for g in final["gaps"]))