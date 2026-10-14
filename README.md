# Market Research Crew

A multi-agent system (LangGraph + Python) that turns a research question into a **cited report that keeps facts and
public opinion apart**. It runs on **Gemini's free tier** (or Grok), Tavily's free search tier, Hacker News and,
optionally, a local Hugging Face sentiment model.

> Example: *"Compare Notion, Coda and ClickUp for small teams"* → the planner discovers which dimensions matter →
> a **facts track** (web pages, every claim checked against its source) and an **opinions track** (Hacker News and
> Reddit threads, quotes verified, sentiment scored) run in parallel → the analyst links verified facts to what
> users say → a report with separate, cited fact and opinion voices.

## How it works

```
                 ┌────────────── FACTS TRACK ──────────────┐
                 │ Web researcher ×N ──► Fact-Checker ─┐   │  (capped re-research loop)
Planner ─► fan-out                                     ├─► Gate ─► Analyst ─► Writer ─► Editor
                 │ Community researcher ×D ─► Opinion  ┘   │
                 └────────────── OPINIONS TRACK ───────────┘
```

There is no router: both tracks always run, because they do different jobs.

| Agent | Job | AI or code? |
|---|---|---|
| Planner | Rewrites the question, scouts comparison pages, picks 3-6 value-adding dimensions, writes ≥1 sub-question each | LLM judges; **code** scores, selects, checks coverage |
| Web researcher (×N) | Searches, reads pages, extracts claims with a verbatim quote and source URL | LLM + search/fetch |
| Fact-Checker | Does the quote exist in the fetched page? Does it support the claim or contradict another? | **code**, then LLM |
| Community researcher (×D, one per dimension) | Collects Hacker News (Algolia API) and Reddit (via Tavily domain filter) posts, filters short, promotional, old and duplicate ones, extracts opinion snippets | code filters; LLM extracts; **code verifies each quote is in the post** |
| Opinion processor | Scores polarity, groups snippets into themes, aggregates | hybrid, see below; **code** aggregates |
| Gate | Waits for both tracks, then starts the analyst exactly once | **code** |
| Analyst | Links verified facts to user themes (confirms / contradicts / qualifies), flags user concerns no fact addresses, summarises public opinion | LLM relates; **code** counts and builds tables |
| Writer | Drafts with separate fact and opinion sections, using only numbered claims and themes | LLM |
| Editor | Strips invalid citations, drops uncited factual and unattributed opinion sentences, renumbers, appends Sources | **code** (+ one LLM repair pass) |

### Hybrid sentiment

- The LLM extracts snippets and themes. It does not decide the numbers.
- A local Hugging Face model (default `cardiffnlp/twitter-roberta-base-sentiment-latest`) scores plain snippets.
  `SENTIMENT_BACKEND=llm` swaps in an LLM scorer; if `transformers`/`torch` are missing the app falls back to the
  LLM scorer by itself and says so in the run info.
- **Comparative snippets** ("switched from A to B, much better") go to the LLM scorer, because a text-only model
  cannot know which entity is favoured.
- Code aggregates: sentiment is the mean of per-thread means (one noisy thread cannot dominate); a theme's stance
  needs a net score ≥ 0.34; confidence is low / medium / high for under 3, 3-4, 5+ threads; themes seen in fewer
  than 2 threads are dropped as anecdotal. Results are scoped to the sample ("in N threads"), never "the market".

### Design decisions worth knowing

- **Code checks the AI's work.** Quote verification is fuzzy string matching (threshold 0.9), not another opinion,
  so a hallucinated quote cannot pass, for facts or for community quotes.
- **Facts and opinions are different jobs.** Web claims get a truth check against their source. Opinions cannot be
  "true or false", so they get authenticity (the quote exists), filtering, thread weighting and a confidence label.
- **Claim lifecycle** (`unverified → verified / rejected / conflicting`); the Writer only sees verified claims.
- **A safe join.** The gate node plus `gate_route` start the analyst once, only when both tracks are ready, even
  while the fact-check retry loop is still sending work back. A test fails if the gate is replaced by a naive join.
- **Capped loops and budget.** Re-research runs at most `max_loops` times; `MAX_LLM_CALLS` (default 120) and a graph
  recursion limit stop runaway runs. A `BudgetExceeded` is never swallowed by an agent.
- **One failed researcher does not sink the run**; the error goes into the trace.
- **Free-tier aware.** Rate limiting (`LLM_RPM`), retry with backoff on 429/5xx, call and token counting.

### The Planner (dimension discovery)

A user's question rarely contains the criteria that matter, so the Planner finds them:

1. **Analyze (LLM):** rewrites the question, infers audience and decision, proposes scouting searches.
2. **Scout (code):** searches and reads real comparison / buyer's-guide pages.
3. **Propose (LLM):** 6-10 candidate dimensions rated 1-5 on relevance, differentiation and answerability, citing
   which scouted pages compare on each.
4. **Select (code):** score = 0.4·relevance + 0.3·differentiation + 0.3·answerability + 0.25 per scouted page
   (max +1). Fake URLs earn nothing. Names are deduplicated, user-forced dimensions (`--dimensions`) come first,
   and the best 3-6 are kept.
5. **Sub-questions (LLM) and coverage check (code):** every dimension gets at least one sub-question; a miss
   triggers one repair call, then a template question.

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt   # optional: local sentiment model (large, pulls torch)
cp .env.example .env                 # then fill in the keys
```

| Key | Where to get it |
|---|---|
| `GEMINI_API_KEY` | free, https://aistudio.google.com/apikey |
| `TAVILY_API_KEY` | free tier, https://tavily.com |
| `XAI_API_KEY` | only if you want Grok (`LLM_PROVIDER=grok`) |

Hacker News needs no key. Model names change; if one is rejected, set `GEMINI_MODEL` / `GROK_MODEL` in `.env`.

## Run

```bash
python -m app.main "Compare Notion, Coda and ClickUp for small teams"     # writes outputs/<slug>.md
python -m app.main "..." --dimensions pricing,security                     # force dimensions
python -m app.main "..." --no-community                                    # facts only
python -m app.main "..." --sentiment-backend llm                           # no local model
streamlit run ui/streamlit_app.py                                          # live UI
docker build -t market-research-crew .                                     # add --build-arg WITH_ML=1 for the HF model
docker run --env-file .env -p 8501:8501 market-research-crew
```

Other flags: `--provider gemini|grok`, `--no-factcheck` (ablation), `--no-scout`, `--out DIR`. Exit code 3 means the
LLM-call budget was hit. Expect roughly 45-70 LLM calls per full run; at 8 requests/minute that is several minutes.

## Tests (no API keys needed)

```bash
python -m pytest
```

The tests run the whole graph with scripted fakes for the LLM, search, fetch, community sources and sentiment
scorer. They cover rejected hallucinated quotes, the re-research loop, the gate join, the opinions pipeline end to
end, fact-to-evidence links, a facts-only run, citation stripping, the planner's scoring and coverage repair,
opinion filtering and aggregation, the hybrid scorer's fallbacks, and the call budget.

## Evals

```bash
python -m evals.run_evals --limit 2       # crew vs crew_no_factcheck vs crew_no_community vs single_agent
python -m evals.planner_eval --limit 3    # dimension recall/precision: scouted vs llm_only
python -m evals.sentiment_eval            # HF model(s) vs LLM scorer on labelled snippets
python -m evals.theme_review export "Compare Notion, Coda and ClickUp for small teams"   # then label the CSV
python -m evals.theme_review score evals/theme_review.csv
```

| Eval | Measures |
|---|---|
| `run_evals` | citation accuracy (does the quote appear in the cited page?), coverage, themes, source domains, calls, tokens, seconds; ablations for the fact-checker and the opinions track |
| `planner_eval` | recall and precision of planned dimensions against expert dimensions |
| `sentiment_eval` | accuracy, macro-F1, plain vs comparative accuracy, seconds and LLM calls per scorer |
| `theme_review` | your y/n judgement of theme faithfulness and of the fact-to-theme links |

**Replace the starter data before quoting numbers:** `evals/expert_dimensions.json` holds starter labels and
`evals/sentiment_labels.json` holds 27 synthetic examples. Use dimensions from human-written comparison reports and
about 50 real, hand-labelled snippets from your own runs.

## Known limitations

- **Not yet run against live services in development.** The Gemini/Grok, Tavily, Hacker News and Reddit calls and
  the Hugging Face model download were exercised only through fakes. Run one real question and the evals first.
- Community data is a sample (Hacker News skews technical; Reddit via Tavily returns search snippets, not full
  threads). G2 and Capterra are deliberately skipped because of bot blocking and terms of service.
- A local polarity model cannot judge comparative sentences; those are routed to the LLM, which costs calls.
- Pages that need JavaScript or a login are unreadable; the search snippet is the fallback.
- The citation check proves a quote exists in its source, and an LLM judges whether it supports the claim. It does
  not prove the source is right.
- The uncited and unattributed sentence checks are sentence-level heuristics.
- Entity and dimension labels from the LLM are mapped onto the planner's lists by simple matching.

## Resume line

"Built a multi-agent market-research system (LangGraph, Gemini, Tavily, Pydantic, Hugging Face) with parallel fact
and opinion tracks: a fact-checking agent that verifies every quote against its source, a community-opinion
pipeline with hybrid LLM + local-model sentiment, and an analyst that links verified facts to user evidence;
reduced unsupported citations from X% to Y% versus a single-agent baseline across N benchmark questions."
*(fill in X, Y, N from `evals/results.md`)*
