# Market Research Crew

A multi-agent system (LangGraph + Python) that turns a research question into a **cited report that keeps facts and
public opinion apart**. It runs on **Gemini's free tier** (or Grok), Tavily's free search tier, Hacker News and,
optionally, a local Hugging Face sentiment model.

> Example: *"Compare Notion, Coda and ClickUp for small teams"* → the planner discovers which dimensions matter →
> a **facts track** (official websites, plus App Store data for apps) and an **opinions track** (Hacker News and
> Reddit threads, quotes verified, sentiment scored) run in parallel → the analyst links verified facts to what
> users say → a report with separate, cited fact and opinion voices.

## How it works

```
                          ┌────────────── FACTS TRACK ──────────────┐
                          │ Web researcher ×N (official sites only) ─┐  │
                          │ App Store lookup ×E (apps only) ─────────┤  │
Planner ─► Site finder ─► fan-out                                    ├─► Gate ─► Analyst ─► Writer
                          │ Community researcher ×D ─► Opinion ──────┘  │
                          └────────────── OPINIONS TRACK ───────────┘
```

There is no router: both tracks always run, because they do different jobs.

| Agent | Job | AI or code? |
|---|---|---|
| Planner | Rewrites the question, picks 3-6 value-adding dimensions, writes ≥1 sub-question each (3 LLM calls, no searches) | LLM proposes; **code** keeps your numbers, merges forced dimensions, checks coverage |
| Site finder | Finds each entity's official website once (`--sites notion=notion.com` overrides it) | **code** + Tavily; a model picks only when no result matches the name |
| Web researcher (×N) | Searches only the official site(s) of the entities in its sub-question, reads the pages, extracts claims with a verbatim quote and source URL. One LLM call per task, no retries | search/fetch by **code**; LLM only reads and extracts |
| App Store lookup (×E, one per entity; only for apps) | Reads each entity's Apple App Store listing: average rating and count, price, last update. Installs and in-app purchases are not published by Apple, so they are not reported | **code** (free iTunes Search API, no LLM, no Tavily) |
| Community researcher (×D, one per dimension) | Collects Hacker News (Algolia API) and Reddit (via Tavily domain filter) posts, filters short, promotional, old and duplicate ones, extracts opinion snippets | code filters; LLM extracts; **code verifies each quote is in the post** |
| Opinion processor | Scores polarity, groups snippets into themes, aggregates | hybrid, see below; **code** aggregates |
| Gate | Waits for both tracks, then starts the analyst exactly once | **code** |
| Analyst | Links verified facts to user themes (confirms / contradicts / qualifies), flags user concerns no fact addresses, summarises public opinion | LLM relates; **code** counts and builds tables |
| Writer | Drafts with separate fact and opinion sections, using only numbered claims and themes; returns the report and the list of sources behind its [n] markers | LLM |

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

- **Code checks the AI's work, for community quotes.** Each snippet's quote must appear in its post (fuzzy match,
  threshold 0.9). **There is currently no fact-checker or editor:** web claims reach the report as the model wrote
  them, the report has no Sources section, and its [n] markers are not cleaned. The writer does return the list of
  sources behind each marker, so `run_evals` can still measure how many quotes are really on their page.
- **Facts and opinions are different jobs.** Facts come from official sites and the App Store. Opinions cannot be
  "true or false", so they get authenticity (the quote exists), filtering, thread weighting and a confidence label.
- **Claim status.** With no fact-checker, the researcher marks its claims "verified" with the note "not
  fact-checked"; that label is a placeholder so the analyst and writer can use them. App Store claims are built in
  code from Apple's response.
- **Official sources only for facts.** Code limits every web search to the entity's official domain(s) and drops any
  page outside them, whatever the search tool returned. Opinions come from the community track instead. A task with
  no official page produces no claims (a gap in the report), not a fallback to blogs.
- **No retry loop.** A task that finds nothing is not searched again; the gap is reported.
- **A safe join.** The opinions track is one step longer than the others. The gate node plus `gate_route` start the
  analyst once, after it finishes, and straight after the facts tracks when the community track is switched off.
- **Budget.** `MAX_LLM_CALLS` (default 120) and a graph recursion limit stop runaway runs. A `BudgetExceeded` is
  never swallowed by an agent.
- **One failed researcher does not sink the run**; the error goes into the trace.
- **Free-tier aware.** Rate limiting (`LLM_RPM`), retry with backoff on 429/5xx, call and token counting.

### The Planner (dimension discovery)

A user's question rarely contains the criteria that matter, so the Planner works them out. It makes exactly three
LLM calls and no web searches:

1. **Rewrite (LLM):** a clear, self-contained question, plus entities, audience and decision. The prompt says to
   keep every number and budget and to leave the audience blank rather than guess. Code then checks that every
   number from your question is still in the rewrite and appends any that were lost.
2. **Dimensions (LLM):** 3-6 comparison dimensions, most important first, each with a one-line reason for this
   audience. Code merges in your forced dimensions (`--dimensions pricing,security`, placed first), drops
   overlapping names and trims to the limit.
3. **Sub-questions (LLM):** 1-2 per dimension. Code checks that every dimension is covered; a dimension left empty
   gets a template question (no extra call).

The dimensions come from the model's own knowledge. `agents/scout.py` (search and read comparison pages first) is
kept in the repo but is not called at the moment.

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
python -m app.main "..." --sites notion=notion.com,coda=coda.io           # fix a wrong official domain
python -m app.main "..." --apps                                            # always add App Store data (default: auto)
python -m app.main "..." --no-community                                    # facts only
python -m app.main "..." --sentiment-backend llm                           # no local model
streamlit run ui/streamlit_app.py                                          # live UI
docker build -t market-research-crew .                                     # add --build-arg WITH_ML=1 for the HF model
docker run --env-file .env -p 8501:8501 market-research-crew
```

Other flags: `--provider gemini|grok`, `--no-apps`, `--out DIR`. `APP_COUNTRY` (default `us`) sets the App Store storefront. Exit code 3 means the
LLM-call budget was hit. Expect roughly 40-65 LLM calls per full run; at 8 requests/minute that is several minutes. Tavily credits: about 1 per entity for the site lookup plus 1 per entity named in each sub-question (roughly 15-25 per run with 3 entities).

## Tests (no API keys needed)

```bash
python -m pytest
```

The tests run the whole graph with scripted fakes for the LLM, search, fetch, community sources and sentiment
scorer. They cover official-site-only research and the site finder, the gate join, the opinions pipeline end to
end, fact-to-evidence links, a facts-only run, the planner's number check, dimension tidy-up and coverage fallback,
opinion filtering and aggregation, the hybrid scorer's fallbacks, and the call budget.

## Evals

```bash
python -m evals.run_evals --limit 2       # crew vs crew_no_community vs single_agent
python -m evals.planner_eval --limit 3    # dimension recall/precision against expert dimensions
python -m evals.sentiment_eval            # HF model(s) vs LLM scorer on labelled snippets
python -m evals.theme_review export "Compare Notion, Coda and ClickUp for small teams"   # then label the CSV
python -m evals.theme_review score evals/theme_review.csv
```

| Eval | Measures |
|---|---|
| `run_evals` | citation accuracy (does the quote appear in the cited page?), coverage, themes, source domains, calls, tokens, seconds; an ablation for the opinions track |
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
- Official sites are marketing pages: strong for pricing, features and integrations, weak for performance and
  reliability. Check the `sites:` lines in the run info; a wrong or missing domain means no facts for that entity
  (fix it with `--sites`).
- Pages that need JavaScript or a login are unreadable; the search snippet is the fallback.
- With no fact-checker, nothing proves a web claim's quote exists in its source or supports the claim. It does
  not prove the source is right.
- App Store data covers iOS only, one storefront (`us` by default), and Apple's soft limit of about 20 calls a minute.
- Entity and dimension labels from the LLM are mapped onto the planner's lists by simple matching.

## Resume line

"Built a multi-agent market-research system (LangGraph, Gemini, Tavily, Pydantic, Hugging Face) with parallel fact
and opinion tracks: official-site research, an App Store data agent, a community-opinion
pipeline with hybrid LLM + local-model sentiment, and an analyst that links facts to user evidence;
measured citation accuracy against a single-agent baseline across N benchmark questions."
*(fill in the numbers from `evals/results.md`; the fact-checker is removed for now, so do not claim it)*