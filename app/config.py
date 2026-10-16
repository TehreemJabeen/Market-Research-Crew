#app/config.py: Holds all settings and limits, such as the maximum number of dimensions and the LLM call budget. It reads overrides from .env.
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    # LLM
    provider: str = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    grok_model: str = "grok-4-fast"
    gemini_api_key: str = ""
    xai_api_key: str = ""
    rpm: int = 8  # free tiers are rate limited
    max_llm_calls: int = 120  # hard budget per run

    # Search
    tavily_api_key: str = ""

    # Planner
    max_dimensions: int = 6
    min_dimensions: int = 3  # topped up from lower-scored candidates if fewer pass the threshold
    min_dimension_score: float = 3.0  # candidate score needed to be chosen (see planner.score_candidate)
    max_subquestions: int = 8  # total cap; every dimension still keeps at least one
    subq_per_dim_max: int = 2
    enable_scout: bool = True  # search for comparison pages first; off = model's own knowledge only
    max_scout_queries: int = 3
    max_scout_sources: int = 5
    scout_chars: int = 3000

    # Facts track
    max_sources_per_task: int = 3
    source_chars: int = 8000  # characters of each source shown to the LLM and kept for verification
    min_verified_per_task: int = 1
    max_loops: int = 2  # max re-research rounds
    enable_factchecker: bool = True  # turn off for the ablation
    fuzzy_threshold: float = 0.9  # quote-match score needed to count as "found in source"

    # Opinions track
    enable_community: bool = True  # turn off for the ablation (facts only)
    max_posts_per_task: int = 25  # posts kept per dimension after filtering
    post_chars: int = 1500  # characters of each post shown to the LLM and kept for verification
    max_snippets_per_dimension: int = 12
    community_max_age_days: int = 1095  # ignore posts older than ~3 years
    max_snippets_for_themes: int = 40
    theme_min_threads: int = 2  # a theme seen in fewer discussion threads is dropped as anecdotal
    sentiment_backend: str = "hf"  # hf | llm ; hf falls back to llm if transformers/torch are not installed
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _flag(value: str) -> bool:
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def get_settings(**overrides) -> Settings:
    """Build settings from environment variables (and an optional .env file)."""
    load_dotenv()
    env = os.getenv
    settings = Settings(
        provider=env("LLM_PROVIDER", "gemini").lower(),
        gemini_model=env("GEMINI_MODEL", "gemini-2.5-flash"),
        grok_model=env("GROK_MODEL", "grok-4-fast"),
        gemini_api_key=env("GEMINI_API_KEY", ""),
        xai_api_key=env("XAI_API_KEY", ""),
        tavily_api_key=env("TAVILY_API_KEY", ""),
        rpm=int(env("LLM_RPM", "8")),
        max_llm_calls=int(env("MAX_LLM_CALLS", "120")),
        enable_community=_flag(env("ENABLE_COMMUNITY", "1")),
        sentiment_backend=env("SENTIMENT_BACKEND", "hf").lower(),
        sentiment_model=env("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    )
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    return settings
