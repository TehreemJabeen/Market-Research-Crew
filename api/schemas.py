"""Request and response shapes for the web API."""
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.tools.domains import host_of

DOMAIN_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")
Status = Literal["queued", "running", "done", "failed", "stopped", "cancelled"]


class RunRequest(BaseModel):
    question: str = Field(min_length=8, max_length=500)
    dimensions: list[str] = Field(default_factory=list, max_length=6)
    sites: dict[str, str] = Field(default_factory=dict, max_length=6)  # entity -> official domain
    provider: Literal["gemini", "grok"] | None = None
    community: bool = True
    apps: Literal["auto", "on", "off"] = "auto"
    sentiment_backend: Literal["hf", "llm"] | None = None

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 8:
            raise ValueError("question is too short")
        return v

    @field_validator("dimensions")
    @classmethod
    def _dimensions(cls, v: list[str]) -> list[str]:
        out = [" ".join(d.split()) for d in v if d and d.strip()]
        if any(len(d) > 40 for d in out):
            raise ValueError("a dimension name is longer than 40 characters")
        return out

    @field_validator("sites")
    @classmethod
    def _sites(cls, v: dict[str, str]) -> dict[str, str]:
        out = {}
        for entity, domain in v.items():
            host = host_of(domain)
            if not entity.strip() or len(entity) > 60 or not DOMAIN_RE.match(host):
                raise ValueError(f"'{entity}={domain}' is not an entity and a domain such as notion=notion.com")
            out[entity.strip()] = host
        return out


class RunSummary(BaseModel):
    id: str
    question: str
    status: Status
    created_at: float
    finished_at: float | None = None
    seconds: float | None = None
    llm_calls: int | None = None
    tavily_calls: int | None = None


class RunDetail(RunSummary):
    options: dict
    error: str | None = None
    usage: dict | None = None
    result: dict | None = None


class RunCreated(BaseModel):
    run_id: str
