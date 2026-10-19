"""App Store agent: one run per entity. Reads the entity's Apple App Store listing (iTunes Search API) and writes
claims about its rating, price and last update. No LLM, no Tavily: one free HTTP call per entity.
Installs and in-app purchases are not published by Apple, so they are not reported."""
from typing import TypedDict

from app.tools.appstore import build_app_claims, pick_app


class AppInput(TypedDict):
    entity: str
    domains: list[str]
    dimensions: list[str]


def run_appstore(payload, deps) -> dict:
    entity, domains, dimensions = payload["entity"], payload.get("domains") or [], payload.get("dimensions") or []
    country = deps.settings.app_country.lower()
    label = f"appstore[{entity}]"
    if deps.apps is None:
        return {"trace": [f"{label}: no App Store source configured"]}
    try:
        results = deps.apps(entity, country)
    except Exception as exc:  # one failed lookup must not sink the run
        return {"trace": [f"{label}: lookup failed ({exc})"]}
    picked = pick_app(entity, results, domains)
    if picked is None:
        return {"trace": [f"{label}: no matching iOS app found (it may be web-only or Android-only)"]}
    app, confirmed = picked
    claims, text, url = build_app_claims(entity, app, confirmed, domains, dimensions, country)
    how = "developer site matched" if confirmed else "name match only"
    return {"claims": claims, "sources": {url: text}, "trace": [f"{label}: '{app.get('trackName')}' ({how}) -> {len(claims)} claims"]}