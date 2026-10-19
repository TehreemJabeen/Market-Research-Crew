"""Apple App Store facts from Apple's free iTunes Search API (no key): rating, price, last update.

Apple does not publish install counts or in-app purchase details through this API, so they are never reported.
Claims are built in code from the API response; each claim's quote is a line of the text block rendered from that
response, so nothing here involves a model.
"""
import time
from datetime import date

import httpx

from app.state import Claim, ClaimStatus
from app.tools.domains import name_key, on_domain

APP_DIMENSION = "app store ratings"
ITUNES_URL = "https://itunes.apple.com/search"


def app_enabled(settings, is_app: bool) -> bool:
    """`auto` follows the planner's is_app flag; `on` / `off` force it."""
    mode = (settings.app_mode or "auto").lower()
    return mode == "on" or (mode == "auto" and is_app)


def itunes_search(term: str, country: str = "us", limit: int = 5) -> list[dict]:
    params = {"term": term, "country": country, "media": "software", "entity": "software", "limit": limit}
    for attempt in range(2):
        resp = httpx.get(ITUNES_URL, params=params, timeout=20)
        if resp.status_code in (403, 429) and attempt == 0:  # Apple's soft rate limit (about 20 calls a minute)
            time.sleep(5)
            continue
        resp.raise_for_status()
        return resp.json().get("results", [])
    return []


def pick_app(entity: str, results: list[dict], domains: list[str]) -> tuple[dict, bool] | None:
    """The app whose name contains the entity's name. One whose developer website is on the entity's official
    domain wins; otherwise the first name match is used and flagged as unconfirmed. Returns (app, confirmed)."""
    key = name_key(entity)
    named = [
        r for r in results
        if key and (key in name_key(r.get("trackName", "")) or key in name_key(r.get("artistName", "")))
    ]
    for r in named:
        if domains and r.get("sellerUrl") and on_domain(r["sellerUrl"], domains):
            return r, True
    return (named[0], False) if named else None


def price_dimension(dimensions: list[str]) -> str:
    """Price belongs under the planned pricing dimension if there is one."""
    return next((d for d in dimensions if any(w in name_key(d) for w in ("pric", "cost"))), APP_DIMENSION)


def _price(app: dict) -> str:
    if app.get("formattedPrice"):
        return app["formattedPrice"]
    price = app.get("price")
    return "Free" if not price else f"{price} {app.get('currency', '')}".strip()


def build_app_claims(entity: str, app: dict, confirmed: bool, domains: list[str], dimensions: list[str],
                     country: str, today: str | None = None) -> tuple[list[Claim], str, str]:
    """Returns (claims, the text block they quote from, the listing url)."""
    today = today or date.today().isoformat()
    store = f"{country.upper()} App Store"
    url = app.get("trackViewUrl") or f"{ITUNES_URL}?term={entity}&country={country}"
    count, avg = app.get("userRatingCount") or 0, app.get("averageUserRating")
    rating_line = (
        f"Average rating: {avg:.1f} out of 5 from {count:,} ratings" if count and avg is not None
        else "Average rating: no ratings yet"
    )
    cur_avg, cur_count = app.get("averageUserRatingForCurrentVersion"), app.get("userRatingCountForCurrentVersion") or 0
    current_line = f"Current version rating: {cur_avg:.1f} out of 5 from {cur_count:,} ratings" if cur_avg is not None and cur_count else ""
    price_line = f"Price: {_price(app)} ({store})"  # long enough for the quote check (it ignores quotes under 20 characters)
    updated = (app.get("currentVersionReleaseDate") or "")[:10]
    version_line = f"Version {app['version']}, last updated {updated}" if app.get("version") and updated else ""
    lines = [
        f"App Store listing: {app.get('trackName', entity)} by {app.get('sellerName') or app.get('artistName', 'unknown')} "
        f"({store}, retrieved {today})",
        rating_line, current_line, price_line, version_line,
    ]
    text = "\n".join(line for line in lines if line)

    caveat = "" if confirmed or not domains else " (the developer's website did not match the official site, so this may be a different app)"
    stamp = f"on the {store}, as of {today}"
    rating = (
        f"{entity} has no ratings yet {stamp}" if not (count and avg is not None)
        else f"{entity} has an average rating of {avg:.1f} out of 5 from {count:,} ratings {stamp}"
        + (f" ({cur_avg:.1f} from {cur_count:,} ratings for the current version)" if current_line else "")
    )
    specs = [(APP_DIMENSION, rating, rating_line), (price_dimension(dimensions), f"{entity} is listed at {_price(app)} {stamp}", price_line)]
    if version_line:
        specs.append((APP_DIMENSION, f"{entity}'s latest version is {app['version']}, last updated {updated} ({stamp})", version_line))
    note = f"Apple iTunes Search API, {country.upper()} storefront, retrieved {today}; built from the API response, not independently checked"
    claims = [
        Claim(
            id=f"app-{name_key(entity)}-{i + 1}", task_id="app", entity=entity, dimension=dim, text=claim_text + caveat + ".",
            source_url=url, supporting_quote=quote, status=ClaimStatus.VERIFIED, check_note=note, source_type="appstore",
        )
        for i, (dim, claim_text, quote) in enumerate(specs)
    ]
    return claims, text, url