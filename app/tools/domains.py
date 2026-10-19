"""Small, pure helpers for working with website domains (used to find and enforce official sites)."""
import re
from urllib.parse import urlparse

# Sites that write ABOUT products rather than being the product's own site. Never treated as "official".
NOT_OFFICIAL = {
    "wikipedia.org", "g2.com", "capterra.com", "trustpilot.com", "reddit.com", "youtube.com", "linkedin.com",
    "facebook.com", "twitter.com", "x.com", "instagram.com", "medium.com", "quora.com", "producthunt.com",
    "softwareadvice.com", "getapp.com", "trustradius.com", "gartner.com", "forbes.com", "techcrunch.com",
    "pcmag.com", "zapier.com", "github.com", "crunchbase.com", "ycombinator.com", "stackshare.io", "alternativeto.net",
    "sourceforge.net", "slashdot.org", "play.google.com", "apps.apple.com", "amazon.com", "wikimedia.org",
}
TWO_PART_SUFFIXES = {"co.uk", "com.au", "co.in", "co.jp", "com.br", "co.nz", "com.pk", "org.uk"}


def host_of(url: str) -> str:
    try:
        host = (urlparse(url if "://" in url else "https://" + url).hostname or "").casefold()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def registered_domain(url_or_host: str) -> str:
    """'https://docs.notion.so/x' -> 'notion.so'. Handles a few two-part suffixes such as co.uk."""
    parts = host_of(url_or_host).split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    n = 3 if ".".join(parts[-2:]) in TWO_PART_SUFFIXES else 2
    return ".".join(parts[-n:])


def second_level_label(domain: str) -> str:
    """'docs.notion.so' -> 'notion'; 'bbc.co.uk' -> 'bbc'."""
    return registered_domain(domain).split(".")[0]


def on_domain(url: str, domains: list[str]) -> bool:
    """True if the url's host is one of the domains or a subdomain of one (docs.notion.so is on notion.so)."""
    host = host_of(url)
    return bool(host) and any(host == d or host.endswith("." + d) for d in (host_of(d) for d in domains) if d)


def is_third_party(domain: str) -> bool:
    return registered_domain(domain) in NOT_OFFICIAL


def name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def name_match(entity: str, domain: str) -> int:
    """2 = the domain's name IS the entity name (notion.so for Notion); 1 = one contains the other; 0 = no match."""
    token, label = name_key(entity), name_key(second_level_label(domain))
    if not token or not label:
        return 0
    if token == label:
        return 2
    if token in label or (len(label) >= 4 and label in token):
        return 1
    return 0