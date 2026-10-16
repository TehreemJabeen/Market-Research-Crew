import ipaddress
import re
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; MarketResearchCrew/0.1)"


def is_public_url(url: str) -> bool:
    """Basic SSRF guard: http(s) only, and the host must not resolve to a private address."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return True  # can't resolve locally (e.g. behind a proxy); the request will fail on its own
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def fetch_text(url: str, timeout: float = 15.0, max_chars: int = 20000) -> str:
    """Download a page and return its visible text ('' on any failure)."""
    if not is_public_url(url):
        return ""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except httpx.HTTPError:
        return ""
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return text[:max_chars]
