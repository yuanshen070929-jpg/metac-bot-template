"""Find and read the web pages a question names as its resolution source.

Many questions say exactly where the answer will come from (a statistics page,
an official tracker, a leaderboard). Reading that page directly is often the
single most useful piece of research, and it is cheap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s<>\"'\[\]{}|\\^`]+")
_TRAILING = ".,;:!?*_~"
_SKIP_HOST_SUFFIXES = ("metaculus.com",)
_SKIP_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".mp4", ".zip", ".xlsx", ".xls")
_TEXT_TYPES = ("text/html", "text/plain", "application/json", "application/xml", "text/xml", "text/csv",
               "application/xhtml+xml")
USER_AGENT = "Mozilla/5.0 (compatible; forecasting-research-bot/0.1; +https://www.metaculus.com/futureeval/)"


def _clean_url(url: str) -> str:
    while url:
        if url[-1] in _TRAILING:
            url = url[:-1]
        elif url.endswith(")") and url.count("(") < url.count(")"):
            url = url[:-1]
        else:
            break
    return url


def extract_urls(*texts: str | None, limit: int | None = None) -> list[str]:
    """URLs in reading order, de-duplicated, without Metaculus or media links."""
    seen: set[str] = set()
    urls: list[str] = []
    for text in texts:
        if not text:
            continue
        for match in _URL_RE.finditer(text):
            url = _clean_url(match.group(0))
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            if not host or any(host == s or host.endswith("." + s) for s in _SKIP_HOST_SUFFIXES):
                continue
            if parsed.path.lower().endswith(_SKIP_EXTENSIONS):
                continue
            key = url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            urls.append(url)
            if limit is not None and len(urls) >= limit:
                return urls
    return urls


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe", "nav", "footer", "form"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
              "table", "ul", "ol", "dd", "dt", "header", "main", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")
        elif tag in {"td", "th"}:
            self._parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n")]
        out: list[str] = []
        for line in lines:
            if line or (out and out[-1]):
                out.append(line)
        return "\n".join(out).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed HTML: keep whatever was parsed
        logger.debug("HTML parse error", exc_info=True)
    body = parser.text()
    if parser.title and not body.startswith(parser.title):
        return f"{parser.title}\n\n{body}"
    return body


def _decode(raw: bytes, content_type_header: str) -> str:
    """Declared charset first, then UTF-8; requests' ISO-8859-1 default garbles most pages."""
    match = re.search(r"charset=[\"']?([\w-]+)", content_type_header, re.I)
    for encoding in [match.group(1) if match else None, "utf-8"]:
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def fetch_text(url: str, timeout_s: float = 15, max_bytes: int = 2_000_000, max_chars: int = 15_000) -> str | None:
    """Plain text of a page, or None if it is not a readable text page."""
    response = requests.get(
        url,
        timeout=timeout_s,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/json;q=0.9,*/*;q=0.5"},
        stream=True,
        allow_redirects=True,
    )
    try:
        if response.status_code >= 400:
            logger.info(f"Resolution source {url} returned HTTP {response.status_code}")
            return None
        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type and not content_type.startswith(_TEXT_TYPES):
            logger.info(f"Resolution source {url} is {content_type}; not read")
            return None
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=65536):
            chunks.append(chunk)
            size += len(chunk)
            if size >= max_bytes:
                break
        raw = b"".join(chunks)
        text = _decode(raw, response.headers.get("Content-Type") or "")
    finally:
        response.close()

    if "json" in content_type:
        try:
            text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
        except ValueError:
            pass
    elif "html" in content_type or text.lstrip().startswith("<"):
        text = html_to_text(text)
    text = text.strip()
    if len(text) < 40:
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... page truncated ...]"
    return text


async def fetch_text_async(url: str, **kwargs) -> str | None:  # noqa: ANN003
    return await asyncio.to_thread(fetch_text, url, **kwargs)
