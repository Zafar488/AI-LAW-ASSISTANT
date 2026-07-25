"""Production-oriented legal knowledge-base loader.

This module can:
1. Extract text from local PDF files.
2. Crawl official legal websites listed in lawyer_links.json.
3. Split documents into retrieval-friendly chunks while preserving metadata.
4. Save resumable JSON checkpoints safely.
5. Optionally build a FAISS index.

Recommended links-only command:
    python -u -m app.loader --links-only --skip-embeddings --resume

The implementation intentionally keeps the public function names used by the
original project so it can replace app/loader.py with minimal integration work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse

import fitz
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import (
    BASE_DIR,
    INPUT_DIR,
    EMBEDDINGS_DIR,
    LINKS_JSON,
    OUTPUT_JSON,
    OFFICIAL_OUTPUT_JSON,
    PDF_UPDATES_JSON,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_PATH = Path(BASE_DIR)
INPUT_PATH = Path(INPUT_DIR)
EMBEDDINGS_PATH = Path(EMBEDDINGS_DIR)
LINKS_PATH = Path(LINKS_JSON)
OUTPUT_PATH = Path(OUTPUT_JSON)
OFFICIAL_OUTPUT_PATH = Path(OFFICIAL_OUTPUT_JSON)
PDF_UPDATES_PATH = Path(PDF_UPDATES_JSON)

LOG_DIR = BASE_PATH / "logs"
CRAWL_STATE_PATH = EMBEDDINGS_PATH / "crawl_state.json"
CRAWL_REPORT_PATH = EMBEDDINGS_PATH / "crawl_report.json"

CONNECT_TIMEOUT_SECONDS = float(os.getenv("LEGAL_CRAWLER_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT_SECONDS = float(os.getenv("LEGAL_CRAWLER_READ_TIMEOUT", "60"))
DEFAULT_DELAY_SECONDS = float(os.getenv("LEGAL_CRAWLER_DELAY_SECONDS", "0.5"))
DEFAULT_MAX_SOURCE_PAGES = int(os.getenv("MAX_SOURCE_PAGES", "500"))
MAX_RESPONSE_BYTES = int(os.getenv("LEGAL_CRAWLER_MAX_RESPONSE_MB", "60")) * 1024 * 1024
MIN_DOCUMENT_CHARACTERS = int(os.getenv("LEGAL_CRAWLER_MIN_DOCUMENT_CHARS", "120"))
MAX_REDIRECTS = int(os.getenv("LEGAL_CRAWLER_MAX_REDIRECTS", "10"))

USER_AGENT = os.getenv(
    "LEGAL_CRAWLER_USER_AGENT",
    "PakistanLegalResearchBot/2.0 (+local educational legal research; contact: local-admin)",
)

LAW_LINK_TERMS = (
    "act",
    "law",
    "laws",
    "code",
    "ordinance",
    "rule",
    "rules",
    "regulation",
    "regulations",
    "statute",
    "legislation",
    "gazette",
    "sro",
    "s.r.o",
    "publication",
    "download",
    "document",
    "pdf",
    "alphabet",
    "chronological",
    "category",
    "department",
    "page=",
    "constitution",
    "notification",
    "amendment",
)

NON_LAW_URL_TERMS = (
    "annual-report",
    "annual_report",
    "user-manual",
    "user_manual",
    "/manual",
    "/enforcement/",
    "/orders/",
    "/tender",
    "/careers",
    "/jobs",
    "/faq",
    "finance-bill",
    "draft-",
    "/news/",
    "/press-release",
)

SKIP_EXTENSIONS = {
    ".apk",
    ".exe",
    ".msi",
    ".dmg",
    ".iso",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}

BLOCKED_URL_FRAGMENTS = (
    "/apk/",
    "/apps/",
    "/download-app/",
    "drs-app-release",
    "javascript:",
    "mailto:",
    "tel:",
    "/login",
    "/signin",
    "/signup",
    "facebook.com",
    "twitter.com",
    "x.com/",
    "youtube.com",
    "instagram.com",
    "linkedin.com",
)

SUPPORTED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)

TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}

SECTION_PATTERNS = (
    re.compile(r"^\s*(?:section\s+)?(\d+[A-Z]?(?:-\d+)?)\s*[.:-]\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bSection\s+(\d+[A-Z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bArticle\s+(\d+[A-Z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bRule\s+(\d+[A-Z]?(?:-\d+)?)\b", re.IGNORECASE),
)

PAKISTAN_CODE_FALLBACK_URLS = (
    "https://pakistancode.gov.pk/",
    "https://pakistancode.gov.pk/english/LGu0xAD.php",
    "https://pakistancode.gov.pk/english/LGu0xBD.php",
    "https://pakistancode.gov.pk/english/LGu0xVD.php",
)

LOGGER = logging.getLogger("legal_loader")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SourceStats:
    name: str
    start_url: str
    jurisdiction: str = "Unknown"
    visited_pages: int = 0
    extracted_documents: int = 0
    accepted_documents: int = 0
    created_chunks: int = 0
    skipped_urls: int = 0
    failed_urls: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: utc_now_iso())
    finished_at: str | None = None

    def finish(self) -> None:
        self.finished_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_url": self.start_url,
            "jurisdiction": self.jurisdiction,
            "visited_pages": self.visited_pages,
            "extracted_documents": self.extracted_documents,
            "accepted_documents": self.accepted_documents,
            "created_chunks": self.created_chunks,
            "skipped_urls": self.skipped_urls,
            "failed_urls": self.failed_urls,
            "errors": self.errors,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(slots=True)
class FetchedDocument:
    url: str
    content_type: str
    title: str
    pages: list[dict[str, Any]]
    fetched_at: str

    @property
    def full_text(self) -> str:
        return "\n\n".join(str(page.get("text", "")) for page in self.pages).strip()


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def configure_logging(verbose: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_DIR / "loader.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def load_json_file(path: Path, *, default: Any = None) -> Any:
    """Load JSON safely, accepting UTF-8 files with or without a BOM."""
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalized_text_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def canonicalize_url(url: str) -> str:
    """Remove fragments/tracking parameters without changing meaningful query data."""
    clean_url, _ = urldefrag(str(url).strip())
    parsed = urlparse(clean_url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""

    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower() if parsed.hostname else ""
    if not host:
        return ""

    port = parsed.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_QUERY_KEYS
    ]
    query = urlencode(query_items, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def host_key(url_or_host: str) -> str:
    parsed = urlparse(url_or_host) if "://" in url_or_host else None
    host = (parsed.hostname if parsed else url_or_host) or ""
    host = host.lower().split(":", 1)[0]
    return host[4:] if host.startswith("www.") else host


def safe_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return max(minimum, default)


def should_skip_url(url: str) -> bool:
    canonical = canonicalize_url(url)
    if not canonical:
        return True

    lowered = canonical.casefold()
    parsed = urlparse(canonical)
    suffix = PurePosixPath(parsed.path).suffix.casefold()
    return suffix in SKIP_EXTENSIONS or any(fragment in lowered for fragment in BLOCKED_URL_FRAGMENTS)


def content_type_without_parameters(value: str) -> str:
    return value.split(";", 1)[0].strip().casefold()


def response_is_supported(response: requests.Response) -> bool:
    content_type = content_type_without_parameters(response.headers.get("Content-Type", ""))
    suffix = PurePosixPath(urlparse(response.url).path).suffix.casefold()
    if any(content_type.startswith(value) for value in SUPPORTED_CONTENT_TYPES):
        return True
    return suffix in {".pdf", ".html", ".htm", ".php", ".aspx", ".jsp", ".docx"}


def validate_chunk_settings() -> None:
    if int(CHUNK_SIZE) <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")
    if int(CHUNK_OVERLAP) < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative.")
    if int(CHUNK_OVERLAP) >= int(CHUNK_SIZE):
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


# ---------------------------------------------------------------------------
# HTTP and extraction
# ---------------------------------------------------------------------------
def create_scraping_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)

    session = requests.Session()
    session.max_redirects = MAX_REDIRECTS
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/pdf,"
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document;q=0.9,"
                "application/xml;q=0.8,*/*;q=0.5"
            ),
            "Accept-Language": "en-US,en;q=0.9,ur;q=0.6",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return session


def fetch_response(session: requests.Session, url: str) -> requests.Response | None:
    if should_skip_url(url):
        LOGGER.debug("Skipping blocked or unsupported URL: %s", url)
        return None

    canonical = canonicalize_url(url)
    try:
        response = session.get(
            canonical,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            allow_redirects=True,
            stream=True,
        )
    except requests.RequestException as exc:
        LOGGER.warning("Request failed: %s -> %s", canonical, exc)
        return None

    if response.status_code != 200:
        LOGGER.warning("HTTP %s: %s", response.status_code, canonical)
        response.close()
        return None

    if not response_is_supported(response):
        LOGGER.info(
            "Unsupported content type %s: %s",
            response.headers.get("Content-Type", "unknown"),
            canonical,
        )
        response.close()
        return None

    declared_length = response.headers.get("Content-Length")
    if declared_length:
        try:
            if int(declared_length) > MAX_RESPONSE_BYTES:
                LOGGER.warning("Response too large (%s bytes): %s", declared_length, canonical)
                response.close()
                return None
        except ValueError:
            pass

    body = bytearray()
    try:
        for block in response.iter_content(chunk_size=64 * 1024):
            if not block:
                continue
            body.extend(block)
            if len(body) > MAX_RESPONSE_BYTES:
                LOGGER.warning("Response exceeded maximum size: %s", canonical)
                response.close()
                return None
    except requests.RequestException as exc:
        LOGGER.warning("Failed while reading response: %s -> %s", canonical, exc)
        response.close()
        return None

    response._content = bytes(body)  # requests uses this field for .content/.text
    response._content_consumed = True
    response.close()
    return response


def fetch_with_fallbacks(
    session: requests.Session,
    urls: Sequence[str],
) -> requests.Response | None:
    for candidate in urls:
        response = fetch_response(session, candidate)
        if response is not None:
            return response
    return None


def _extract_html_title(soup: BeautifulSoup, fallback_url: str) -> str:
    if soup.title and soup.title.get_text(strip=True):
        return normalize_whitespace(soup.title.get_text(" ", strip=True))[:300]
    heading = soup.find(["h1", "h2"])
    if heading:
        return normalize_whitespace(heading.get_text(" ", strip=True))[:300]
    return urlparse(fallback_url).path.rsplit("/", 1)[-1] or fallback_url


def _clean_html_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = _extract_html_title(soup, "")

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "template",
            "svg",
            "canvas",
            "header",
            "footer",
            "nav",
            "aside",
            "form",
            "button",
        ]
    ):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    raw_lines = [normalize_whitespace(line) for line in main.get_text("\n").splitlines()]

    cleaned_lines: list[str] = []
    previous = None
    for line in raw_lines:
        if not line or line == previous:
            continue
        if len(line) == 1 and not line.isdigit():
            continue
        cleaned_lines.append(line)
        previous = line

    return normalize_whitespace("\n".join(cleaned_lines)), title


def _extract_docx_text(content: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        LOGGER.warning("DOCX extraction failed: %s", exc)
        return ""

    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"</w:tr>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    text = BeautifulSoup(xml, "xml").get_text(" ")
    return normalize_whitespace(text)


def _rapidocr_result_text(result: Any) -> str:
    texts = getattr(result, "txts", None)
    if texts:
        return "\n".join(str(item) for item in texts if str(item).strip())

    if isinstance(result, tuple) and result:
        first = result[0]
        if isinstance(first, list):
            extracted: list[str] = []
            for item in first:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    extracted.append(str(item[1]))
            return "\n".join(extracted)
    return ""


def extract_pdf_pages_from_bytes(
    content: bytes,
    *,
    source: str,
    enable_ocr: bool = False,
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = normalize_whitespace(page.get_text("text"))
                if text:
                    pages.append(
                        {
                            "page": page_number,
                            "text": text,
                            "has_section": detect_section_number(text) is not None,
                        }
                    )
    except Exception as exc:
        LOGGER.warning("PDF extraction failed for %s: %s", source, exc)
        return []

    if pages or not enable_ocr:
        return pages

    try:
        import numpy as np
        from rapidocr import RapidOCR
    except ImportError:
        LOGGER.warning("OCR requested but rapidocr/numpy is not installed: %s", source)
        return []

    LOGGER.info("No embedded text found; running OCR: %s", source)
    engine = RapidOCR()
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                channels = pixmap.n
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height,
                    pixmap.width,
                    channels,
                )
                text = normalize_whitespace(_rapidocr_result_text(engine(image)))
                if text:
                    pages.append(
                        {
                            "page": page_number,
                            "text": text,
                            "has_section": detect_section_number(text) is not None,
                            "ocr": True,
                        }
                    )
                if page_number % 25 == 0 or page_number == len(document):
                    LOGGER.info("OCR progress %s: page %s/%s", source, page_number, len(document))
    except Exception as exc:
        LOGGER.exception("OCR failed for %s: %s", source, exc)
    return pages


def extract_text_from_pdf(path: str | Path, *, enable_ocr: bool = False) -> list[dict[str, Any]]:
    pdf_path = Path(path)
    try:
        content = pdf_path.read_bytes()
    except OSError as exc:
        LOGGER.warning("Cannot read PDF %s: %s", pdf_path, exc)
        return []
    return extract_pdf_pages_from_bytes(content, source=str(pdf_path), enable_ocr=enable_ocr)


def extract_document_from_response(
    response: requests.Response,
    *,
    enable_pdf_ocr: bool = False,
) -> FetchedDocument | None:
    content_type = content_type_without_parameters(response.headers.get("Content-Type", ""))
    suffix = PurePosixPath(urlparse(response.url).path).suffix.casefold()
    fetched_at = utc_now_iso()

    if content_type == "application/pdf" or suffix == ".pdf":
        pages = extract_pdf_pages_from_bytes(
            response.content,
            source=response.url,
            enable_ocr=enable_pdf_ocr,
        )
        title = suffix and PurePosixPath(urlparse(response.url).path).name or response.url
        return FetchedDocument(response.url, "application/pdf", title, pages, fetched_at)

    if "wordprocessingml.document" in content_type or suffix == ".docx":
        text = _extract_docx_text(response.content)
        pages = [{"page": None, "text": text}] if text else []
        title = PurePosixPath(urlparse(response.url).path).name or response.url
        return FetchedDocument(response.url, content_type or "application/docx", title, pages, fetched_at)

    response.encoding = response.encoding or response.apparent_encoding or "utf-8"
    text, title = _clean_html_text(response.text)
    pages = [{"page": None, "text": text}] if text else []
    return FetchedDocument(response.url, content_type or "text/html", title, pages, fetched_at)


def extract_text_from_response(response: requests.Response) -> str:
    """Backward-compatible helper returning only combined text."""
    document = extract_document_from_response(response)
    return document.full_text if document else ""


def extract_text_from_url(
    url: str,
    session: requests.Session | None = None,
    *,
    enable_pdf_ocr: bool = False,
) -> str:
    owns_session = session is None
    client = session or create_scraping_session()
    try:
        response = fetch_response(client, url)
        if response is None:
            return ""
        document = extract_document_from_response(response, enable_pdf_ocr=enable_pdf_ocr)
        return document.full_text if document else ""
    finally:
        if owns_session:
            client.close()


# ---------------------------------------------------------------------------
# Source-list parsing
# ---------------------------------------------------------------------------
def _entry_name(entry: Mapping[str, Any], fallback: str) -> str:
    return str(entry.get("name") or entry.get("title") or entry.get("label") or fallback).strip()


def extract_urls_from_json(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract official source definitions and direct legal URLs.

    The function handles the known lawyer_links.json sections and also performs
    a conservative recursive pass for objects containing a ``url`` field.
    """
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(entry: Mapping[str, Any] | None, *, name: str | None = None, url: str | None = None) -> None:
        raw = dict(entry or {})
        candidate = canonicalize_url(str(url or raw.get("url") or raw.get("href") or raw.get("link") or ""))
        if not candidate or candidate in seen or should_skip_url(candidate):
            return
        seen.add(candidate)
        raw["url"] = candidate
        raw["name"] = str(name or _entry_name(raw, urlparse(candidate).netloc))
        raw.setdefault("jurisdiction", "Unknown")
        raw.setdefault("source_type", "official_web")
        raw.setdefault("discover_links", False)
        results.append(raw)

    for source in data.get("official_sources", []) or []:
        if isinstance(source, Mapping):
            add(source)

    for law in data.get("major_laws", []) or []:
        if isinstance(law, Mapping):
            add(law)

    for section_name in (
        "other_important_law_portals",
        "navigation_links",
        "category_wise_links",
    ):
        section = data.get(section_name, {}) or {}
        if isinstance(section, Mapping):
            for key, value in section.items():
                prefix = {
                    "navigation_links": "Navigation",
                    "category_wise_links": "Category",
                }.get(section_name)
                display_name = f"{prefix}: {key}" if prefix else str(key).replace("_", " ").title()
                if isinstance(value, str):
                    add(None, name=display_name, url=value)
                elif isinstance(value, Mapping):
                    add(value, name=display_name)

    def walk(node: Any) -> Iterator[Mapping[str, Any]]:
        if isinstance(node, Mapping):
            if any(key in node for key in ("url", "href", "link")):
                yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    for candidate in walk(data):
        add(candidate)

    return results


# ---------------------------------------------------------------------------
# Legal-document filtering and crawling
# ---------------------------------------------------------------------------
def is_legal_document(text: str, url: str, title: str = "") -> bool:
    lowered_url = url.casefold()
    lowered_title = title.casefold()
    if any(term in lowered_url for term in NON_LAW_URL_TERMS):
        return False

    normalized = f" {normalized_text_for_hash(text)} "
    marker_patterns = (
        r"\bsection\s+\d+",
        r"\barticle\s+\d+",
        r"\brule\s+\d+",
        r"\bsubsection\b",
        r"\bshort title\b",
        r"\bcommencement\b",
        r"\bdefinitions?\b",
        r"\bbe it enacted\b",
        r"\bordinance\b",
        r"\bregulations?\b",
        r"\bgazette\b",
        r"\bs\.?r\.?o\.?\s*\d+",
    )
    marker_count = sum(len(re.findall(pattern, normalized, re.IGNORECASE)) for pattern in marker_patterns)
    legal_name_signal = any(term in f"{lowered_url} {lowered_title}" for term in LAW_LINK_TERMS)

    word_count = len(normalized.split())
    if word_count < 30:
        return False
    return marker_count >= 3 or (legal_name_signal and marker_count >= 1 and word_count >= 80)


def _is_crawlable_link(
    url: str,
    allowed_hosts: set[str],
    anchor_text: str = "",
    *,
    crawl_all_internal: bool = False,
) -> bool:
    canonical = canonicalize_url(url)
    if not canonical or should_skip_url(canonical):
        return False

    parsed = urlparse(canonical)
    if host_key(parsed.netloc) not in allowed_hosts:
        return False

    if crawl_all_internal:
        return True

    suffix = PurePosixPath(parsed.path).suffix.casefold()
    if suffix in {".pdf", ".docx"}:
        return True

    haystack = f"{canonical} {anchor_text}".casefold()
    return any(term in haystack for term in LAW_LINK_TERMS)


def crawl_official_source(
    source: Mapping[str, Any],
    *,
    delay_seconds: float | None = None,
    max_pages_override: int | None = None,
    enable_pdf_ocr: bool = False,
) -> tuple[list[FetchedDocument], SourceStats]:
    start_url = canonicalize_url(str(source["url"]))
    source_name = str(source.get("name", start_url))
    jurisdiction = str(source.get("jurisdiction", "Unknown"))
    stats = SourceStats(source_name, start_url, jurisdiction)

    configured_limit = safe_int(source.get("max_pages"), DEFAULT_MAX_SOURCE_PAGES)
    max_pages = min(configured_limit, max_pages_override) if max_pages_override else configured_limit
    actual_delay = max(0.0, float(delay_seconds if delay_seconds is not None else source.get("delay_seconds", DEFAULT_DELAY_SECONDS)))

    allowed_hosts = {host_key(start_url)}
    for allowed in source.get("allowed_hosts", []) or []:
        allowed_hosts.add(host_key(str(allowed)))

    fallback_urls = [start_url]
    fallback_urls.extend(str(item) for item in source.get("fallback_urls", []) or [])
    if "pakistancode" in host_key(start_url):
        fallback_urls.extend(PAKISTAN_CODE_FALLBACK_URLS)

    queue: deque[str] = deque([start_url])
    queued = {start_url}
    visited: set[str] = set()
    final_urls_seen: set[str] = set()
    documents: list[FetchedDocument] = []
    session = create_scraping_session()

    try:
        while queue and len(visited) < max_pages:
            requested_url = queue.popleft()
            if requested_url in visited:
                continue
            visited.add(requested_url)
            stats.visited_pages = len(visited)

            if requested_url == start_url and len(fallback_urls) > 1:
                response = fetch_with_fallbacks(session, fallback_urls)
            else:
                response = fetch_response(session, requested_url)

            if response is None:
                stats.failed_urls += 1
                if actual_delay:
                    time.sleep(actual_delay)
                continue

            final_url = canonicalize_url(response.url)
            if final_url in final_urls_seen:
                if actual_delay:
                    time.sleep(actual_delay)
                continue
            final_urls_seen.add(final_url)

            document = extract_document_from_response(response, enable_pdf_ocr=enable_pdf_ocr)
            if document and len(document.full_text) >= MIN_DOCUMENT_CHARACTERS:
                documents.append(document)
                stats.extracted_documents += 1

            content_type = content_type_without_parameters(response.headers.get("Content-Type", ""))
            should_discover = bool(source.get("discover_links", False))
            if should_discover and content_type in {"text/html", "application/xhtml+xml", ""}:
                response.encoding = response.encoding or response.apparent_encoding or "utf-8"
                soup = BeautifulSoup(response.text, "html.parser")
                crawl_all = bool(source.get("crawl_all_internal", False))
                for link in soup.find_all("a", href=True):
                    absolute = canonicalize_url(urljoin(response.url, str(link["href"])))
                    if not absolute or absolute in queued or absolute in visited:
                        continue
                    anchor = link.get_text(" ", strip=True)
                    if _is_crawlable_link(
                        absolute,
                        allowed_hosts,
                        anchor,
                        crawl_all_internal=crawl_all,
                    ):
                        queued.add(absolute)
                        queue.append(absolute)
                    else:
                        stats.skipped_urls += 1

            if actual_delay:
                time.sleep(actual_delay)
    except KeyboardInterrupt:
        LOGGER.warning("Crawl interrupted by user: %s", source_name)
        raise
    except Exception as exc:
        stats.errors.append(str(exc))
        LOGGER.exception("Unexpected crawl error for %s: %s", source_name, exc)
    finally:
        session.close()
        stats.finish()

    LOGGER.info(
        "Crawl complete | source=%s visited=%s documents=%s failed=%s",
        source_name,
        stats.visited_pages,
        stats.extracted_documents,
        stats.failed_urls,
    )
    return documents, stats


# ---------------------------------------------------------------------------
# Chunking and metadata
# ---------------------------------------------------------------------------
def detect_section_number(text: str) -> str | None:
    for pattern in SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _find_chunk_boundary(text: str, start: int, tentative_end: int) -> int:
    if tentative_end >= len(text):
        return len(text)

    minimum_boundary = start + int(int(CHUNK_SIZE) * 0.65)
    window = text[start:tentative_end]
    candidates = (
        window.rfind("\n\n"),
        window.rfind("\n"),
        window.rfind(". "),
        window.rfind("; "),
    )
    best = max(candidates)
    absolute = start + best + (2 if best >= 0 and window[best:best + 2] in {". ", "; ", "\n\n"} else 1)
    return absolute if best >= 0 and absolute >= minimum_boundary else tentative_end


def chunk_text(text: str, source_file: str = "") -> list[dict[str, Any]]:
    validate_chunk_settings()
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    sequence = 0
    chunk_size = int(CHUNK_SIZE)
    overlap = int(CHUNK_OVERLAP)

    while start < len(cleaned):
        tentative_end = min(len(cleaned), start + chunk_size)
        end = _find_chunk_boundary(cleaned, start, tentative_end)
        if end <= start:
            end = tentative_end

        chunk = cleaned[start:end].strip()
        if chunk:
            section_number = detect_section_number(chunk)
            normalized_hash = sha256_text(normalized_text_for_hash(chunk))
            chunks.append(
                {
                    "text": chunk,
                    "section": section_number,
                    "has_section": section_number is not None,
                    "chunk_index": sequence,
                    "character_start": start,
                    "character_end": end,
                    "content_hash": normalized_hash,
                    "source_file": source_file,
                }
            )
            sequence += 1

        if end >= len(cleaned):
            break
        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks


def make_chunk_id(source: str, page: int | None, index: int, content_hash: str) -> str:
    basis = f"{source}|{page}|{index}|{content_hash}"
    return hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()[:32]


def chunks_from_pages(
    pages: Sequence[Mapping[str, Any]],
    *,
    source: str,
    source_name: str,
    jurisdiction: str,
    source_type: str,
    official: bool,
    title: str | None = None,
    content_type: str | None = None,
    fetched_at: str | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    global_index = 0

    for page in pages:
        page_number = page.get("page")
        for local_chunk in chunk_text(str(page.get("text", "")), source):
            content_hash = str(local_chunk["content_hash"])
            output.append(
                {
                    "id": make_chunk_id(source, page_number, global_index, content_hash),
                    "source": source,
                    "source_name": source_name,
                    "document_title": title or source_name,
                    "jurisdiction": jurisdiction,
                    "source_type": source_type,
                    "official": official,
                    "page": page_number,
                    "content_type": content_type,
                    "fetched_at": fetched_at,
                    "text": local_chunk["text"],
                    "section": local_chunk["section"],
                    "has_section": local_chunk["has_section"],
                    "chunk_index": global_index,
                    "content_hash": content_hash,
                    "ocr": bool(page.get("ocr", False)),
                }
            )
            global_index += 1
    return output


def filter_legal_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    for chunk in chunks:
        key = (chunk.get("source_name"), chunk.get("source"))
        documents.setdefault(key, []).append(chunk)

    filtered: list[dict[str, Any]] = []
    for (_, url), document_chunks in documents.items():
        first = document_chunks[0]
        if not first.get("official"):
            filtered.extend(document_chunks)
            continue

        text = " ".join(str(chunk.get("text", "")) for chunk in document_chunks)
        if is_legal_document(text, str(url or ""), str(first.get("document_title", ""))):
            filtered.extend(document_chunks)
    return filtered


def deduplicate_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by content while retaining alternate source provenance."""
    unique: dict[str, dict[str, Any]] = {}

    for raw_chunk in chunks:
        chunk = dict(raw_chunk)
        text = normalize_whitespace(str(chunk.get("text", "")))
        if not text:
            continue
        content_hash = str(chunk.get("content_hash") or sha256_text(normalized_text_for_hash(text)))
        chunk["text"] = text
        chunk["content_hash"] = content_hash

        existing = unique.get(content_hash)
        if existing is None:
            chunk.setdefault("alternate_sources", [])
            unique[content_hash] = chunk
            continue

        alternate = {
            "source": chunk.get("source"),
            "source_name": chunk.get("source_name"),
            "page": chunk.get("page"),
            "official": chunk.get("official"),
        }
        if alternate not in existing.setdefault("alternate_sources", []):
            existing["alternate_sources"].append(alternate)

        # Prefer an official source as the primary record.
        if chunk.get("official") and not existing.get("official"):
            chunk["alternate_sources"] = existing.get("alternate_sources", []) + [
                {
                    "source": existing.get("source"),
                    "source_name": existing.get("source_name"),
                    "page": existing.get("page"),
                    "official": existing.get("official"),
                }
            ]
            unique[content_hash] = chunk

    return list(unique.values())


# ---------------------------------------------------------------------------
# FAISS index creation
# ---------------------------------------------------------------------------
def create_faiss_index(chunks: Sequence[Mapping[str, Any]]):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    texts = [str(chunk["text"]) for chunk in chunks if str(chunk.get("text", "")).strip()]
    if not texts:
        raise ValueError("Cannot create a FAISS index from an empty chunk list.")

    LOGGER.info("Initializing embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    batch_size = safe_int(os.getenv("EMBEDDING_BATCH_SIZE", 32), 32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")

    metric = os.getenv("FAISS_METRIC", "l2").strip().casefold()
    if metric == "cosine":
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        metric = "l2"
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    manifest = {
        "created_at": utc_now_iso(),
        "embedding_model": EMBEDDING_MODEL,
        "metric": metric,
        "dimension": int(embeddings.shape[1]),
        "chunk_count": len(texts),
    }
    return index, model, manifest


def save_faiss_artifacts(chunks: Sequence[dict[str, Any]]) -> None:
    import faiss

    EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
    index, _model, manifest = create_faiss_index(chunks)

    final_index = EMBEDDINGS_PATH / "faiss.index"
    temp_index = EMBEDDINGS_PATH / ".faiss.index.tmp"
    faiss.write_index(index, str(temp_index))
    os.replace(temp_index, final_index)

    final_chunks = EMBEDDINGS_PATH / "chunks.pkl"
    temp_chunks = EMBEDDINGS_PATH / ".chunks.pkl.tmp"
    with temp_chunks.open("wb") as handle:
        pickle.dump(list(chunks), handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_chunks, final_chunks)

    atomic_write_json(EMBEDDINGS_PATH / "index_manifest.json", manifest)
    LOGGER.info("Saved FAISS index and metadata for %s chunks.", len(chunks))


# ---------------------------------------------------------------------------
# Local PDF processing
# ---------------------------------------------------------------------------
def process_local_pdfs(*, enable_ocr: bool = False) -> list[dict[str, Any]]:
    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(path for path in INPUT_PATH.iterdir() if path.suffix.casefold() == ".pdf")
    LOGGER.info("Found %s local PDF files in %s", len(pdf_files), INPUT_PATH)

    chunks: list[dict[str, Any]] = []
    for index, path in enumerate(pdf_files, start=1):
        LOGGER.info("Processing local PDF %s/%s: %s", index, len(pdf_files), path.name)
        try:
            pages = extract_text_from_pdf(path, enable_ocr=enable_ocr)
            if not pages:
                LOGGER.warning("No readable text extracted; skipping: %s", path.name)
                continue
            document_chunks = chunks_from_pages(
                pages,
                source=path.name,
                source_name=path.stem,
                jurisdiction="Pakistan",
                source_type="local_pdf",
                official=False,
                title=path.stem,
                content_type="application/pdf",
                fetched_at=utc_now_iso(),
            )
            chunks.extend(document_chunks)
            LOGGER.info("Created %s chunks from %s", len(document_chunks), path.name)
        except Exception as exc:
            LOGGER.exception("Local PDF processing failed for %s: %s", path, exc)
    return chunks


def process_missing_pdfs() -> None:
    metadata_path = EMBEDDINGS_PATH / "chunks.pkl"
    embedded_sources: set[str] = set()

    if metadata_path.exists():
        try:
            with metadata_path.open("rb") as handle:
                embedded_sources = {
                    str(chunk.get("source"))
                    for chunk in pickle.load(handle)
                    if isinstance(chunk, Mapping) and not chunk.get("official")
                }
        except Exception as exc:
            LOGGER.warning("Could not read active chunk metadata: %s", exc)

    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    missing = sorted(
        path
        for path in INPUT_PATH.iterdir()
        if path.suffix.casefold() == ".pdf" and path.name not in embedded_sources
    )
    LOGGER.info("Found %s PDFs missing from active metadata.", len(missing))

    updates: list[dict[str, Any]] = []
    for index, path in enumerate(missing, start=1):
        LOGGER.info("Extracting missing PDF %s/%s: %s", index, len(missing), path.name)
        pages = extract_text_from_pdf(path, enable_ocr=True)
        updates.extend(
            chunks_from_pages(
                pages,
                source=path.name,
                source_name=path.stem,
                jurisdiction="Pakistan",
                source_type="local_pdf",
                official=False,
                title=path.stem,
                content_type="application/pdf",
                fetched_at=utc_now_iso(),
            )
        )

    updates = deduplicate_chunks(updates)
    atomic_write_json(PDF_UPDATES_PATH, updates)
    LOGGER.info("Saved %s missing-PDF chunks to %s", len(updates), PDF_UPDATES_PATH)


# ---------------------------------------------------------------------------
# Web-source processing and resumable checkpoints
# ---------------------------------------------------------------------------
def load_crawl_state() -> dict[str, Any]:
    state = load_json_file(CRAWL_STATE_PATH, default={})
    if not isinstance(state, dict):
        return {}
    state.setdefault("completed_sources", [])
    state.setdefault("updated_at", utc_now_iso())
    return state


def save_crawl_state(state: Mapping[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(CRAWL_STATE_PATH, payload)


def process_web_sources(
    entries: Sequence[Mapping[str, Any]],
    *,
    existing_chunks: Sequence[dict[str, Any]] | None = None,
    output_path: Path,
    resume: bool = False,
    source_offset: int = 0,
    source_limit: int | None = None,
    max_pages_override: int | None = None,
    delay_seconds: float | None = None,
    enable_pdf_ocr: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = list(entries)
    if source_offset:
        selected = selected[source_offset:]
    if source_limit is not None:
        selected = selected[: max(0, source_limit)]

    all_chunks = list(existing_chunks or [])
    state = load_crawl_state() if resume else {"completed_sources": []}
    completed = set(str(item) for item in state.get("completed_sources", []))
    reports: list[dict[str, Any]] = []

    LOGGER.info("Selected %s web sources for crawling.", len(selected))
    for source_number, entry in enumerate(selected, start=1):
        url = canonicalize_url(str(entry.get("url", "")))
        name = str(entry.get("name", url))
        jurisdiction = str(entry.get("jurisdiction", "Unknown"))

        if resume and url in completed:
            LOGGER.info("Skipping completed source %s/%s: %s", source_number, len(selected), name)
            continue

        LOGGER.info(
            "Scraping source %s/%s | %s [%s] -> %s",
            source_number,
            len(selected),
            name,
            jurisdiction,
            url,
        )

        try:
            if bool(entry.get("discover_links", False)):
                documents, stats = crawl_official_source(
                    entry,
                    delay_seconds=delay_seconds,
                    max_pages_override=max_pages_override,
                    enable_pdf_ocr=enable_pdf_ocr,
                )
            else:
                session = create_scraping_session()
                stats = SourceStats(name, url, jurisdiction)
                try:
                    response = fetch_response(session, url)
                    documents = []
                    stats.visited_pages = 1
                    if response is not None:
                        document = extract_document_from_response(response, enable_pdf_ocr=enable_pdf_ocr)
                        if document:
                            documents.append(document)
                            stats.extracted_documents = 1
                    else:
                        stats.failed_urls = 1
                finally:
                    session.close()
                    stats.finish()

            source_chunks: list[dict[str, Any]] = []
            for document in documents:
                if len(document.full_text) < MIN_DOCUMENT_CHARACTERS:
                    continue
                if not is_legal_document(document.full_text, document.url, document.title):
                    LOGGER.debug("Rejected non-legal document: %s", document.url)
                    continue

                stats.accepted_documents += 1
                source_chunks.extend(
                    chunks_from_pages(
                        document.pages,
                        source=document.url,
                        source_name=name,
                        jurisdiction=jurisdiction,
                        source_type=str(entry.get("source_type", "official_web")),
                        official=True,
                        title=document.title,
                        content_type=document.content_type,
                        fetched_at=document.fetched_at,
                    )
                )

            stats.created_chunks = len(source_chunks)
            reports.append(stats.to_dict())
            all_chunks.extend(source_chunks)
            all_chunks = deduplicate_chunks(filter_legal_chunks(all_chunks))

            # Save after every source so a long crawl can be resumed safely.
            atomic_write_json(output_path, all_chunks)
            completed.add(url)
            state["completed_sources"] = sorted(completed)
            save_crawl_state(state)
            LOGGER.info("Created %s chunks from %s", len(source_chunks), name)
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted. Existing checkpoint was preserved.")
            raise
        except Exception as exc:
            LOGGER.exception("Source failed: %s -> %s", name, exc)
            reports.append(
                {
                    "name": name,
                    "start_url": url,
                    "jurisdiction": jurisdiction,
                    "errors": [str(exc)],
                    "finished_at": utc_now_iso(),
                }
            )

    return deduplicate_chunks(filter_legal_chunks(all_chunks)), reports


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def process_pdfs_and_links(
    official_only: bool = False,
    skip_embeddings: bool = False,
    append_official: bool = False,
    source_offset: int = 0,
    *,
    links_only: bool = False,
    local_pdfs_only: bool = False,
    resume: bool = False,
    source_limit: int | None = None,
    max_pages: int | None = None,
    delay_seconds: float | None = None,
    ocr_local_pdfs: bool = False,
    ocr_remote_pdfs: bool = False,
) -> list[dict[str, Any]]:
    """Build the knowledge base while preserving original API compatibility."""
    validate_chunk_settings()
    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)

    if links_only and local_pdfs_only:
        raise ValueError("--links-only and --local-pdfs-only cannot be used together.")

    # Preserve legacy behavior: --official-only means web links without local PDFs.
    links_mode = bool(links_only or official_only)
    process_local = not links_mode
    process_links = not local_pdfs_only
    output_path = OFFICIAL_OUTPUT_PATH if links_mode else OUTPUT_PATH

    all_chunks: list[dict[str, Any]] = []
    should_load_existing = append_official or resume
    if should_load_existing and output_path.exists():
        existing = load_json_file(output_path, default=[])
        if isinstance(existing, list):
            all_chunks = [dict(item) for item in existing if isinstance(item, Mapping)]
            LOGGER.info("Loaded %s existing chunks from %s", len(all_chunks), output_path)

    if process_local:
        local_chunks = process_local_pdfs(enable_ocr=ocr_local_pdfs)
        all_chunks.extend(local_chunks)

    crawl_reports: list[dict[str, Any]] = []
    if process_links:
        if not LINKS_PATH.exists():
            raise FileNotFoundError(f"Links JSON not found: {LINKS_PATH}")

        LOGGER.info("Loading links from %s", LINKS_PATH)
        links_data = load_json_file(LINKS_PATH)
        if not isinstance(links_data, Mapping):
            raise ValueError(f"Expected a JSON object at the root of {LINKS_PATH}")

        entries = extract_urls_from_json(links_data)
        LOGGER.info("Found %s unique URLs in lawyer_links.json", len(entries))
        all_chunks, crawl_reports = process_web_sources(
            entries,
            existing_chunks=all_chunks,
            output_path=output_path,
            resume=resume,
            source_offset=source_offset,
            source_limit=source_limit,
            max_pages_override=max_pages,
            delay_seconds=delay_seconds,
            enable_pdf_ocr=ocr_remote_pdfs,
        )

    all_chunks = deduplicate_chunks(filter_legal_chunks(all_chunks))
    if not all_chunks:
        LOGGER.warning("No valid text was extracted from the selected inputs.")
        atomic_write_json(output_path, [])
        return []

    atomic_write_json(output_path, all_chunks)
    LOGGER.info("Saved %s valid chunks to %s", len(all_chunks), output_path)

    report = {
        "created_at": utc_now_iso(),
        "mode": "links_only" if links_mode else "local_pdfs_only" if local_pdfs_only else "combined",
        "output_path": str(output_path),
        "total_chunks": len(all_chunks),
        "sources": crawl_reports,
    }
    atomic_write_json(CRAWL_REPORT_PATH, report)

    if skip_embeddings:
        LOGGER.info("Skipped FAISS rebuild as requested.")
    else:
        save_faiss_artifacts(all_chunks)

    return all_chunks


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Pakistan legal knowledge base from local PDFs and official links.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--links-only",
        action="store_true",
        help="Scrape lawyer_links.json only; do not process local PDFs.",
    )
    mode.add_argument(
        "--local-pdfs-only",
        action="store_true",
        help="Process local PDFs only; do not crawl lawyer_links.json.",
    )
    mode.add_argument(
        "--official-only",
        action="store_true",
        help="Legacy alias for --links-only.",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Save chunk JSON without rebuilding the FAISS index.",
    )
    parser.add_argument(
        "--append-official",
        action="store_true",
        help="Load and merge existing output before crawling.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a prior crawl and skip completed source URLs.",
    )
    parser.add_argument(
        "--reset-crawl-state",
        action="store_true",
        help="Delete the crawl-state checkpoint before starting.",
    )
    parser.add_argument(
        "--source-offset",
        type=int,
        default=0,
        help="Skip the first N configured web sources.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Process at most N web sources after applying the offset.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Override the maximum pages crawled per source.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay in seconds between requests.",
    )
    parser.add_argument(
        "--ocr-local-pdfs",
        action="store_true",
        help="Use RapidOCR when a local PDF contains no extractable text.",
    )
    parser.add_argument(
        "--ocr-remote-pdfs",
        action="store_true",
        help="Use RapidOCR when a downloaded PDF contains no extractable text.",
    )
    parser.add_argument(
        "--missing-pdfs-only",
        action="store_true",
        help="Extract only local PDFs absent from active FAISS metadata.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        if args.reset_crawl_state and CRAWL_STATE_PATH.exists():
            CRAWL_STATE_PATH.unlink()
            LOGGER.info("Deleted crawl state: %s", CRAWL_STATE_PATH)

        if args.missing_pdfs_only:
            process_missing_pdfs()
            return 0

        process_pdfs_and_links(
            official_only=args.official_only,
            skip_embeddings=args.skip_embeddings,
            append_official=args.append_official,
            source_offset=max(0, args.source_offset),
            links_only=args.links_only,
            local_pdfs_only=args.local_pdfs_only,
            resume=args.resume,
            source_limit=args.source_limit,
            max_pages=args.max_pages,
            delay_seconds=args.delay,
            ocr_local_pdfs=args.ocr_local_pdfs,
            ocr_remote_pdfs=args.ocr_remote_pdfs,
        )
        LOGGER.info("Processing complete.")
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Processing stopped by user. Checkpoints were preserved.")
        return 130
    except Exception as exc:
        LOGGER.exception("Processing failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())