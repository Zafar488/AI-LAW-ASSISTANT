"""Production-Grade Ingestion and FAISS Integration for data/new_pdfs.

Processes 173 newly added Pakistani legal PDFs:
- Robust text extraction using PyMuPDF (fitz)
- Smart statutory title and jurisdiction normalization
- Section, Article, Order, and Rule detection
- Context header injection: [Statute: <Title> | Section: <Sec> | Page: <Page>]
- Boundary-aware sliding window chunking
- Deduplication against existing 17,097 passages in embeddings/chunks.pkl
- L2-normalized dense embeddings via SentenceTransformer (all-MiniLM-L6-v2)
- Incremental merge into active FAISS IndexFlatIP (cosine similarity)
- Atomic staged persistence of .faiss.index, .chunks.pkl, and index_manifest.json
- Standalone provenance artifact saved to embeddings/new_pdfs_chunks.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

NEW_PDFS_DIR = BASE_DIR / "data" / "new_pdfs"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.pkl"
MANIFEST_PATH = EMBEDDINGS_DIR / "index_manifest.json"
NEW_CHUNKS_JSON = EMBEDDINGS_DIR / "new_pdfs_chunks.json"

CHUNK_SIZE = 850
CHUNK_OVERLAP = 180
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("ingest_new_pdfs")

SECTION_PATTERNS = [
    re.compile(r"\bSection\s+(\d+[A-Za-z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bSec\.?\s+(\d+[A-Za-z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bArticle\s+(\d+[A-Za-z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bArt\.?\s+(\d+[A-Za-z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bOrder\s+([IVXLCDM\d]+)\s+Rule\s+(\d+[A-Za-z]?)\b", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(\d+[A-Za-z]?)\.\s+[A-Z]", re.MULTILINE),
]


def clean_legal_text(raw_text: str) -> str:
    """Normalize whitespace, repair hyphenated line wraps, and strip page headers."""
    if not raw_text:
        return ""
    # Repair hyphenated line wraps: e.g. "juris-\ndiction" -> "jurisdiction"
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", raw_text)
    # Normalize multiple newlines and spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove standalone page numbers or headers
    text = re.sub(r"(?i)\bpage\s+\d+\s+(?:of\s+\d+)?\b", "", text)
    return text.strip()


def detect_section(text: str) -> str | None:
    """Identify statutory section, article, or Order/Rule from text."""
    for pattern in SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return f"Order {groups[0]} Rule {groups[1]}"
            return f"Section {groups[0]}" if "article" not in pattern.pattern.lower() else f"Article {groups[0]}"
    return None


def extract_smart_title(doc: fitz.Document, filename: str) -> str:
    """Extract clean statutory title from PDF text headers, falling back to cleaned filename."""
    # First search first 2 pages for official Act/Ordinance/Rules heading
    for page_idx in range(min(2, len(doc))):
        text = doc[page_idx].get_text("text")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            if re.search(r"\b(ACT|ORDINANCE|RULES|ORDER|CODE)\b", line, re.I):
                if 10 < len(line) < 130 and not re.search(r"^(PART|REGISTERED|PUBLISHED|GOVERNMENT|EXTRA|ISLAMABAD|LAHORE|PESHAWAR|PRICE|NO\.\s*F\.)", line, re.I):
                    cleaned = re.sub(r"\s+", " ", line).strip()
                    # Strip leading numbering like '1.' or '[1]'
                    cleaned = re.sub(r"^[0-9\[\]\s\.\-]+", "", cleaned).strip()
                    if len(cleaned) > 8:
                        return cleaned

    # Fallback: clean filename
    stem = Path(filename).stem
    s = re.sub(r"^\d{4}_\d+_\s*", "", stem)
    s = re.sub(r"^\d{8,}\s*", "", s)
    s = re.sub(r"^[0-9a-f]{10,}_\d+_\s*", "", s, flags=re.I)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?i)\b(updatedupto.*|pdf|compressed|withindex.*|dated.*|act.*of.*202\d)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.title() if s else stem


def detect_jurisdiction(title: str, text: str) -> str:
    """Infer jurisdiction (Khyber Pakhtunkhwa, Punjab, Sindh, Federal, etc.)."""
    haystack = f"{title} {text[:500]}".lower()
    if "khyber pakhtunkhwa" in haystack or "kp " in haystack or "nwfp" in haystack or "north-west frontier" in haystack:
        return "Khyber Pakhtunkhwa"
    if "punjab" in haystack:
        return "Punjab"
    if "sindh" in haystack:
        return "Sindh"
    if "balochistan" in haystack:
        return "Balochistan"
    if "west pakistan" in haystack:
        return "West Pakistan (Provincial)"
    return "Federal / Pakistan"


def chunk_document_page(
    page_text: str,
    doc_title: str,
    jurisdiction: str,
    page_num: int,
    source_filename: str,
) -> list[dict[str, Any]]:
    """Chunk a single page of text with boundary preservation and metadata injection."""
    cleaned = clean_legal_text(page_text)
    if not cleaned or len(cleaned) < 40:
        return []

    chunks = []
    start = 0
    seq = 0

    while start < len(cleaned):
        tentative_end = min(len(cleaned), start + CHUNK_SIZE)

        # Look for natural sentence or paragraph boundary
        if tentative_end < len(cleaned):
            window = cleaned[start:tentative_end]
            candidates = [
                window.rfind("\n\n"),
                window.rfind(".\n"),
                window.rfind(". "),
                window.rfind(";\n"),
                window.rfind("; "),
            ]
            best = max(candidates)
            min_boundary = int(CHUNK_SIZE * 0.60)
            if best >= min_boundary:
                end = start + best + (2 if window[best:best + 2] in {". ", "; "} else 1)
            else:
                end = tentative_end
        else:
            end = len(cleaned)

        raw_chunk_text = cleaned[start:end].strip()
        if len(raw_chunk_text) >= 40:
            section_found = detect_section(raw_chunk_text)

            # Context Header: anchors vector embedding and LLM prompt
            header = f"[Statute: {doc_title}"
            if section_found:
                header += f" | {section_found}"
            header += f" | Page: {page_num}]"

            chunk_with_header = f"{header}\n{raw_chunk_text}"

            content_hash = hashlib.sha256(raw_chunk_text.encode("utf-8")).hexdigest()
            chunk_id = hashlib.md5(f"{source_filename}_{page_num}_{seq}_{content_hash[:16]}".encode("utf-8")).hexdigest()

            chunks.append(
                {
                    "id": chunk_id,
                    "source": source_filename,
                    "source_name": doc_title,
                    "document_title": doc_title,
                    "jurisdiction": jurisdiction,
                    "source_type": "local_pdf",
                    "official": True,
                    "page": page_num,
                    "section": section_found,
                    "has_section": section_found is not None,
                    "chunk_index": seq,
                    "text": chunk_with_header,
                    "raw_text": raw_chunk_text,
                    "content_hash": content_hash,
                }
            )
            seq += 1

        if end >= len(cleaned):
            break
        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end

    return chunks


def main() -> None:
    LOGGER.info("Starting production ingestion of new PDFs from: %s", NEW_PDFS_DIR)

    if not NEW_PDFS_DIR.exists():
        LOGGER.error("Directory not found: %s", NEW_PDFS_DIR)
        sys.exit(1)

    pdf_files = sorted(NEW_PDFS_DIR.glob("*.pdf"))
    LOGGER.info("Discovered %s PDF files in data/new_pdfs", len(pdf_files))

    # 1. Load active chunks to check duplicates
    if not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
        LOGGER.error("Active FAISS index or chunks.pkl missing at %s", EMBEDDINGS_DIR)
        sys.exit(1)

    LOGGER.info("Loading active knowledge base from %s...", CHUNKS_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        existing_chunks: list[dict[str, Any]] = pickle.load(f)

    LOGGER.info("Active knowledge base currently contains %s passages.", len(existing_chunks))

    # Index existing hashes for fast deduplication
    existing_hashes = {
        str(c.get("content_hash") or hashlib.sha256(c.get("text", "")[:250].encode("utf-8")).hexdigest())
        for c in existing_chunks
        if isinstance(c, dict)
    }

    # 2. Extract text and create chunks from all new PDFs
    all_new_chunks: list[dict[str, Any]] = []
    scanned_files: list[str] = []
    processed_files = 0
    start_extract = time.time()

    for idx, pdf_path in enumerate(pdf_files, 1):
        try:
            doc = fitz.open(pdf_path)
            title = extract_smart_title(doc, pdf_path.name)
            first_text = doc[0].get_text("text") if len(doc) > 0 else ""
            jurisdiction = detect_jurisdiction(title, first_text)

            doc_chunks: list[dict[str, Any]] = []
            pages_with_text = 0

            for page_idx, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                if len(page_text.strip()) > 30:
                    pages_with_text += 1
                page_chunks = chunk_document_page(
                    page_text=page_text,
                    doc_title=title,
                    jurisdiction=jurisdiction,
                    page_num=page_idx,
                    source_filename=pdf_path.name,
                )
                doc_chunks.extend(page_chunks)

            num_pages = len(doc)
            doc.close()

            if pages_with_text == 0:
                scanned_files.append(pdf_path.name)
                LOGGER.warning("[%s/%s] %s -> SCANNED (0 text pages, skipped)", idx, len(pdf_files), pdf_path.name)
            else:
                processed_files += 1
                all_new_chunks.extend(doc_chunks)
                if idx % 20 == 0 or idx == len(pdf_files):
                    LOGGER.info(
                        "[%s/%s] Processed '%s' (%s pages -> %s chunks)",
                        idx,
                        len(pdf_files),
                        title[:45],
                        num_pages,
                        len(doc_chunks),
                    )

        except Exception as exc:
            LOGGER.error("[%s/%s] Failed to extract %s: %s", idx, len(pdf_files), pdf_path.name, exc)

    extract_time = time.time() - start_extract
    LOGGER.info(
        "Text extraction finished in %.1fs: %s digital files processed, %s scanned files skipped.",
        extract_time,
        processed_files,
        len(scanned_files),
    )
    LOGGER.info("Total raw candidate chunks extracted: %s", len(all_new_chunks))

    # 3. Deduplicate new chunks among themselves and against active knowledge base
    unique_new_chunks: list[dict[str, Any]] = []
    seen_in_batch = set()

    for chunk in all_new_chunks:
        h = chunk["content_hash"]
        if h not in existing_hashes and h not in seen_in_batch:
            seen_in_batch.add(h)
            unique_new_chunks.append(chunk)

    LOGGER.info(
        "Deduplication complete: %s brand-new distinct passages to be merged into FAISS.",
        len(unique_new_chunks),
    )

    if not unique_new_chunks:
        LOGGER.info("No new unique chunks to add. FAISS index is already up to date!")
        return

    # Save standalone backup artifact of the new chunks
    LOGGER.info("Writing backup artifact: %s", NEW_CHUNKS_JSON)
    temp_json = NEW_CHUNKS_JSON.with_suffix(".tmp")
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(unique_new_chunks, f, ensure_ascii=False, indent=2)
    os.replace(temp_json, NEW_CHUNKS_JSON)

    # 4. Generate dense embeddings using SentenceTransformer
    LOGGER.info("Loading embedding model: %s...", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts_to_embed = [c["text"] for c in unique_new_chunks]
    LOGGER.info("Encoding %s passages (batch_size=%s)...", len(texts_to_embed), BATCH_SIZE)

    t0_embed = time.time()
    embeddings = model.encode(
        texts_to_embed,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    embed_dur = time.time() - t0_embed
    LOGGER.info("Encoding completed in %.1fs (%.1f chunks/sec).", embed_dur, len(texts_to_embed) / max(0.01, embed_dur))

    # 5. L2 Normalization for Cosine Similarity
    LOGGER.info("Normalizing vectors for cosine similarity (IndexFlatIP)...")
    faiss.normalize_L2(embeddings)

    # 6. Load active FAISS index and append new vectors
    LOGGER.info("Loading active FAISS index: %s...", INDEX_PATH)
    index = faiss.read_index(str(INDEX_PATH))
    old_vector_count = index.ntotal
    LOGGER.info("Active FAISS index currently has %s vectors.", old_vector_count)

    if index.d != embeddings.shape[1]:
        LOGGER.error("Dimension mismatch: index has %s, model produced %s", index.d, embeddings.shape[1])
        sys.exit(1)

    LOGGER.info("Adding %s new vectors to FAISS index...", embeddings.shape[0])
    index.add(embeddings)
    new_vector_count = index.ntotal
    LOGGER.info("Updated FAISS index now has %s vectors.", new_vector_count)

    # Combine metadata chunks
    combined_chunks = existing_chunks + unique_new_chunks
    if len(combined_chunks) != new_vector_count:
        LOGGER.error(
            "Integrity check failed: combined chunks count (%s) != index vectors count (%s)",
            len(combined_chunks),
            new_vector_count,
        )
        sys.exit(1)

    # 7. Atomic staged persistence
    LOGGER.info("Performing atomic staged save of updated knowledge-base artifacts...")
    temp_index = EMBEDDINGS_DIR / ".faiss.index.tmp"
    temp_chunks = EMBEDDINGS_DIR / ".chunks.pkl.tmp"
    temp_manifest = EMBEDDINGS_DIR / ".index_manifest.tmp"

    faiss.write_index(index, str(temp_index))
    with open(temp_chunks, "wb") as f:
        pickle.dump(combined_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "metric": "cosine",
        "dimension": int(index.d),
        "chunk_count": len(combined_chunks),
        "vector_count": new_vector_count,
    }
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Commit staged files
    os.replace(temp_index, INDEX_PATH)
    os.replace(temp_chunks, CHUNKS_PATH)
    os.replace(temp_manifest, MANIFEST_PATH)

    LOGGER.info("=" * 80)
    LOGGER.info("INGESTION & MERGE SUCCESSFULLY COMPLETED!")
    LOGGER.info(" - Previous passages in index: %s", old_vector_count)
    LOGGER.info(" - Newly added unique passages: %s", len(unique_new_chunks))
    LOGGER.info(" - Total active passages now: %s", new_vector_count)
    LOGGER.info(" - Total digital PDFs ingested: %s", processed_files)
    LOGGER.info(" - Scanned PDFs skipped: %s", len(scanned_files))
    LOGGER.info(" - Artifacts updated atomically: faiss.index, chunks.pkl, index_manifest.json")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
