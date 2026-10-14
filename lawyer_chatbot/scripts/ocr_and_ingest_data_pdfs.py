"""OCR Ingestion for the 10 remaining scanned PDFs in data/pdfs.

Extracts text via RapidOCR from:
- 1. Virtual Assets Ordinance, 2025.pdf
- Acid Control and Acid Crime Prevention Act, 2011.pdf
- The-Khyber-Pakhtunkhwa-Rules-of-Business-20151-Approved.pdf
- The-Balochistan-Rules-of-Business-2012.-Amendment-2020Service-Rules-2020.pdf
- 1782293069_736.pdf
- 1787218119_962.pdf
- 1787218209_147.pdf
- 1787311828_865.pdf
- 1787311922_693.pdf
- Fundamental Law of Pakistan-by A. K. Brohi.pdf

Chunks, embeds with all-MiniLM-L6-v2, and merges into the active FAISS index.
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
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PDFS_DIR = BASE_DIR / "data" / "pdfs"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.pkl"
MANIFEST_PATH = EMBEDDINGS_DIR / "index_manifest.json"
OCR_OUTPUT_JSON = EMBEDDINGS_DIR / "ocr_data_pdfs_chunks.json"

CHUNK_SIZE = 850
CHUNK_OVERLAP = 180
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("ocr_data_pdfs")

# The 10 scanned files identified in data/pdfs
TARGET_FILES = [
    "Acid Control and Acid Crime Prevention Act, 2011.pdf",
    "1782293069_736.pdf",
    "1787218119_962.pdf",
    "1787218209_147.pdf",
    "1787311828_865.pdf",
    "1787311922_693.pdf",
    "The-Balochistan-Rules-of-Business-2012.-Amendment-2020Service-Rules-2020.pdf",
    "1. Virtual Assets Ordinance, 2025.pdf",
    "The-Khyber-Pakhtunkhwa-Rules-of-Business-20151-Approved.pdf",
    "Fundamental Law of Pakistan-by A. K. Brohi.pdf",
]

SECTION_PATTERNS = [
    re.compile(r"\bSection\s+(\d+[A-Za-z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bSec\.?\s+(\d+[A-Za-z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bArticle\s+(\d+[A-Za-z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bArt\.?\s+(\d+[A-Za-z]?(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bOrder\s+([IVXLCDM\d]+)\s+Rule\s+(\d+[A-Za-z]?)\b", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(\d+[A-Za-z]?)\.\s+[A-Z]", re.MULTILINE),
]


def clean_legal_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", raw_text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?i)\bpage\s+\d+\s+(?:of\s+\d+)?\b", "", text)
    return text.strip()


def detect_section(text: str) -> str | None:
    for pattern in SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return f"Order {groups[0]} Rule {groups[1]}"
            return f"Section {groups[0]}" if "article" not in pattern.pattern.lower() else f"Article {groups[0]}"
    return None


def clean_title(filename: str) -> str:
    stem = Path(filename).stem
    s = re.sub(r"^\d+[\.\s_\-]+", "", stem)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title() if s else stem


def chunk_ocr_page(
    page_text: str,
    doc_title: str,
    jurisdiction: str,
    page_num: int,
    source_filename: str,
) -> list[dict[str, Any]]:
    cleaned = clean_legal_text(page_text)
    if not cleaned or len(cleaned) < 35:
        return []

    chunks = []
    start = 0
    seq = 0

    while start < len(cleaned):
        tentative_end = min(len(cleaned), start + CHUNK_SIZE)
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
        if len(raw_chunk_text) >= 35:
            section_found = detect_section(raw_chunk_text)
            header = f"[Statute/Book: {doc_title}"
            if section_found:
                header += f" | {section_found}"
            header += f" | Page: {page_num} (OCR)]"

            chunk_with_header = f"{header}\n{raw_chunk_text}"
            content_hash = hashlib.sha256(raw_chunk_text.encode("utf-8")).hexdigest()
            chunk_id = hashlib.md5(f"ocr_d_{source_filename}_{page_num}_{seq}_{content_hash[:16]}".encode("utf-8")).hexdigest()

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
                    "ocr": True,
                }
            )
            seq += 1

        if end >= len(cleaned):
            break
        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end

    return chunks


def main() -> None:
    LOGGER.info("Starting OCR extraction for %s remaining scanned PDFs in data/pdfs...", len(TARGET_FILES))
    ocr_engine = RapidOCR()

    LOGGER.info("Loading active knowledge base from %s...", CHUNKS_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        existing_chunks: list[dict[str, Any]] = pickle.load(f)
    LOGGER.info("Active knowledge base currently has %s chunks.", len(existing_chunks))

    existing_hashes = {
        str(c.get("content_hash") or hashlib.sha256(c.get("text", "")[:250].encode("utf-8")).hexdigest())
        for c in existing_chunks
        if isinstance(c, dict)
    }

    all_ocr_chunks: list[dict[str, Any]] = []
    total_pages = 0
    t0_ocr = time.time()

    for idx, fname in enumerate(TARGET_FILES, 1):
        pdf_path = PDFS_DIR / fname
        if not pdf_path.exists():
            LOGGER.warning("[%s/%s] Not found: %s", idx, len(TARGET_FILES), fname)
            continue

        doc_title = clean_title(fname)
        jurisdiction = "Khyber Pakhtunkhwa" if "khyber" in fname.lower() or "kp" in fname.lower() else "Balochistan" if "balochistan" in fname.lower() else "Federal / Pakistan"

        LOGGER.info("[%s/%s] Starting OCR on %s...", idx, len(TARGET_FILES), fname)
        file_chunks = []
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            total_pages += num_pages

            for page_idx in range(num_pages):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                res, _ = ocr_engine(img)

                page_text = "\n".join(item[1] for item in res) if res else ""
                if len(page_text.strip()) > 30:
                    chunks = chunk_ocr_page(
                        page_text=page_text,
                        doc_title=doc_title,
                        jurisdiction=jurisdiction,
                        page_num=page_idx + 1,
                        source_filename=fname,
                    )
                    file_chunks.extend(chunks)

                if (page_idx + 1) % 15 == 0 or (page_idx + 1) == num_pages:
                    LOGGER.info(
                        "  - %s: Page %s/%s OCR complete (%s chunks generated)",
                        fname[:30],
                        page_idx + 1,
                        num_pages,
                        len(file_chunks),
                    )

            doc.close()
            all_ocr_chunks.extend(file_chunks)
            LOGGER.info("[%s/%s] Finished %s (%s pages -> %s chunks)", idx, len(TARGET_FILES), fname, num_pages, len(file_chunks))

        except Exception as exc:
            LOGGER.error("[%s/%s] Error OCRing %s: %s", idx, len(TARGET_FILES), fname, exc)

    ocr_dur = time.time() - t0_ocr
    LOGGER.info("All OCR completed in %.1fs across %s pages. Total candidate chunks: %s", ocr_dur, total_pages, len(all_ocr_chunks))

    # Deduplicate against active knowledge base
    unique_ocr_chunks = []
    seen = set()
    for c in all_ocr_chunks:
        h = c["content_hash"]
        if h not in existing_hashes and h not in seen:
            seen.add(h)
            unique_ocr_chunks.append(c)

    LOGGER.info("Deduplication complete: %s brand-new distinct chunks to merge.", len(unique_ocr_chunks))

    if not unique_ocr_chunks:
        LOGGER.info("No new chunks to add.")
        return

    # Save backup JSON
    temp_json = OCR_OUTPUT_JSON.with_suffix(".tmp")
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(unique_ocr_chunks, f, ensure_ascii=False, indent=2)
    os.replace(temp_json, OCR_OUTPUT_JSON)

    # Embed chunks
    LOGGER.info("Embedding %s chunks with %s (batch_size=%s)...", len(unique_ocr_chunks), EMBEDDING_MODEL, BATCH_SIZE)
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts_to_embed = [c["text"] for c in unique_ocr_chunks]

    t0_embed = time.time()
    embeddings = model.encode(texts_to_embed, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    LOGGER.info("Embedding finished in %.1fs.", time.time() - t0_embed)

    # Append to active FAISS index
    index = faiss.read_index(str(INDEX_PATH))
    old_vector_count = index.ntotal
    index.add(embeddings)
    new_vector_count = index.ntotal

    combined_chunks = existing_chunks + unique_ocr_chunks
    LOGGER.info("Index updated from %s to %s vectors.", old_vector_count, new_vector_count)

    # Staged atomic persistence
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

    os.replace(temp_index, INDEX_PATH)
    os.replace(temp_chunks, CHUNKS_PATH)
    os.replace(temp_manifest, MANIFEST_PATH)

    LOGGER.info("=" * 80)
    LOGGER.info("ALL 10 SCANNED PDFS IN data/pdfs SUCCESSFULLY OCR-EXTRACTED & MERGED!")
    LOGGER.info(" - Total new OCR chunks: %s", len(unique_ocr_chunks))
    LOGGER.info(" - Total active knowledge base vectors now: %s", new_vector_count)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
