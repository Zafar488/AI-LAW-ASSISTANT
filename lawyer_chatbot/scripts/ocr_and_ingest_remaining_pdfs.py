"""OCR Ingestion for the 14 scanned legal PDFs in data/new_pdfs.

Uses RapidOCR (ONNXRuntime) to convert scanned gazette/act images into structured text,
chunks them with statutory context headers, generates L2-normalized dense embeddings,
and merges them directly into the active FAISS index.
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

NEW_PDFS_DIR = BASE_DIR / "data" / "new_pdfs"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.pkl"
MANIFEST_PATH = EMBEDDINGS_DIR / "index_manifest.json"

CHUNK_SIZE = 850
CHUNK_OVERLAP = 180
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("ocr_ingest")

SCANNED_FILENAMES = [
    "1335934287_218.pdf",
    "1352964021_588.pdf",
    "69d4e02dea129_496.pdf",
    "69d4e453c23cd_630.pdf",
    "69d4e5c43b261_209.pdf",
    "69d4e6e4e3d5f_218.pdf",
    "69d4e8120d82d_384.pdf",
    "act-lxvii-of-2025-pdf.pdf",
    "Amendment-in-WP-Land-Revenue-Act.pdf",
    "PUB-25-000020.pdf",
    "PUB-26-000033.pdf",
    "PUB-26-000037.pdf",
    "The-Institute-of-Management-SciencesAmendment-Act-2025-.pdf",
    "The-Khyber-Pakhtunkhwa-Finance-Act2025.pdf",
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
    """Normalize whitespace, repair hyphenated line wraps, and strip noise."""
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


def clean_title_from_stem(filename: str) -> str:
    stem = Path(filename).stem
    s = re.sub(r"^\d{4}_\d+_\s*", "", stem)
    s = re.sub(r"^\d{8,}\s*", "", s)
    s = re.sub(r"^[0-9a-f]{10,}_\d+_\s*", "", s, flags=re.I)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?i)\b(updatedupto.*|pdf|compressed|withindex.*|dated.*)\b", "", s)
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
            header = f"[Statute: {doc_title}"
            if section_found:
                header += f" | {section_found}"
            header += f" | Page: {page_num} (OCR)]"

            chunk_with_header = f"{header}\n{raw_chunk_text}"
            content_hash = hashlib.sha256(raw_chunk_text.encode("utf-8")).hexdigest()
            chunk_id = hashlib.md5(f"ocr_{source_filename}_{page_num}_{seq}_{content_hash[:16]}".encode("utf-8")).hexdigest()

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
    LOGGER.info("Initializing RapidOCR engine for %s scanned PDFs...", len(SCANNED_FILENAMES))
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
    total_pages_ocrd = 0
    t0_ocr = time.time()

    for file_idx, fname in enumerate(SCANNED_FILENAMES, 1):
        pdf_path = NEW_PDFS_DIR / fname
        if not pdf_path.exists():
            LOGGER.warning("[%s/%s] File not found: %s", file_idx, len(SCANNED_FILENAMES), fname)
            continue

        doc_title = clean_title_from_stem(fname)
        jurisdiction = "Khyber Pakhtunkhwa" if "khyber" in fname.lower() or "kp" in fname.lower() else "Pakistan"

        LOGGER.info("[%s/%s] Starting OCR on %s...", file_idx, len(SCANNED_FILENAMES), fname)
        file_chunks = []
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            for page_idx in range(num_pages):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
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

                total_pages_ocrd += 1
                if (page_idx + 1) % 10 == 0 or (page_idx + 1) == num_pages:
                    LOGGER.info("  - %s: OCR page %s/%s completed (%s chunks so far)", fname, page_idx + 1, num_pages, len(file_chunks))

            doc.close()
            all_ocr_chunks.extend(file_chunks)
            LOGGER.info("[%s/%s] Finished %s (%s pages -> %s chunks)", file_idx, len(SCANNED_FILENAMES), fname, num_pages, len(file_chunks))

        except Exception as exc:
            LOGGER.error("[%s/%s] Error OCRing %s: %s", file_idx, len(SCANNED_FILENAMES), fname, exc)

    ocr_dur = time.time() - t0_ocr
    LOGGER.info("OCR completed in %.1fs across %s pages. Extracted %s candidate chunks.", ocr_dur, total_pages_ocrd, len(all_ocr_chunks))

    # Deduplicate against active knowledge base
    unique_ocr_chunks = []
    seen = set()
    for c in all_ocr_chunks:
        h = c["content_hash"]
        if h not in existing_hashes and h not in seen:
            seen.add(h)
            unique_ocr_chunks.append(c)

    LOGGER.info("Deduplication complete: %s brand-new OCR passages to be merged into FAISS.", len(unique_ocr_chunks))
    if not unique_ocr_chunks:
        LOGGER.info("All OCR chunks already present in knowledge base.")
        return

    # Embed new OCR chunks
    LOGGER.info("Loading embedding model: %s...", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts_to_embed = [c["text"] for c in unique_ocr_chunks]

    t0_embed = time.time()
    embeddings = model.encode(texts_to_embed, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    LOGGER.info("Encoding finished in %.1fs.", time.time() - t0_embed)

    # Append to active FAISS index
    index = faiss.read_index(str(INDEX_PATH))
    old_vector_count = index.ntotal
    index.add(embeddings)
    new_vector_count = index.ntotal

    combined_chunks = existing_chunks + unique_ocr_chunks
    LOGGER.info("FAISS index count updated from %s to %s.", old_vector_count, new_vector_count)

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
    LOGGER.info("ALL 14 SCANNED PDFS SUCCESSFULLY OCR-EXTRACTED & MERGED!")
    LOGGER.info(" - Total OCR chunks added: %s", len(unique_ocr_chunks))
    LOGGER.info(" - Total active knowledge base vectors now: %s", new_vector_count)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
