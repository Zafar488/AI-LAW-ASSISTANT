"""Extract, clean, structure, and chunk all local legal PDFs in data/pdfs.

Designed with legal RAG best-practices:
- Robust text extraction using PyMuPDF (fitz)
- Hyphenated line-break repair and ligature normalization
- Section, Article, Order, and Rule detection
- Context prefix injection: [Document: <Title> | Section: <Sec> | Page: <Page>]
- Sliding window chunking respecting paragraph and sentence boundaries
- Structured JSON output with complete provenance metadata
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import fitz

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_JSON = BASE_DIR / "embeddings" / "kb_chunks.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180

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
    # Repair hyphenated line breaks (e.g., 'juris-\ndiction' -> 'jurisdiction')
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", raw_text)
    # Normalize multiple newlines and spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove standalone page numbers or header lines like 'Page 12 of 85'
    text = re.sub(r"(?i)\bpage\s+\d+\s+(?:of\s+\d+)?\b", "", text)
    return text.strip()


def detect_section(text: str) -> str | None:
    """Identify statutory section or article from chunk text."""
    for pattern in SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return f"Order {groups[0]} Rule {groups[1]}"
            return f"Section {groups[0]}" if "article" not in pattern.pattern.lower() else f"Article {groups[0]}"
    return None


def get_document_title(filename: str) -> str:
    """Clean statutory title from filename."""
    stem = Path(filename).stem
    # Remove numbers, hashes, trailing garbage
    title = re.sub(r"^\d+[\s._-]+", "", stem)
    title = re.sub(r"[_\-]+", " ", title)
    title = re.sub(r"(?i)\b(pdf|compressed|dated.*|doc.*)\b", "", title).strip()
    return title.title() if title else stem


def chunk_document_page(
    page_text: str,
    doc_title: str,
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
                    "jurisdiction": "Pakistan",
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


def process_all_local_pdfs() -> list[dict[str, Any]]:
    if not PDF_DIR.exists():
        print(f"Error: PDF directory not found at {PDF_DIR}")
        return []

    pdf_files = sorted([f for f in PDF_DIR.iterdir() if f.suffix.lower() == ".pdf"])
    print(f"Found {len(pdf_files)} local PDF files in {PDF_DIR}.")

    all_chunks: list[dict[str, Any]] = []
    scanned_files: list[str] = []
    processed_count = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_title = get_document_title(pdf_path.name)
        try:
            doc = fitz.open(pdf_path)
            doc_chunks = []
            pages_with_text = 0

            for page_idx, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                if len(page_text.strip()) > 30:
                    pages_with_text += 1
                page_chunks = chunk_document_page(
                    page_text=page_text,
                    doc_title=doc_title,
                    page_num=page_idx,
                    source_filename=pdf_path.name,
                )
                doc_chunks.extend(page_chunks)

            if pages_with_text == 0:
                scanned_files.append(pdf_path.name)
                print(f"[{idx}/{len(pdf_files)}] {pdf_path.name} -> SCANNED (0 text pages, skipped for OCR)")
            else:
                processed_count += 1
                all_chunks.extend(doc_chunks)
                print(f"[{idx}/{len(pdf_files)}] {pdf_path.name} ({len(doc)} pages) -> {len(doc_chunks)} chunks")

        except Exception as e:
            print(f"[{idx}/{len(pdf_files)}] ERROR reading {pdf_path.name}: {e}")

    # Deduplicate chunks across documents
    unique_chunks = []
    seen_hashes = set()
    for chunk in all_chunks:
        h = chunk["content_hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_chunks.append(chunk)

    print("\n" + "=" * 80)
    print(f"Extraction Summary:")
    print(f"Total PDFs examined: {len(pdf_files)}")
    print(f"Successfully processed PDFs: {processed_count}")
    print(f"Scanned / image-only PDFs: {len(scanned_files)}")
    print(f"Total raw chunks extracted: {len(all_chunks)}")
    print(f"Unique deduplicated chunks: {len(unique_chunks)}")
    print("=" * 80)

    # Save to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    temp_json = OUTPUT_JSON.with_suffix(".tmp")
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, ensure_ascii=False, indent=2)
    os.replace(temp_json, OUTPUT_JSON)
    print(f"Successfully saved {len(unique_chunks)} chunks to {OUTPUT_JSON}")

    return unique_chunks


if __name__ == "__main__":
    process_all_local_pdfs()
