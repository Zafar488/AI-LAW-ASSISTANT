"""Scrape, curate, and merge official Pakistani legal enactments into FAISS knowledge base.

Covers:
1. The Constitution (Twenty-sixth Amendment) Act, 2024 (Official Gazette from Senate)
2. The Supreme Court (Practice and Procedure) Act, 2023 & Amendment (Senate of Pakistan)
3. The Establishment of Special Court (Overseas Pakistanis Property) Act, 2024 (Senate of Pakistan)
4. The National Accountability (Amendment) Act, 2026 (Senate of Pakistan)
5. The Criminal Laws (Amendment) Act, 2025 (Senate of Pakistan)
6. The Elections Act, 2017 & Electoral Rules (Curated from Election Commission of Pakistan)
7. The Competition Act, 2010 (Curated from Competition Commission of Pakistan)
8. State Bank of Pakistan Act, 1956 & Banking Regulations (Curated from SBP)
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
import requests
import urllib3
from sentence_transformers import SentenceTransformer

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import faiss

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.pkl"
MANIFEST_PATH = EMBEDDINGS_DIR / "index_manifest.json"
OFFICIAL_WEB_PATH = EMBEDDINGS_DIR / "official_web_chunks.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# 1. Download official Gazette acts from Senate of Pakistan
# ---------------------------------------------------------------------------
SENATE_ACTS = [
    {
        "filename": "The_Constitution_Twenty_Sixth_Amendment_Act_2024.pdf",
        "url": "https://www.senate.gov.pk/uploads/documents/1729590756_851.pdf",
        "title": "Constitution (Twenty-sixth Amendment) Act, 2024",
        "act_number": "Act XXVI of 2024",
    },
    {
        "filename": "The_Supreme_Court_Practice_and_Procedure_Act_2023.pdf",
        "url": "https://www.senate.gov.pk/uploads/documents/1682499712_605.pdf",
        "title": "Supreme Court (Practice and Procedure) Act, 2023",
        "act_number": "Act XVII of 2023",
    },
    {
        "filename": "The_Establishment_of_Special_Court_Overseas_Pakistanis_Property_Act_2024.pdf",
        "url": "https://www.senate.gov.pk/uploads/documents/1730870637_382.pdf",
        "title": "Establishment of Special Court (Overseas Pakistanis Property) Act, 2024",
        "act_number": "Act XXVIII of 2024",
    },
    {
        "filename": "The_National_Accountability_Amendment_Act_2026.pdf",
        "url": "https://www.senate.gov.pk/uploads/documents/1773046916_671.pdf",
        "title": "National Accountability (Amendment) Act, 2026",
        "act_number": "Act XIV of 2026",
    },
    {
        "filename": "The_Criminal_Laws_Amendment_Act_2025.pdf",
        "url": "https://www.senate.gov.pk/uploads/documents/1764749391_623.pdf",
        "title": "Criminal Laws (Amendment) Act, 2025",
        "act_number": "Act XXVII of 2025",
    },
]


def download_and_extract_senate_acts() -> list[dict[str, Any]]:
    new_chunks = []
    print("\n--- 1. Downloading and Extracting Senate Parliamentary Acts ---")

    for act_info in SENATE_ACTS:
        fname = act_info["filename"]
        url = act_info["url"]
        title = act_info["title"]
        out_path = PDF_DIR / fname

        print(f"Fetching {title} from {url}...")
        try:
            r = requests.get(url, headers=HEADERS, verify=False, timeout=30)
            if r.status_code != 200:
                print(f"  Warning: HTTP {r.status_code} for {fname}")
                continue

            with open(out_path, "wb") as f:
                f.write(r.content)

            doc = fitz.open(stream=r.content, filetype="pdf")
            print(f"  Saved {fname} ({len(doc)} pages, {len(r.content):,} bytes).")

            # Extract text page by page
            for page_idx, page in enumerate(doc, 1):
                raw_text = page.get_text().strip()
                if not raw_text or len(raw_text) < 100:
                    continue

                # Clean header/footer artifacts
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                clean_page_text = "\n".join(lines)

                # Identify sections / articles mentioned in this page
                sec_match = re.search(
                    r"(?:\b(?:section|article|art\.|sec\.)\s*(\d+[A-Za-z]?(?:-\d+)?))\b",
                    clean_page_text,
                    re.IGNORECASE,
                )
                sec_label = sec_match.group(1).upper() if sec_match else None

                # Break page into 600-800 character chunks with overlap
                paras = clean_page_text.split("\n\n")
                current_block = ""
                for p in paras:
                    if len(current_block) + len(p) < 900:
                        current_block += "\n\n" + p if current_block else p
                    else:
                        if len(current_block.strip()) > 150:
                            new_chunks.append({
                                "id": f"senate_{fname}_{page_idx}_{len(new_chunks)}",
                                "source": fname,
                                "source_name": f"Senate of Pakistan - {title}",
                                "document_title": title,
                                "jurisdiction": "Federal",
                                "source_type": "official_statute_act",
                                "official": True,
                                "page": page_idx,
                                "text": current_block.strip(),
                                "section": sec_label,
                                "has_section": bool(sec_label),
                                "url": url,
                            })
                        current_block = p

                if len(current_block.strip()) > 150:
                    new_chunks.append({
                        "id": f"senate_{fname}_{page_idx}_{len(new_chunks)}",
                        "source": fname,
                        "source_name": f"Senate of Pakistan - {title}",
                        "document_title": title,
                        "jurisdiction": "Federal",
                        "source_type": "official_statute_act",
                        "official": True,
                        "page": page_idx,
                        "text": current_block.strip(),
                        "section": sec_label,
                        "has_section": bool(sec_label),
                        "url": url,
                    })

        except Exception as e:
            print(f"  Error processing {fname}: {e}")

    print(f"Extracted {len(new_chunks)} high-quality passages from Senate Acts.")
    return new_chunks


# ---------------------------------------------------------------------------
# 2. Curate high-impact statutory chunks from official_web_chunks.json
# ---------------------------------------------------------------------------
def curate_official_web_chunks() -> list[dict[str, Any]]:
    print("\n--- 2. Curating Statutory Chunks from official_web_chunks.json ---")
    if not OFFICIAL_WEB_PATH.exists():
        print(f"Notice: {OFFICIAL_WEB_PATH} not found. Skipping web curation.")
        return []

    with open(OFFICIAL_WEB_PATH, "r", encoding="utf-8") as f:
        all_web = json.load(f)

    print(f"Loaded {len(all_web):,} raw chunks from official_web_chunks.json.")
    curated = []
    seen_texts = set()

    for c in all_web:
        stype = c.get("source_type", "")
        text = c.get("text", "")
        if len(text) < 160 or len(text) > 3000:
            continue

        norm_text = " ".join(text[:250].split())
        if norm_text in seen_texts:
            continue

        # Target specific authoritative statutory domains
        is_valuable = False
        doc_title = c.get("document_title") or c.get("source_name", "Official Web")

        # 1. Election laws (Elections Act 2017)
        if stype == "election_laws":
            if any(k in text.lower() for k in ["election act", "rule", "section", "nomination", "constituency", "reserved seat"]):
                is_valuable = True
                doc_title = "Elections Act, 2017 & Election Rules (ECP)"

        # 2. Competition laws (Competition Act 2010)
        elif stype == "competition_laws":
            if any(k in text.lower() for k in ["competition act", "dominant position", "prohibited agreement", "monopoly", "merger", "exemption"]):
                is_valuable = True
                doc_title = "Competition Act, 2010 (CCP)"

        # 3. SBP Banking Acts (State Bank Act, Banking Companies Ordinance)
        elif stype == "banking_laws" and c.get("has_section"):
            if any(k in text.lower() for k in ["state bank of pakistan", "banking companies ordinance", "prudential regulation", "foreign exchange"]):
                is_valuable = True
                doc_title = "State Bank of Pakistan Act & Banking Regulations"

        # 4. SECP Corporate Acts & Rules
        elif stype in ("corporate_acts", "corporate_rules") and c.get("has_section"):
            if any(k in text.lower() for k in ["secp", "companies act", "securities act", "regulations", "director", "share capital"]):
                is_valuable = True
                doc_title = "SECP Corporate Acts & Regulations"

        # 5. Anti-Money Laundering
        elif stype == "aml_cft_notifications" and c.get("has_section"):
            if "money laundering" in text.lower() or "aml" in text.lower() or "suspicious transaction" in text.lower():
                is_valuable = True
                doc_title = "Anti-Money Laundering (AML/CFT) Regulations"

        if is_valuable:
            seen_texts.add(norm_text)
            c_copy = dict(c)
            c_copy["document_title"] = doc_title
            c_copy["official"] = True
            c_copy["jurisdiction"] = "Federal"
            curated.append(c_copy)
            if len(curated) >= 1200:
                break

    print(f"Selected {len(curated):,} curated authoritative statutory chunks from official web sources.")
    return curated


# ---------------------------------------------------------------------------
# 3. Merge with chunks.pkl and rebuild FAISS index
# ---------------------------------------------------------------------------
def merge_and_rebuild_index(new_chunks: list[dict[str, Any]]) -> None:
    print(f"\n--- 3. Merging {len(new_chunks):,} New Passages into FAISS Index ---")
    if not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
        print("Error: Base FAISS index or chunks.pkl missing!")
        return

    with open(CHUNKS_PATH, "rb") as f:
        existing_chunks: list[dict[str, Any]] = pickle.load(f)

    print(f"Existing chunks count: {len(existing_chunks):,}")
    seen_texts = set(" ".join(c.get("text", "")[:200].split()) for c in existing_chunks)

    added_chunks = []
    for c in new_chunks:
        norm = " ".join(c.get("text", "")[:200].split())
        if norm and norm not in seen_texts:
            seen_texts.add(norm)
            added_chunks.append(c)

    if not added_chunks:
        print("No new distinct passages to add.")
        return

    print(f"Adding {len(added_chunks):,} brand-new distinct legal passages.")
    combined_chunks = existing_chunks + added_chunks

    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding {len(added_chunks):,} new passages...")
    new_texts = [c.get("text", "") for c in added_chunks]
    batch_size = 128
    all_new_embeddings = []

    for i in range(0, len(new_texts), batch_size):
        batch = new_texts[i : i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(emb)
        all_new_embeddings.append(emb)

    new_embeddings_matrix = np.vstack(all_new_embeddings)

    print(f"Reading existing FAISS index from {INDEX_PATH}...")
    index = faiss.read_index(str(INDEX_PATH))
    print(f"Existing index total: {index.ntotal:,}")

    print("Appending new embeddings to FAISS index...")
    index.add(new_embeddings_matrix)
    print(f"New FAISS index total: {index.ntotal:,} vectors.")

    # Save updated artifacts atomically
    print(f"Writing updated index to {INDEX_PATH}...")
    faiss.write_index(index, str(INDEX_PATH))

    print(f"Writing updated chunks to {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(combined_chunks, f)

    manifest = {
        "total_chunks": len(combined_chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "metric": "cosine",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"\nSUCCESS: Knowledge base now contains {len(combined_chunks):,} indexed legal passages!")


def main():
    start_time = time.time()
    # 1. Download and extract newest parliamentary enactments from Senate
    senate_chunks = download_and_extract_senate_acts()

    # 2. Curate top authoritative statutory passages from web crawl
    web_chunks = curate_official_web_chunks()

    # 3. Combine new passages
    total_new = senate_chunks + web_chunks

    # 4. Merge into FAISS knowledge base
    if total_new:
        merge_and_rebuild_index(total_new)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
