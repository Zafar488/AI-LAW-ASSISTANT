from pathlib import Path

# Base directory (project root: lawyer_chatbot/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
INPUT_DIR = BASE_DIR / "data" / "pdfs"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
LINKS_JSON = BASE_DIR / "data" / "lawyer_links.json"
OUTPUT_JSON = EMBEDDINGS_DIR / "kb_chunks.json"
OFFICIAL_OUTPUT_JSON = EMBEDDINGS_DIR / "official_web_chunks.json"
PDF_UPDATES_JSON = EMBEDDINGS_DIR / "pdf_updates.json"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
