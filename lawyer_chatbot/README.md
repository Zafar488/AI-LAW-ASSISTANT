# Pakistan Legal Research Chatbot

A multilingual legal research assistant for Pakistani law. The project combines
a FastAPI backend, a Streamlit chat and voice interface, FAISS semantic search,
and Groq-hosted language and speech models.

The assistant retrieves relevant passages from the local legal knowledge base
before generating an answer and returns the supporting sources with each
response.

> **Disclaimer:** This application is intended for legal research and
> educational use only. Its responses may be incomplete or inaccurate and do
> not constitute legal advice. Consult a qualified lawyer for advice about a
> specific matter.

## Features

- Retrieval-augmented answers grounded in Pakistani legal documents
- Local FAISS vector index using `all-MiniLM-L6-v2` embeddings
- English, Urdu, Hindi, and Pashto language detection
- Voice input through Groq Whisper transcription
- Browser-based text-to-speech for generated answers
- Source references returned with answers
- FastAPI endpoints with API-key authentication
- Streamlit interface with conversation export
- PDF ingestion and configurable official-source web crawling

## Project structure

```text
lawyer_chatbot/
├── api/                  # FastAPI application and endpoints
├── app/                  # Chatbot, retrieval, ingestion, and language logic
├── data/
│   ├── pdfs/             # Local legal PDF documents
│   └── lawyer_links.json # Configured web sources
├── embeddings/           # FAISS index, chunks, and crawl metadata
├── ui/                   # Streamlit and voice-output interface
├── requirements.txt
└── run_server.py
```

## Requirements

- Python 3.11 to 3.13
- A [Groq API key](https://console.groq.com/keys)
- Enough disk space and memory to load the embedding and translation models

The first model load may take longer because Hugging Face model files need to be
downloaded.

## Installation

Run all commands from the directory containing `run_server.py`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
API_KEY=choose_a_private_key_for_this_application

# Optional
GROQ_MODEL=llama-3.1-8b-instant
STT_MODEL=whisper-large-v3-turbo
API_URL=http://127.0.0.1:8000
```

`API_KEY` is a private key you choose yourself. The Streamlit UI sends it to the
backend in the `key` request header. Do not commit `.env`; it is already ignored
by Git.

## Running the application

Start the API:

```bash
python run_server.py
```

The backend is available at `http://127.0.0.1:8000`. Confirm that its
configuration and knowledge base are ready:

```text
http://127.0.0.1:8000/health
```

In a second terminal, activate the same virtual environment and start the UI:

```bash
streamlit run ui/streamlit_app.py
```

Open the local URL displayed by Streamlit, normally
`http://localhost:8501`.

## API usage

Interactive OpenAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Ask a question with PowerShell:

```powershell
$headers = @{
    key = "your_application_api_key"
    "Content-Type" = "application/json"
}
$body = @{
    message = "What are the requirements for a valid contract in Pakistan?"
    top_k = 8
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/ask" `
    -Method Post `
    -Headers $headers `
    -Body $body
```

Or with cURL:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "key: your_application_api_key" \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the requirements for a valid contract in Pakistan?","top_k":8}'
```

The main endpoints are:

| Method | Endpoint | Purpose | Authentication |
|---|---|---|---|
| `GET` | `/` | API information | None |
| `GET` | `/health` | Configuration and index readiness | None |
| `POST` | `/ask` | Retrieve sources and answer a question | `key` header |
| `POST` | `/transcribe` | Transcribe an uploaded audio file | `key` header |

`top_k` accepts a value from 1 to 15 and defaults to 8.

## Knowledge-base management

The application uses these active retrieval artifacts:

- `embeddings/faiss.index`
- `embeddings/chunks.pkl`

If they already exist, no rebuild is needed to run the chatbot.

### Build from configured web links

The default loader mode reads sources from `data/lawyer_links.json`:

```bash
python -m app.loader
```

Useful crawl options include:

```bash
python -m app.loader --resume
python -m app.loader --source-limit 5 --max-pages 20
python -m app.loader --skip-embeddings
```

### Build from local PDFs

PDF files are processed only when explicitly requested:

```bash
python -m app.loader --local-pdfs-only
```

To process only PDFs missing from the active index metadata:

```bash
python -m app.loader --missing-pdfs-only
```

Optional OCR flags are available for PDFs without extractable text:

```bash
python -m app.loader --local-pdfs-only --ocr-local-pdfs
python -m app.loader --ocr-remote-pdfs
```

OCR may require additional optional dependencies not listed in the base
requirements.

### Rebuild FAISS from existing chunk data

After collecting chunks with `--skip-embeddings`, rebuild the index with:

```bash
python -m app.embedder
```

Large crawls can take considerable time. Use `--resume`, `--source-limit`, and
`--max-pages` to control the workload.

## Tests

Run the test suite from the project root:

```bash
python -m unittest -v
```

The included tests cover language detection and voice-service behavior without
making live speech API calls.

## Troubleshooting

- **`/health` reports `not_ready`:** Check that `API_KEY` and `GROQ_API_KEY`
  are set and that both FAISS artifacts exist.
- **The UI cannot connect:** Start `run_server.py` first and verify `API_URL`.
- **The first question is slow:** The embedding and translation models are
  loaded lazily and may be downloaded on first use.
- **Unauthorized response:** Ensure the UI or client sends the same `API_KEY`
  value in the `key` header.
- **No microphone access:** Grant microphone permission to the browser. Audio is
  read in memory for transcription and is not saved by the application.

## Security notes

- Keep `.env` and all API keys private.
- Replace the permissive CORS configuration before exposing the API publicly.
- Use HTTPS and a secret-management service in production.
- Review generated answers and cited source material before relying on them.
