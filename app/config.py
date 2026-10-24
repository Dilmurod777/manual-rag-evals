"""Every knob in one place, overridable from the environment."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
CHAT_URL = f"{OLLAMA_HOST}/api/chat"
MODEL = os.environ.get("RAG_MODEL", "qwen3.8-27b")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "qwen3-embedding:0.6b")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
REFUSE_BELOW = float(os.environ.get("RAG_REFUSE_BELOW", "0.58"))
NUM_CTX = int(os.environ.get("RAG_NUM_CTX", "8192"))
LOG_LEVEL = os.environ.get("RAG_LOG", "INFO")
CORPUS_URL = os.environ.get(
    "RAG_CORPUS_URL",
    "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43.13-1B_w-chg1.pdf",
)
