import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Общие пути проекта
ROOT_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT_DIR / "questions.json"
GROUND_TRUTH_PATH = ROOT_DIR / "ground_truth.json"
METRICS_DIR = ROOT_DIR / "metrics"

# Настройки для 01_llm_without_rag.py
BASELINE_MODEL_NAME = os.getenv("BASELINE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
BASELINE_MAX_INPUT_TOKENS = 2048
BASELINE_MAX_NEW_TOKENS = 120

# Настройки для 02_rag_pipeline.py
ARTICLES_PATH = ROOT_DIR / "articles.json"
CHROMA_DIR = ROOT_DIR / "chroma_db"
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
RETRIEVER_K = 3
RAG_MAX_INPUT_TOKENS = 3072
RAG_MAX_NEW_TOKENS = 120

# Настройки для llm_as_judge
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")
