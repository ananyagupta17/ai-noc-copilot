import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
INCIDENTS_DIR = DATA_DIR / "incidents"
LOGS_DIR = DATA_DIR / "logs"
ALERTS_DIR = DATA_DIR / "alerts"
RUNBOOKS_DIR = DATA_DIR / "runbooks"
TOPOLOGY_DIR = DATA_DIR / "topology"

# DB
DB_DIR = BASE_DIR / "db"
SQLITE_PATH = DB_DIR / "noc.db"
CHROMA_PATH = DB_DIR / "chroma"

# LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = "gemini-2.5-flash" 

# RAG
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 5

# Agent
MAX_TOOL_CALLS = 10