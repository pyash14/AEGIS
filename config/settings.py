import os
from dotenv import load_dotenv

load_dotenv()

# Anthropic
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
CLAUDE_MODEL = 'claude-sonnet-4-5'
MAX_TOKENS = 2048

# FAISS
MEDICAL_INDEX_DIR = 'indexes/medical_index'
NUTRITION_INDEX_DIR = 'indexes/nutrition_index'
MEDICAL_DOCS_DIR = 'data/medical_docs'
FDC_CSV_PATH = 'data/fdc_foods_filtered.csv'
TOP_K_CHUNKS = 4

# Database
DB_PATH = 'database/patient_logs.db'
CHAT_HISTORY_LIMIT = 5

# Clinical constants (DO NOT CHANGE)
ICR = 10.0        # g carbs per unit insulin
ISF = 3.0         # mmol/L drop per unit
TARGET = 7.8      # mmol/L
HYPO_THRESHOLD = 3.9
HYPER_THRESHOLD = 10.0

# Safety monitor thresholds
HYPO_COUNT_24H = 3
NOCTURNAL_START = 0
NOCTURNAL_END = 6
STACKING_IOB_THRESHOLD = 3.0
RAPID_DROP_RATE = 0.1