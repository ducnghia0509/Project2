# core/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Tải biến môi trường từ file .env

# --- Cấu hình Database ---
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

# --- Cấu hình API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Nếu dùng

# --- Cấu hình Thu thập Dữ liệu ---
DATA_START_DATE = os.getenv("StartDate", "2015-01-01")
DATA_END_DATE = os.getenv("EndDate", "2025-03-31") 
DEFAULT_CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "RNDR-USD", "FET-USD", "AGIX-USD", "OCEAN-USD",
    "ARB-USD", "OP-USD", "MATIC-USD", "STRK-USD",
    "UNI-USD", "AAVE-USD", "MKR-USD", "CRV-USD", "SNX-USD",
    "USDT-USD", "USDC-USD",
    "DOGE-USD", "ADA-USD", "TRX-USD",
]
UPDATE_DEFAULT_START_DATE = "2010-01-01" 

# --- Cấu hình Mô hình Dự đoán ---
PREDICTION_TARGET_TICKERS = [
    # Dài hạn
    "BTC-USD", "ETH-USD",
    # Ngắn hạn
    "SOL-USD", "BNB-USD", "XRP-USD",
    # Tiềm năng (AI, Layer 2)
    "RNDR-USD", "FET-USD", "AGIX-USD", "OCEAN-USD", # AI
    "ARB-USD", "OP-USD", "MATIC-USD", "STRK-USD",  # Layer 2
    # DeFi
    "UNI-USD", "AAVE-USD", "MKR-USD", "CRV-USD", "SNX-USD",
    # Stablecoins 
    "USDT-USD", "USDC-USD",
    # Khác
    "DOGE-USD", "ADA-USD", "TRX-USD",
]
PREDICTION_OUTPUT_LENGTHS = [15, 30, 60]
PREDICTION_INPUT_FEATURES = ['open', 'high', 'low', 'close', 'volume']
PREDICTION_TARGET_FEATURE = 'close'
PREDICTION_SEQUENCE_LENGTH = 90

PREDICTION_NUM_FEATURES = len(PREDICTION_INPUT_FEATURES)
try:
    PREDICTION_TARGET_COL_INDEX_GLOBAL = PREDICTION_INPUT_FEATURES.index(PREDICTION_TARGET_FEATURE) 
except ValueError:
    raise ValueError(f"Cấu hình lỗi: PREDICTION_TARGET_FEATURE '{PREDICTION_TARGET_FEATURE}' "
                     f"không tìm thấy trong PREDICTION_INPUT_FEATURES.")

PREPARED_DATA_DIR_BASE = 'prepared_data_multi'
TRAINED_MODELS_DIR_BASE = 'trained_models'

# Cấu hình huấn luyện
EPOCHS = 24
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
LSTM_UNITS = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.00005 
PATIENCE_EARLY_STOPPING = 10
PATIENCE_REDUCE_LR = 5

# --- Cấu hình RAG & LLM ---
RAG_DOCUMENTS_PATH = "knowledge_base/documents"
RAG_VECTORSTORE_PATH = "knowledge_base/vectorstore_chroma"
RAG_CHUNK_SIZE = 8000
RAG_CHUNK_OVERLAP = 100
RAG_EMBEDDING_PROVIDER = "HuggingFace"
# RAG_EMBEDDING_MODEL_NAME_HF = "sentence-transformers/multilingual-e5-large"
RAG_EMBEDDING_MODEL_NAME_HF = "BAAI/bge-m3"
RAG_EMBEDDING_MODEL_NAME_GOOGLE = "models/embedding-001" 
RAG_LLM_PROVIDER = "Google"
RAG_LLM_MODEL_NAME_GOOGLE = "gemini-2.0-flash"

# --- Cấu hình API & Dashboard ---
API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PREDICTION_HORIZON_DISPLAY = 15
AVAILABLE_PREDICTION_HORIZONS_DISPLAY = [15] 

# --- Các hàm tiện ích chung ---
def get_db_table_name(ticker_symbol: str) -> str:
    base_ticker = ticker_symbol.split('-')[0].lower().replace('.', '_')
    return f"{base_ticker}_price"