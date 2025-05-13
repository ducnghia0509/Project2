# app/main_api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
from datetime import date, datetime
from typing import List, Optional
from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle
from sqlalchemy import text

from llm_rag.rag_handler import ask_question as ask_rag_question, ingest_data_to_vectorstore as ingest_rag_data
from core.db_connect import engine
from analysis.technical_analyzer import load_price_data_and_calculate_ta
from core.config import (
    PREPARED_DATA_DIR_BASE, 
    TRAINED_MODELS_DIR_BASE,
    PREDICTION_SEQUENCE_LENGTH,
    PREDICTION_TARGET_FEATURE, 
    PREDICTION_OUTPUT_LENGTHS,
    get_db_table_name 
)
from prediction.prediction_utils import inverse_transform_predictions

app = FastAPI(
    title="Crypto Analysis API",
    description="API cung cấp dữ liệu giá, phân tích kỹ thuật và dự đoán crypto.",
    version="0.1.0"
)

# --- Định nghĩa Model cho Dữ liệu (Pydantic) ---
class PriceDataPoint(BaseModel):
    date: date 
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    SMA_20: Optional[float] = None
    SMA_50: Optional[float] = None
    EMA_20: Optional[float] = None
    RSI_14: Optional[float] = None
    MACD_12_26_9: Optional[float] = None
    MACDh_12_26_9: Optional[float] = None
    MACDs_12_26_9: Optional[float] = None
    BBL_20_2_0: Optional[float] = None
    BBM_20_2_0: Optional[float] = None
    BBU_20_2_0: Optional[float] = None

class PredictionPoint(BaseModel):
    date_index: int
    predicted_price: float

class FuturePredictionResponse(BaseModel):
    ticker: str
    predictions: List[PredictionPoint]
    last_actual_date: Optional[date] = None
    last_actual_close: Optional[float] = None

class RagQuery(BaseModel):
    question: str

class RagSource(BaseModel):
    content_preview: Optional[str] = None
    metadata: Optional[dict] = None

class RagResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    sources: Optional[List[RagSource]] = None
    error: Optional[str] = None

class NewsArticle(BaseModel):
    article_id: str
    title: str
    url: str
    domain: Optional[str] = None
    published_at: datetime
    source_info: Optional[str] = None
    related_currencies: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None

# --- Helper functions ---
MODEL_CACHE = {}

def get_specific_model_path(base_dir: str, ticker: str, output_len: int) -> str:
    ticker_subdir = ticker.lower().replace('-', '_')
    model_dir = os.path.join(base_dir, ticker_subdir, f"output_{output_len}")
    return os.path.join(model_dir, "model_lstm.h5")

def get_specific_data_path_elements(base_dir: str, ticker: str, output_len: int) -> dict:
    ticker_subdir = ticker.lower().replace('-', '_')
    data_dir = os.path.join(base_dir, ticker_subdir, f"output_{output_len}")
    return {
        "dir": data_dir,
        "scaler": os.path.join(data_dir, "data_scaler.pkl"),
        "original_scaled": os.path.join(data_dir, "data_original_scaled.npy"),
        "target_idx": os.path.join(data_dir, "data_target_idx.txt")
    }

def get_prediction_model_and_data(ticker_symbol: str, prediction_horizon: int):
    cache_key = (ticker_symbol, prediction_horizon)
    if cache_key in MODEL_CACHE:
        print(f"Model and data for {ticker_symbol}/Out:{prediction_horizon} found in cache.")
        return MODEL_CACHE[cache_key]

    model_path = get_specific_model_path(TRAINED_MODELS_DIR_BASE, ticker_symbol, prediction_horizon)
    data_paths = get_specific_data_path_elements(PREPARED_DATA_DIR_BASE, ticker_symbol, prediction_horizon)

    if not os.path.exists(model_path):
        print(f"Model for {ticker_symbol}/Out:{prediction_horizon} not found at {model_path}")
        return None, None, None, -1, -1

    required_data_files = ["scaler", "original_scaled", "target_idx"]
    if not all(os.path.exists(data_paths[key]) for key in required_data_files):
        missing = [key for key in required_data_files if not os.path.exists(data_paths[key])]
        print(f"Data files {missing} for {ticker_symbol}/Out:{prediction_horizon} not found in {data_paths['dir']}")
        return None, None, None, -1, -1

    try:
        model = load_model(model_path)
        print(f"Model for {ticker_symbol}/Out:{prediction_horizon} loaded from {model_path}")

        with open(data_paths["scaler"], 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        original_scaled_data = np.load(data_paths["original_scaled"])
        with open(data_paths["target_idx"], 'r') as f_idx:
            target_col_index = int(f_idx.read())

        actual_num_features = scaler.n_features_in_

        MODEL_CACHE[cache_key] = (model, scaler, original_scaled_data, target_col_index, actual_num_features)
        return model, scaler, original_scaled_data, target_col_index, actual_num_features
    except Exception as e:
        print(f"Error loading model or data for {ticker_symbol}/Out:{prediction_horizon}: {e}")
        return None, None, None, -1, -1

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "API is running"}

@app.get("/crypto/{ticker_symbol}/history", response_model=List[PriceDataPoint])
async def get_crypto_history_with_ta(ticker_symbol: str, limit: Optional[int] = 100):
    df_with_ta = load_price_data_and_calculate_ta(ticker_symbol)

    if df_with_ta is None or df_with_ta.empty:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu hoặc không thể tính TA cho {ticker_symbol}")

    df_limited = df_with_ta.tail(limit)
    records = df_limited.replace({np.nan: None}).to_dict(orient='records')
    return records

@app.get("/crypto/{ticker_symbol}/predict-future", response_model=FuturePredictionResponse)
async def predict_future_prices(ticker_symbol: str, horizon: int = 30): 
    if horizon not in PREDICTION_OUTPUT_LENGTHS:
        raise HTTPException(status_code=400, detail=f"Horizon không hợp lệ. Chọn từ: {PREDICTION_OUTPUT_LENGTHS}")

    model, scaler, original_scaled_data, target_idx, num_feats = get_prediction_model_and_data(ticker_symbol, horizon)

    if model is None or scaler is None or original_scaled_data is None or target_idx == -1:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy mô hình/dữ liệu cho {ticker_symbol} với horizon {horizon} ngày.")

    if len(original_scaled_data) < PREDICTION_SEQUENCE_LENGTH:
         raise HTTPException(status_code=500, detail=f"Dữ liệu gốc quá ngắn cho {ticker_symbol} / Horizon {horizon}")

    last_sequence = original_scaled_data[-PREDICTION_SEQUENCE_LENGTH:, :] 
    last_sequence_reshaped = np.reshape(last_sequence, (1, PREDICTION_SEQUENCE_LENGTH, num_feats)) 

    predicted_future_scaled = model.predict(last_sequence_reshaped)

    predicted_future_prices = inverse_transform_predictions(
        predicted_future_scaled, scaler, target_idx, num_feats
    )[0]

    try:
        db_table_name = get_db_table_name(ticker_symbol)
        df_raw = pd.read_sql(
            f"SELECT \"date\", \"{PREDICTION_TARGET_FEATURE}\" FROM {db_table_name} ORDER BY \"date\" DESC LIMIT 1",
            engine
        )
        last_actual_date_val = df_raw['date'].iloc[0].date() if not df_raw.empty else None
        last_actual_close_val = df_raw[PREDICTION_TARGET_FEATURE].iloc[0] if not df_raw.empty else None
    except Exception as e:
        print(f"Lỗi khi lấy last actual date/price cho {ticker_symbol}: {e}")
        last_actual_date_val = None
        last_actual_close_val = None

    predictions_output = [
        PredictionPoint(date_index=i + 1, predicted_price=price)
        for i, price in enumerate(predicted_future_prices)
    ]

    return FuturePredictionResponse(
        ticker=ticker_symbol,
        predictions=predictions_output,
        last_actual_date=last_actual_date_val,
        last_actual_close=last_actual_close_val
    )

@app.post("/rag/ask", response_model=RagResponse)
async def ask_knowledge_base(query: RagQuery):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    try:
        result = ask_rag_question(query.question)
        return result
    except Exception as e:
        print(f"Lỗi trong endpoint RAG /ask: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý RAG: {str(e)}")

@app.post("/rag/ingest-data", status_code=202)
async def trigger_rag_ingestion(background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_rag_data)
    return {"message": "Quá trình ingest dữ liệu vào VectorStore đã được bắt đầu trong nền. Quá trình này có thể mất vài phút."}

@app.get("/news/latest", response_model=List[NewsArticle])
async def get_latest_news(limit: int = 20, page: int = 1, ticker: Optional[str] = None):
    offset = (page - 1) * limit
    news_table_name = "crypto_news"
    query_str = f"SELECT article_id, title, url, domain, published_at, source_info, related_currencies, sentiment_score, sentiment_label FROM {news_table_name} "

    conditions = []
    params = {}

    if ticker:
        ticker_code = ticker.split('-')[0].upper()
        conditions.append(f"related_currencies ILIKE :ticker_code_pattern")
        params["ticker_code_pattern"] = f"%{ticker_code}%"

    if conditions:
        query_str += " WHERE " + " AND ".join(conditions)

    query_str += " ORDER BY published_at DESC LIMIT :limit OFFSET :offset"
    params["limit"] = limit
    params["offset"] = offset

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query_str), params)
            keys = result.keys()
            news_data = [dict(zip(keys, row)) for row in result.fetchall()]

        return news_data 
    except Exception as e:
        print(f"Lỗi khi truy vấn tin tức: {e}")
        raise HTTPException(status_code=500, detail="Lỗi máy chủ khi lấy tin tức.")