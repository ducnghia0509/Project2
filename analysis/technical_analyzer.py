# analysis/technical_analyzer.py
import pandas as pd
import pandas_ta as ta 
from core.db_connect import engine 
from sqlalchemy import text
from core.config import get_db_table_name

def calculate_basic_indicators(df_price):
    """
    Tính toán các chỉ báo kỹ thuật cơ bản.
    df_price phải có các cột 'open', 'high', 'low', 'close', 'volume' (viết thường).
    """
    if df_price.empty:
        return df_price

    df_with_ta = df_price.copy()

    df_with_ta.columns = [col.lower() for col in df_with_ta.columns]
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_with_ta.columns for col in required_cols):
        print(f"Lỗi: Dữ liệu thiếu các cột cần thiết: {required_cols}")
        return df_price 
    try:
        df_with_ta.ta.sma(length=20, append=True)
        df_with_ta.ta.sma(length=50, append=True)
        df_with_ta.ta.ema(length=20, append=True)
        df_with_ta.ta.rsi(length=14, append=True)
        df_with_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
        df_with_ta.ta.bbands(length=20, std=2, append=True)

        print(f"Đã tính toán các chỉ báo TA. Các cột mới: {df_with_ta.columns.difference(df_price.columns).tolist()}")

    except Exception as e:
        print(f"Lỗi khi tính toán chỉ báo TA: {e}")
        return df_price 

    return df_with_ta

def load_price_data_and_calculate_ta(ticker, feature_cols=['open', 'high', 'low', 'close', 'volume']):
    """Tải dữ liệu giá từ DB và tính toán TA."""
    table_name = get_db_table_name(ticker)
    print(f"[TA Service] Đang tải dữ liệu từ bảng DB: {table_name} cho TA...")
    try:
        sql_feature_cols = ', '.join([f'"{col}"' for col in feature_cols])
        query = f'SELECT "date", {sql_feature_cols} FROM {table_name} ORDER BY "date" ASC'
        df = pd.read_sql(query, engine, parse_dates=['date'])

        if df.empty:
            print(f"[TA Service] Lỗi: Không tìm thấy dữ liệu trong bảng '{table_name}'.")
            return None

        df_with_ta = calculate_basic_indicators(df)
        return df_with_ta

    except Exception as e:
        print(f"[TA Service] Lỗi khi tải dữ liệu hoặc tính TA cho {ticker}: {e}")
        return None

if __name__ == '__main__':
    target_ticker = "BTC-USD"
    df_btc_ta = load_price_data_and_calculate_ta(target_ticker)

    if df_btc_ta is not None:
        print(f"\nDữ liệu {target_ticker} với các chỉ báo TA (5 hàng cuối):")
        print(df_btc_ta.tail())
        print("\nCác cột có trong DataFrame:")
        print(df_btc_ta.columns.tolist())
    else:
        print(f"Không thể xử lý dữ liệu cho {target_ticker}")