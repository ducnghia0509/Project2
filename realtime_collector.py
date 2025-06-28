# realtime_collector.py
import asyncio
import websockets
import json
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from core.config import DB_CONN_STR, DEFAULT_CRYPTO_TICKERS

# Cấu hình
SYMBOLS_TO_TRACK = [ticker.replace("-", "").upper() for ticker in DEFAULT_CRYPTO_TICKERS if "USD" in ticker or "USDT" in ticker] 
print(f"[RealtimeCollector DEBUG] DEFAULT_CRYPTO_TICKERS: {DEFAULT_CRYPTO_TICKERS}")
print(f"[RealtimeCollector DEBUG] SYMBOLS_TO_TRACK_FROM_CONFIG after processing: {SYMBOLS_TO_TRACK}")
# test
SYMBOLS_TO_TRACK = ["BTCUSDT", "ETHUSDT", "BNBUSDT"] 
print(f"[RealtimeCollector DEBUG] SYMBOLS_TO_TRACK (hardcoded for test): {SYMBOLS_TO_TRACK}")
BINANCE_WS_BASE_URL = "wss://stream.binance.com:9443/ws/"

# Kết nối CSDL
try:
    engine = create_engine(DB_CONN_STR)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    print("[RealtimeCollector] Connected to database.")
except Exception as e:
    print(f"[RealtimeCollector] Error connecting to database: {e}")
    engine = None
    SessionLocal = None

def get_db():
    if not SessionLocal:
        return None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def save_tick_to_db(symbol, price, timestamp_ms, source="binance_ws"):
    if not engine:
        print("[RealtimeCollector CRITICAL] DB not connected in save_tick_to_db. Cannot save tick.") 
        return

    db_session = next(get_db(), None)
    if not db_session:
        print("[RealtimeCollector CRITICAL] Could not get DB session in save_tick_to_db.") 
        return

    try:
        app_symbol = symbol.replace("USDT", "-USD").replace("BUSD","-USD")
        if "USD" not in app_symbol and len(app_symbol) > 3 and app_symbol.isalnum():
             app_symbol = app_symbol[:3] + "-" + app_symbol[3:]

        dt_object_utc = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)

        print(f"[RealtimeCollector DEBUG SaveAttempt] AppSymbol='{app_symbol}', OriginalBinanceSymbol='{symbol}', Price='{price}', TimestampUTC='{dt_object_utc}', Source='{source}'")

        query = text("""
            INSERT INTO realtime_price_ticks (symbol, price, timestamp, source)
            VALUES (:symbol, :price, :timestamp, :source)
        """)
        db_session.execute(query, {
            "symbol": app_symbol, 
            "price": float(price),
            "timestamp": dt_object_utc, 
            "source": source
        })
        db_session.commit()
        print(f"[RealtimeCollector INFO SaveSuccess] Saved: AppSymbol='{app_symbol}', Price='{price}'") 

    except Exception as e:
        db_session.rollback()
        # Log lỗi chi tiết, bao gồm cả dữ liệu gây lỗi
        print(f"[RealtimeCollector ERROR SaveFail] Error saving tick. OriginalSymbol='{symbol}', AppSymbol='{app_symbol}', Price='{price}', TimestampMs='{timestamp_ms}'. Error: {e}")
        # import traceback # Để xem full traceback nếu cần
        # print(traceback.format_exc())
    finally:
        if db_session:
            db_session.close()


async def binance_agg_trade_stream_listener(symbols):
    # Tạo URL stream cho nhiều symbol: /ws/btcusdt@aggTrade/ethusdt@aggTrade/...
    streams = "/".join([f"{s.lower()}@aggTrade" for s in symbols])
    uri = f"{BINANCE_WS_BASE_URL}{streams}"
    print(f"[RealtimeCollector] Connecting to Binance WebSocket: {uri}")

    while True: # Vòng lặp để tự động kết nối lại nếu mất kết nối
        try:
            async with websockets.connect(uri) as websocket:
                print(f"[RealtimeCollector] Connected to {', '.join(symbols)} stream.")
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'e' in data and data['e'] == 'aggTrade': 
                            stream_symbol = data['s']
                            price = data['p']
                            timestamp_ms = data['T'] 
                            
                            asyncio.create_task(save_tick_to_db(stream_symbol, price, timestamp_ms))

                    except json.JSONDecodeError:
                        print(f"[RealtimeCollector] Invalid JSON received: {message}")
                    except KeyError as e:
                        print(f"[RealtimeCollector] Missing key in data: {e} - Data: {data}")
                    except Exception as e:
                        print(f"[RealtimeCollector] Error processing message: {e} - Message: {message}")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[RealtimeCollector] Connection closed: {e}. Reconnecting in 5 seconds...")
        except Exception as e:
            print(f"[RealtimeCollector] WebSocket error: {e}. Reconnecting in 5 seconds...")
        await asyncio.sleep(5) # Đợi 5 giây trước khi thử kết nối lại


async def main():
    if not SYMBOLS_TO_TRACK:
        print("[RealtimeCollector] No symbols configured to track.")
        return
    
    # Tạo bảng nếu chưa tồn tại (chạy 1 lần khi khởi động)
    if engine:
        try:
            with engine.connect() as connection:
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS realtime_price_ticks (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        source VARCHAR(50)
                    );"""))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_realtime_ticks_symbol_timestamp ON realtime_price_ticks (symbol, timestamp DESC);"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_realtime_ticks_timestamp ON realtime_price_ticks (timestamp);"))
                connection.commit()
                print("[RealtimeCollector] Table 'realtime_price_ticks' checked/created.")
        except Exception as e:
            print(f"[RealtimeCollector] Error ensuring table exists: {e}")


    await binance_agg_trade_stream_listener(SYMBOLS_TO_TRACK)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[RealtimeCollector] Collector stopped by user.")