# data_processing/initial_load.py
from data_processing.data_ingestion import get_crypto_data
from core.db_connect import engine
import os
from dotenv import load_dotenv
load_dotenv()

start_date = os.getenv("StartDate")
end_date = os.getenv("EndDate")
# --- Danh sách các mã ticker cần lấy ---
tickers = [
    # Dài hạn
    "BTC-USD", "ETH-USD",
    # Ngắn hạn
    "SOL-USD", "BNB-USD", "XRP-USD",
    # Tiềm năng (AI, Layer 2)
    "RNDR-USD", "FET-USD", "AGIX-USD", "OCEAN-USD", # AI
    "ARB-USD", "OP-USD", "MATIC-USD", "STRK-USD",  # Layer 2
    # DeFi
    "UNI-USD", "AAVE-USD", "MKR-USD", "CRV-USD", "SNX-USD",
    # Stablecoins (có thể dùng để theo dõi)
    "USDT-USD", "USDC-USD",
    # Khác
    "DOGE-USD", "ADA-USD", "TRX-USD",
    # "DOT-USD", "LINK-USD" # Ví dụ thêm nếu cần
]

print(f"Bắt đầu quá trình lấy và chèn dữ liệu cho {len(tickers)} mã...")
print(f"Khoảng thời gian: {start_date} đến {end_date}")

successful_inserts = []
failed_fetches = []
failed_inserts = []

for ticker in tickers:
    data = None 
    table_name = "" 

    try:
        data = get_crypto_data(ticker, start_date=start_date, end_date=end_date)

        if data is not None and not data.empty:
            base_ticker = ticker.split('-')[0].lower().replace('.', '_')
            table_name = f"{base_ticker}_price"

            print(f"-> Chuẩn bị chèn {len(data)} hàng vào bảng '{table_name}'...")

            try:
                with engine.begin() as connection: # Sử dụng transaction cho mỗi bảng
                    data.to_sql(
                        name=table_name,
                        con=connection,
                        if_exists="replace",  
                        index=False,          
                        method='multi',       
                        chunksize=300       
                    )
                print(f"==> Chèn dữ liệu cho {ticker} vào bảng '{table_name}' thành công.")
                successful_inserts.append(ticker)

            except Exception as db_err:
                print(f"*** LỖI DATABASE khi chèn {ticker} vào '{table_name}': {db_err}")
                failed_inserts.append(f"{ticker} ({table_name})")

        elif data is None:
            failed_fetches.append(ticker)
        else:
             print(f"--- Bỏ qua {ticker}: DataFrame rỗng sau khi tải.")
             failed_fetches.append(ticker)

    except Exception as loop_err:
        print(f"*** LỖI NGOÀI DỰ KIẾN trong vòng lặp cho {ticker}: {loop_err}")
        failed_inserts.append(f"{ticker} (Lỗi vòng lặp)")

print("\n--- === KẾT QUẢ === ---")
print(f"Tổng số mã xử lý: {len(tickers)}")
print(f"Thành công ({len(successful_inserts)}): {', '.join(successful_inserts) if successful_inserts else 'Không có'}")
print(f"Không tải được/Dữ liệu rỗng ({len(failed_fetches)}): {', '.join(failed_fetches) if failed_fetches else 'Không có'}")
print(f"Lỗi chèn Database/Vòng lặp ({len(failed_inserts)}): {', '.join(failed_inserts) if failed_inserts else 'Không có'}")
print("--- Quá trình hoàn tất ---")
