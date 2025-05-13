# data_processing/update_daily.py
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text # Để thực thi SQL an toàn hơn
from sqlalchemy.exc import ProgrammingError # Để bắt lỗi khi bảng không tồn tại

# Import các thành phần cần thiết từ các file khác
from data_processing.data_ingestion import get_crypto_data
from core.db_connect import engine

tickers = [
    # Dài hạn
    "BTC-USD", "ETH-USD",
    # Ngắn hạn
    "SOL-USD", "BNB-USD", "XRP-USD",
    # Tiềm năng (AI, Layer 2)
    "RNDR-USD", "FET-USD", "AGIX-USD", "OCEAN-USD", 
    "ARB-USD", "OP-USD", "MATIC-USD", "STRK-USD",  
    # DeFi
    "UNI-USD", "AAVE-USD", "MKR-USD", "CRV-USD", "SNX-USD",
    # Stablecoins
    "USDT-USD", "USDC-USD",
    # Khác
    "DOGE-USD", "ADA-USD", "TRX-USD",
]

# Ngày bắt đầu mặc định 
DEFAULT_START_DATE = "2010-01-01"

print("--- Bắt đầu quá trình cập nhật dữ liệu ---")

successful_updates = []
no_new_data = []
failed_updates = []
created_tables = []

for ticker in tickers:
    data = None
    table_name = ""
    try:
        # 1. Xác định tên bảng
        base_ticker = ticker.split('-')[0].lower().replace('.', '_')
        table_name = f"{base_ticker}_price"

        print(f"\n--- Xử lý ticker: {ticker} (Bảng: {table_name}) ---")

        # 2. Tìm ngày cuối cùng trong DB
        latest_date_in_db = None
        try:
            with engine.connect() as connection:
                # Dùng text() để an toàn hơn, dù tên bảng đang được tạo programmatically
                # Lưu ý: Tên bảng không thể truyền qua placeholder chuẩn, cần f-string cẩn thận
                query = text(f"SELECT MAX(date) FROM {table_name}")
                result = connection.execute(query)
                latest_date_in_db = result.scalar() # Lấy giá trị đơn (ngày hoặc None)
                if latest_date_in_db:
                     # Chuyển đổi sang kiểu date của Python nếu cần (thường SQLAlchemy tự xử lý)
                     if isinstance(latest_date_in_db, str):
                         latest_date_in_db = pd.to_datetime(latest_date_in_db).date()
                     elif hasattr(latest_date_in_db, 'date'): # Nếu là datetime object
                         latest_date_in_db = latest_date_in_db.date()
                     print(f"Ngày mới nhất trong DB cho {table_name}: {latest_date_in_db}")
                else:
                     print(f"Không tìm thấy dữ liệu cũ trong bảng {table_name}.")

        except ProgrammingError as pe:
            # Lỗi thường gặp nếu bảng không tồn tại (PostgreSQL: UndefinedTable)
            if "UndefinedTable" in str(pe) or "does not exist" in str(pe):
                print(f"Bảng {table_name} chưa tồn tại. Sẽ tạo mới.")
                created_tables.append(table_name)
            else:
                # Lỗi SQL khác
                print(f"Lỗi SQL khi kiểm tra {table_name}: {pe}")
                failed_updates.append(f"{ticker} (SQL Error Check)")
                continue # Bỏ qua ticker này
        except Exception as e:
             print(f"Lỗi không xác định khi kiểm tra DB cho {table_name}: {e}")
             failed_updates.append(f"{ticker} (DB Check Error)")
             continue # Bỏ qua ticker này


        # 3. Xác định ngày bắt đầu và kết thúc để lấy dữ liệu mới
        if latest_date_in_db:
            # Lấy dữ liệu từ ngày *sau* ngày cuối cùng trong DB
            start_date_dt = latest_date_in_db + timedelta(days=1)
            start_date_str = start_date_dt.strftime('%Y-%m-%d')
        else:
            # Nếu bảng trống hoặc mới, lấy từ ngày mặc định
            start_date_str = DEFAULT_START_DATE

        # Luôn lấy đến ngày hiện tại
        end_date_str = date.today().strftime('%Y-%m-%d')

        # Kiểm tra logic: Nếu ngày bắt đầu >= ngày kết thúc thì không cần lấy
        if start_date_str >= end_date_str:
             print(f"Dữ liệu cho {ticker} đã được cập nhật đến ngày hôm qua ({latest_date_in_db}). Không cần lấy dữ liệu mới.")
             no_new_data.append(ticker)
             continue # Chuyển sang ticker tiếp theo


        print(f"Lấy dữ liệu mới cho {ticker} từ {start_date_str} đến {end_date_str}")

        # 4. Lấy dữ liệu mới
        data = get_crypto_data(ticker, start_date=start_date_str, end_date=end_date_str)

        # 5. Kiểm tra và chèn (Append) dữ liệu mới
        if data is not None and not data.empty:
             # Loại bỏ các hàng có ngày đã tồn tại (phòng trường hợp API trả về trùng lặp)
             if latest_date_in_db and 'date' in data.columns:
                 # Đảm bảo cột date trong data là kiểu date để so sánh
                 data['date'] = pd.to_datetime(data['date']).dt.date
                 original_count = len(data)
                 data = data[data['date'] > latest_date_in_db]
                 if len(data) < original_count:
                     print(f"Đã loại bỏ {original_count - len(data)} hàng trùng ngày với dữ liệu hiện có.")

             if data.empty:
                 print(f"Không có dữ liệu *mới* thực sự cho {ticker} sau khi lọc trùng lặp.")
                 no_new_data.append(ticker)
                 continue

             print(f"-> Chuẩn bị nối {len(data)} hàng mới vào bảng '{table_name}'...")
             try:
                 # Sử dụng transaction để đảm bảo an toàn cho việc append
                 with engine.begin() as connection:
                     data.to_sql(
                         name=table_name,
                         con=connection,
                         if_exists='append', # Nối dữ liệu vào bảng hiện có
                         index=False,        # Không ghi index
                         method='multi',
                        chunksize=300
                     )
                 print(f"==> Nối dữ liệu mới cho {ticker} vào '{table_name}' thành công.")
                 successful_updates.append(ticker)

             except Exception as db_err:
                 print(f"*** LỖI DATABASE khi nối dữ liệu {ticker} vào '{table_name}': {db_err}")
                 failed_updates.append(f"{ticker} (Append Error)")

        elif data is None:
             print(f"Không thể tải dữ liệu mới cho {ticker}.")
             # Lỗi đã được in ra từ get_crypto_data
             failed_updates.append(f"{ticker} (Fetch Error)")
        else: # data is empty
             print(f"Không có dữ liệu mới nào được trả về từ API cho {ticker} trong khoảng thời gian yêu cầu.")
             no_new_data.append(ticker)

    except Exception as loop_err:
        print(f"*** LỖI NGOÀI DỰ KIẾN trong vòng lặp cho {ticker}: {loop_err}")
        failed_updates.append(f"{ticker} (Loop Error)")

print("\n--- === KẾT QUẢ CẬP NHẬT === ---")
print(f"Tổng số mã xử lý: {len(tickers)}")
print(f"Tạo bảng mới ({len(created_tables)}): {', '.join(created_tables) if created_tables else 'Không có'}")
print(f"Cập nhật thành công ({len(successful_updates)}): {', '.join(successful_updates) if successful_updates else 'Không có'}")
print(f"Không có dữ liệu mới/Đã cập nhật ({len(no_new_data)}): {', '.join(no_new_data) if no_new_data else 'Không có'}")
print(f"Lỗi ({len(failed_updates)}): {', '.join(failed_updates) if failed_updates else 'Không có'}")
print("--- Quá trình cập nhật hoàn tất ---")
