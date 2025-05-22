# data_processing/update_daily.py
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text 
from sqlalchemy.exc import ProgrammingError 
import logging 

try:
    from .data_ingestion import get_crypto_data 
except ImportError:
    from data_processing.data_ingestion import get_crypto_data

from core.db_connect import engine
from core.config import (
    DEFAULT_CRYPTO_TICKERS,
    UPDATE_DEFAULT_START_DATE, # Sử dụng hằng số này từ config
    get_db_table_name # Sử dụng hàm helper từ config
)

# --- Cấu hình Logging ---
log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


def main_update_logic():
    logger.info("--- Bắt đầu quá trình cập nhật dữ liệu giá ---")

    # Lấy danh sách tickers và ngày bắt đầu mặc định từ config
    tickers_to_update = DEFAULT_CRYPTO_TICKERS
    default_start_date_for_new_table = UPDATE_DEFAULT_START_DATE

    successful_updates = []
    no_new_data = []
    failed_updates = []
    created_tables = []

    for ticker in tickers_to_update:
        data = None
        table_name = "" # Reset cho mỗi ticker
        try:
            # 1. Xác định tên bảng sử dụng hàm từ config
            table_name = get_db_table_name(ticker)

            logger.info(f"--- Xử lý ticker: {ticker} (Bảng: {table_name}) ---")

            # 2. Tìm ngày cuối cùng trong DB
            latest_date_in_db = None
            try:
                with engine.connect() as connection:
                    query = text(f"SELECT MAX(date) FROM {table_name}")
                    result = connection.execute(query)
                    latest_date_in_db = result.scalar()
                    if latest_date_in_db:
                        if isinstance(latest_date_in_db, str):
                            latest_date_in_db = pd.to_datetime(latest_date_in_db).date()
                        elif hasattr(latest_date_in_db, 'date'):
                            latest_date_in_db = latest_date_in_db.date()
                        logger.info(f"Ngày mới nhất trong DB cho {table_name}: {latest_date_in_db}")
                    else:
                        logger.info(f"Không tìm thấy dữ liệu cũ trong bảng {table_name}.")

            except ProgrammingError as pe:
                if "UndefinedTable" in str(pe).lower() or "does not exist" in str(pe).lower() or "relation" in str(pe).lower() and "does not exist" in str(pe).lower() : # Thêm check cho PostgreSQL
                    logger.info(f"Bảng {table_name} chưa tồn tại. Sẽ tạo mới.")
                    created_tables.append(table_name)
                else:
                    logger.error(f"Lỗi SQL khi kiểm tra {table_name}: {pe}")
                    failed_updates.append(f"{ticker} (SQL Error Check)")
                    continue
            except Exception as e:
                logger.error(f"Lỗi không xác định khi kiểm tra DB cho {table_name}: {e}")
                failed_updates.append(f"{ticker} (DB Check Error)")
                continue

            # 3. Xác định ngày bắt đầu và kết thúc để lấy dữ liệu mới
            if latest_date_in_db:
                start_date_dt = latest_date_in_db + timedelta(days=1)
                start_date_str = start_date_dt.strftime('%Y-%m-%d')
            else:
                start_date_str = default_start_date_for_new_table

            end_date_str = date.today().strftime('%Y-%m-%d')

            if start_date_str > end_date_str: # Sửa: > thay vì >= vì end_date_str của yfinance là exclusive
                logger.info(f"Dữ liệu cho {ticker} đã được cập nhật đến ngày hôm qua ({latest_date_in_db}). Không cần lấy dữ liệu mới.")
                no_new_data.append(ticker)
                continue

            if pd.to_datetime(start_date_str).date() == date.today():
                logger.info(f"Ngày bắt đầu ({start_date_str}) là ngày hôm nay. Không có dữ liệu mới để lấy cho {ticker}.")
                no_new_data.append(ticker)
                continue


            logger.info(f"Lấy dữ liệu mới cho {ticker} từ {start_date_str} đến {end_date_str}")

            # 4. Lấy dữ liệu mới
            data = get_crypto_data(ticker, start_date=start_date_str, end_date=end_date_str)

            # 5. Kiểm tra và chèn (Append) dữ liệu mới
            if data is not None and not data.empty:
                if latest_date_in_db and 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date']).dt.date
                    original_count = len(data)
                    # Chỉ giữ lại những ngày LỚN HƠN ngày cuối cùng trong DB
                    data = data[data['date'] > latest_date_in_db].copy() # Thêm .copy() để tránh SettingWithCopyWarning
                    if len(data) < original_count:
                        logger.info(f"Đã loại bỏ {original_count - len(data)} hàng trùng ngày hoặc cũ hơn với dữ liệu hiện có.")

                if data.empty:
                    logger.info(f"Không có dữ liệu *mới* thực sự cho {ticker} sau khi lọc trùng lặp.")
                    no_new_data.append(ticker)
                    continue

                logger.info(f"-> Chuẩn bị nối {len(data)} hàng mới vào bảng '{table_name}'...")
                try:
                    with engine.begin() as connection:
                        data.to_sql(
                            name=table_name,
                            con=connection,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=1000 # Tăng chunksize có thể nhanh hơn cho insert lớn
                        )
                    logger.info(f"==> Nối dữ liệu mới cho {ticker} vào '{table_name}' thành công.")
                    successful_updates.append(ticker)

                except Exception as db_err:
                    logger.error(f"*** LỖI DATABASE khi nối dữ liệu {ticker} vào '{table_name}': {db_err}")
                    failed_updates.append(f"{ticker} (Append Error)")

            elif data is None:
                logger.warning(f"Không thể tải dữ liệu mới cho {ticker} (API không trả về dữ liệu).")
                failed_updates.append(f"{ticker} (Fetch Error - No Data from API)")
            else: # data is empty
                logger.info(f"Không có dữ liệu mới nào được trả về từ API cho {ticker} trong khoảng [{start_date_str} - {end_date_str}].")
                no_new_data.append(ticker)

        except Exception as loop_err:
            logger.error(f"*** LỖI NGOÀI DỰ KIẾN trong vòng lặp cho {ticker}: {loop_err}", exc_info=True)
            failed_updates.append(f"{ticker} (Loop Error)")

    logger.info("\n--- === KẾT QUẢ CẬP NHẬT DỮ LIỆU GIÁ === ---")
    logger.info(f"Tổng số mã xử lý: {len(tickers_to_update)}")
    logger.info(f"Tạo bảng mới ({len(created_tables)}): {', '.join(created_tables) if created_tables else 'Không có'}")
    logger.info(f"Cập nhật thành công ({len(successful_updates)}): {', '.join(successful_updates) if successful_updates else 'Không có'}")
    logger.info(f"Không có dữ liệu mới/Đã cập nhật ({len(no_new_data)}): {', '.join(no_new_data) if no_new_data else 'Không có'}")
    logger.info(f"Lỗi ({len(failed_updates)}): {', '.join(failed_updates) if failed_updates else 'Không có'}")
    logger.info("--- Quá trình cập nhật dữ liệu giá hoàn tất ---")

if __name__ == "__main__":
    main_update_logic()