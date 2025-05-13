# data_processing/data_ingresion.py
import yfinance as yf
import pandas as pd 


def get_crypto_data(ticker, start_date, end_date):
    print(f"--- Đang tải dữ liệu cho: {ticker} ---")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

        if data.empty:
            print(f"Cảnh báo: Không tải được dữ liệu cho {ticker}. Mã không tồn tại hoặc không có dữ liệu trong khoảng thời gian này.")
            return None 

        data = data.reset_index()
        print(f"Tải dữ liệu cho {ticker} hoàn tất. Số hàng: {len(data)}")

        new_columns = []
        for col_tuple in data.columns:

            core_name = col_tuple[0]

            cleaned_name = str(core_name).lower()
            cleaned_name = cleaned_name.replace(' ', '_').replace('-', '_')


            new_columns.append(cleaned_name)

        data.columns = new_columns

        if 'date' not in data.columns and 'Date' in data.columns:
             data = data.rename(columns={'Date': 'date'})
        elif 'date' not in data.columns:
             print(f"Cảnh báo: Không tìm thấy cột 'Date' hoặc 'date' trong dữ liệu cho {ticker}. Kiểm tra lại kết quả từ yfinance.")

        return data

    except Exception as e:
        print(f"Lỗi khi tải dữ liệu cho {ticker}: {e}")
        return None