# new_handling/news_fetcher.py
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from core.db_connect import engine

load_dotenv()

from core.config import CRYPTO_PANIC_API_KEY
NEWS_API_BASE_URL = "https://cryptopanic.com/api/v1/posts/"

if not CRYPTO_PANIC_API_KEY:
    print("CẢNH BÁO: CRYPTO_PANIC_API_KEY không được tìm thấy trong biến môi trường.")

def get_crypto_news(currencies=None, kind='news', region='en', public='true', page=1):
    """
    Lấy tin tức từ CryptoPanic API.
    currencies: list of currency symbols (e.g., ['BTC', 'ETH']) or None for general news.
    kind: 'news' or 'media'.
    region: 'en', 'de', 'nl', 'es', 'fr', 'it', 'pt', 'ru'.
    public: 'true' for public API.
    page: số trang để lấy.
    """
    if not CRYPTO_PANIC_API_KEY:
        return None

    params = {
        'auth_token': CRYPTO_PANIC_API_KEY,
        'public': public,
        'kind': kind,
        'regions': region,
        'page': page
    }
    if currencies:
        params['currencies'] = ",".join(currencies) 

    try:
        response = requests.get(NEWS_API_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('results', []) 
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gọi CryptoPanic API: {e}")
        return None
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý API response: {e}")
        return None

def save_news_to_db(news_articles, table_name="crypto_news"):
    """Lưu danh sách các bài viết tin tức vào CSDL."""
    if not news_articles:
        print("Không có tin tức nào để lưu.")
        return 0

    df_news = pd.DataFrame(news_articles)

    columns_to_keep = {
        'id': 'article_id',
        'title': 'title',
        'url': 'url',
        'domain': 'domain',
        'created_at': 'published_at',
        'source': 'source_info',
        'currencies': 'related_currencies',
    }
    
    df_selected = pd.DataFrame()
    for original_col, new_col_name in columns_to_keep.items():
        if original_col in df_news.columns:
            df_selected[new_col_name] = df_news[original_col]
        else:
            print(f"Cảnh báo: Cột '{original_col}' không tìm thấy trong dữ liệu API, sẽ bỏ qua.")

    if 'article_id' not in df_selected.columns:
        print("LỖI: Không có cột 'article_id' sau khi chọn lọc. Không thể tiếp tục.")
        return -1 
    df_news = df_selected 

    # Xử lý cột 'published_at' sang datetime
    if 'published_at' in df_news.columns:
        try:
            df_news['published_at'] = pd.to_datetime(df_news['published_at'], errors='coerce')
            # Kiểm tra xem có giá trị NaN nào không
            if df_news['published_at'].isnull().any():
                print("Cảnh báo: Một số giá trị 'published_at' không thể chuyển đổi và đã được đặt thành NaT.")

        except Exception as e:
            print(f"Lỗi khi chuyển đổi 'published_at' sang datetime: {e}")

    else:
        print("Cảnh báo: Không có cột 'published_at' để xử lý.")

    if 'source_info' in df_news.columns:
        df_news['source_info'] = df_news['source_info'].apply(
            lambda x: x.get('title', 'N/A') if isinstance(x, dict) else ('N/A' if pd.isna(x) else str(x))
        )
    else:
        print("Cảnh báo: Không có cột 'source_info'.")

    if 'related_currencies' in df_news.columns:
        def extract_currency_codes(currency_list):
            if isinstance(currency_list, list):
                return ", ".join(sorted(list(set(c.get('code', '') for c in currency_list if isinstance(c, dict) and c.get('code')))))
            return None # Hoặc '' nếu bạn muốn chuỗi rỗng
        df_news['related_currencies'] = df_news['related_currencies'].apply(extract_currency_codes)
    else:
        print("Cảnh báo: Không có cột 'related_currencies'.")

    if 'article_id' in df_news.columns:
        df_news['article_id'] = df_news['article_id'].astype(str) 
        try:
            with engine.connect() as connection:
                existing_ids_query = f"SELECT article_id FROM {table_name}"
                try:
                    existing_ids_df = pd.read_sql(existing_ids_query, connection)
                    existing_ids = set(existing_ids_df['article_id'].astype(str))
                except Exception as e_read:
                    if "UndefinedTable" in str(e_read) or "does not exist" in str(e_read).lower() or "relation" in str(e_read).lower() and "does not exist" in str(e_read).lower():
                        print(f"Bảng {table_name} chưa tồn tại. Sẽ tạo mới, không cần lọc ID.")
                        existing_ids = set()
                    else:
                        raise 
            
            original_count = len(df_news)
            df_news = df_news[~df_news['article_id'].isin(existing_ids)]
            new_articles_count = len(df_news)
            if original_count > new_articles_count:
                 print(f"Đã loại bỏ {original_count - new_articles_count} bài viết đã tồn tại trong DB.")

            if new_articles_count == 0:
                print("Không có tin tức mới nào để thêm vào DB sau khi lọc trùng lặp.")
                return 0

        except SQLAlchemyError as e:
            print(f"Lỗi DB khi kiểm tra article_id đã tồn tại: {e}")
        except Exception as e_outer:
            print(f"Lỗi không xác định khi kiểm tra article_id: {e_outer}")


    if df_news.empty:
        print("Không có tin tức nào để lưu sau tất cả các bước xử lý.")
        return 0
        
    # Lưu vào CSDL
    try:
        with engine.begin() as connection:
            df_news.to_sql(table_name, connection, if_exists='append', index=False, method='multi', chunksize=100) 
        print(f"Đã lưu thành công {len(df_news)} tin tức mới vào bảng '{table_name}'.")
        return len(df_news)
    except SQLAlchemyError as e:
        print(f"Lỗi SQLAlchemy khi lưu tin tức: {e}")
        return -1
    except Exception as e:
        print(f"Lỗi không xác định khi lưu tin tức: {e}")
        return -1

if __name__ == "__main__":
    print("--- Lấy tin tức Crypto chung ---")
    general_news = get_crypto_news(page=1)
    if general_news:
        print(f"Lấy được {len(general_news)} tin tức chung.")
        save_news_to_db(general_news)
    else:
        print("Không lấy được tin tức chung.")

    print("\n--- Lấy tin tức cho BTC và ETH ---")
    btc_eth_news = get_crypto_news(currencies=['BTC', 'ETH'], page=1)
    if btc_eth_news:
        print(f"Lấy được {len(btc_eth_news)} tin tức cho BTC/ETH.")
        save_news_to_db(btc_eth_news)
    else:
        print("Không lấy được tin tức cho BTC/ETH.")