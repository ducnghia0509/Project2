# analysis/sentiment_analyzer.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from core.db_connect import engine 
from sqlalchemy import text, Column, Float, String, DateTime, Integer 
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

# Khởi tạo VADER analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text_content):
    """
    Phân tích tâm lý của một đoạn văn bản sử dụng VADER.
    Trả về điểm 'compound' (từ -1 đến 1).
    """
    if not text_content or not isinstance(text_content, str):
        return None 
    
    vs = analyzer.polarity_scores(text_content)
    return vs['compound'] 

Base = declarative_base()
Session = sessionmaker(bind=engine)

class CryptoNews(Base):
    __tablename__ = 'crypto_news'
    article_id = Column(String, primary_key=True)
    title = Column(String)
    url = Column(String)
    domain = Column(String)
    published_at = Column(DateTime)
    source_info = Column(String)
    related_currencies = Column(String)

    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String, nullable=True)

def column_exists(connection, table_name, column_name, schema='public'):
    query = text("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table_name
              AND column_name = :column_name
        );
    """)
    result = connection.execute(query, {"schema": schema, "table_name": table_name, "column_name": column_name})
    return result.scalar_one()

def add_sentiment_columns_to_db(table_name="crypto_news"):
    """Thêm cột sentiment_score và sentiment_label vào bảng nếu chưa có."""
    with engine.connect() as connection:
        trans = connection.begin() # Bắt đầu transaction
        try:
            if not column_exists(connection, table_name, "sentiment_score"):
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN sentiment_score FLOAT;"))
                print(f"Đã thêm cột 'sentiment_score' vào bảng {table_name}.") # Sẽ chạy trong môi trường UTF-8
            else:
                print(f"Cột 'sentiment_score' đã tồn tại trong bảng {table_name}.")

            if not column_exists(connection, table_name, "sentiment_label"):
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN sentiment_label VARCHAR(50);"))
                print(f"Đã thêm cột 'sentiment_label' vào bảng {table_name}.")
            else:
                print(f"Cột 'sentiment_label' đã tồn tại trong bảng {table_name}.")
            
            trans.commit() # Commit transaction
        except Exception as e:
            trans.rollback() # Rollback nếu có lỗi
            print(f"Lỗi khi kiểm tra/thêm cột sentiment: {e}")


def process_news_sentiment(limit=100, table_name="crypto_news"):
    """
    Lấy các tin tức chưa có điểm sentiment, phân tích và cập nhật vào DB.
    """
    session = Session()
    try:
        news_to_process = session.query(CryptoNews).filter(CryptoNews.sentiment_score == None).limit(limit).all()
        
        if not news_to_process:
            print("Không có tin tức mới nào để phân tích sentiment.")
            return 0

        print(f"Tìm thấy {len(news_to_process)} tin tức để phân tích sentiment.")
        updated_count = 0
        for news_item in news_to_process:
            text_to_analyze = news_item.title 
            
            score = analyze_sentiment_vader(text_to_analyze)
            
            if score is not None:
                news_item.sentiment_score = score
                if score >= 0.05:
                    news_item.sentiment_label = 'positive'
                elif score <= -0.05:
                    news_item.sentiment_label = 'negative'
                else:
                    news_item.sentiment_label = 'neutral'
                updated_count += 1
        
        session.commit()
        print(f"Đã cập nhật sentiment cho {updated_count} tin tức.")
        return updated_count
    except Exception as e:
        session.rollback()
        print(f"Lỗi khi xử lý sentiment tin tức: {e}")
        return -1
    finally:
        session.close()

if __name__ == "__main__":
    add_sentiment_columns_to_db()
    
    # Xử lý sentiment
    processed_count = process_news_sentiment(limit=50) # Xử lý 50 tin một lần
    if processed_count > 0:
        print(f"Đã xử lý sentiment cho {processed_count} tin tức.")
    elif processed_count == 0:
        print("Không có tin tức nào được xử lý sentiment (có thể đã xử lý hết hoặc không có tin mới).")
    else:
        print("Có lỗi xảy ra trong quá trình xử lý sentiment.")
