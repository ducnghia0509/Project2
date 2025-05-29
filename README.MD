# Dự án Phân tích và Dự đoán Tiền Điện Tử (Project 2)

## Mô tả

Một nền tảng web theo dõi và phân tích dữ liệu cho nhiều loại tiền điện tử (crypto). Hệ thống cung cấp dữ liệu lịch sử, phân tích kỹ thuật cơ bản, phân tích tâm lý thị trường từ tin tức, và dự đoán giá đóng cửa trong tương lai (15, 30, 60 ngày) bằng mô hình LSTM. Đặc biệt, dự án tích hợp một chatbot AI (sử dụng Google Gemini và kỹ thuật RAG) để trả lời các câu hỏi về kiến thức crypto dựa trên cơ sở dữ liệu nội bộ.

## Tính năng chính

*   Thu thập và lưu trữ dữ liệu OHLCV lịch sử cho nhiều loại tiền điện tử.
*   Cập nhật dữ liệu giá hàng ngày.
*   Thu thập tin tức crypto từ CryptoPanic.
*   Phân tích tâm lý (Sentiment Analysis) cơ bản dựa trên tiêu đề tin tức.
*   Tính toán các chỉ báo Phân tích Kỹ thuật (TA) cơ bản (SMA, EMA, RSI, MACD, Bollinger Bands).
*   Huấn luyện mô hình dự đoán giá LSTM cho nhiều tickers và nhiều khoảng thời gian (15, 30, 60 ngày).
*   Cung cấp API (FastAPI) để truy cập dữ liệu, phân tích và dự đoán.
*   Giao diện Dashboard (Streamlit) trực quan để hiển thị biểu đồ, tin tức, dự đoán.
*   Chatbot RAG (Retrieval Augmented Generation) sử dụng Google Gemini để trả lời câu hỏi về crypto dựa trên kho kiến thức được cung cấp.

## Cài đặt và Chạy dự án

### 1. Prerequisites (Yêu cầu cài đặt trước)

*   Python (phiên bản 3.9 hoặc mới hơn được khuyến nghị) và `pip`
*   Git
*   PostgreSQL Server đang chạy

### 2. Clone Repository


git clone https://github.com/ducnghia0509/Project2.git
cd Project2

### 3. Tạo và Kích hoạt Môi trường ảo
#### Tạo môi trường ảo (ví dụ tên là 'venv')
python -m venv venv

#### Kích hoạt môi trường ảo
##### Trên Windows:
.\venv\Scripts\activate
##### Trên macOS/Linux:
source venv/bin/activate

### 4. Cài đặt Dependencies
pip install -r requirements.txt

### 5. Thiết lập Cơ sở dữ liệu PostgreSQL
Bạn cần có một database PostgreSQL đang chạy. Tạo một database và một user mới cho dự án này. Ví dụ sử dụng psql hoặc công cụ như pgAdmin:
-- Ví dụ lệnh SQL (chạy với quyền superuser hoặc user có quyền tạo DB/User)
CREATE DATABASE crypto_db;
CREATE USER your_db_user WITH PASSWORD 'your_db_password';
GRANT ALL PRIVILEGES ON DATABASE crypto_db TO your_db_user;
-- Có thể cần cấp thêm quyền cho schema public nếu user không phải là owner
-- GRANT ALL ON SCHEMA public TO your_db_user; -- (Tùy cấu hình PG của bạn)

### 6. Tạo file Biến Môi trường (.env)
Tạo một file tên là .env trong thư mục gốc của dự án và điền các thông tin sau:
#### Cấu hình Database
DB_HOST=localhost        # Hoặc IP/hostname của DB server nếu khác
DB_NAME=crypto_db        # Tên database bạn đã tạo ở bước 5
DB_USER=your_db_user     # Username bạn đã tạo ở bước 5
DB_PASSWORD=your_db_password # Password bạn đã đặt ở bước 5

#### API Keys (Cần thiết cho các dịch vụ tương ứng)
GOOGLE_API_KEY=your_google_ai_studio_api_key_here # Lấy từ Google AI Studio (cho Gemini)
CRYPTO_PANIC_API_KEY=your_crypto_panic_api_key_here # Lấy từ CryptoPanic (cho tin tức)

#### Cấu hình ngày bắt đầu/kết thúc cho lần tải dữ liệu đầu tiên (initial_load)
#### Định dạng: YYYY-MM-DD
StartDate=2020-01-01   # Ngày bắt đầu lấy dữ liệu lịch sử
EndDate=2024-01-01     # Ngày kết thúc lấy dữ liệu lịch sử (thường là một ngày trong quá khứ gần)
Dotenv
Lấy GOOGLE_API_KEY từ Google AI Studio.
Lấy CRYPTO_PANIC_API_KEY từ CryptoPanic Developers.
### 7. Chuẩn bị Dữ liệu và Mô hình
python data_processing/initial_load.py

#### 7.2. Cập nhật dữ liệu giá đến ngày hiện tại
python data_processing/update_daily.py

#### 7.3. Lấy tin tức mới nhất
python news_handling/news_fetcher.py

#### 7.4. Phân tích Sentiment cho tin tức (chạy sau khi lấy tin tức)
python analysis/sentiment_analyzer.py

#### 7.5. Chuẩn bị Cơ sở Tri thức cho RAG
python llm_rag/rag_handler.py ingest

#### 7.6. Huấn luyện Mô hình Dự đoán
python prediction/train_predict_models.py
echo "Model training finished."

### 8. Chạy Ứng dụng
Bạn cần chạy cả Backend API và Frontend Dashboard cùng lúc trong hai cửa sổ terminal riêng biệt (đảm bảo môi trường ảo đã được kích hoạt ở cả hai).
#### Terminal 1: Chạy Backend API (FastAPI)
uvicorn app.main_api:app --reload

API sẽ chạy tại http://127.0.0.1:8000 (hoặc IP của máy bạn nếu truy cập từ máy khác)
Giao diện Swagger UI tại http://127.0.0.1:8000/docs

#### Terminal 2: Chạy Frontend Dashboard (Streamlit)
streamlit run app/app_dashboard.py
 
Dashboard sẽ chạy tại http://localhost:8501

#### Terminal 3: Chạy lấy dữ liệu WebSocket từ sàn Binance
python realtime_collector.py

#### Terminal 4: Chạy tất cả các schedule (re-train, re-analysis, re-getnews,...)
python master_scheduler.py