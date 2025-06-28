# app/app_dashboard.py
import streamlit as st
import requests # Để gọi API backend
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from core.config import (
    API_BASE_URL,
    DEFAULT_PREDICTION_HORIZON_DISPLAY,
    AVAILABLE_PREDICTION_HORIZONS_DISPLAY,
    DEFAULT_CRYPTO_TICKERS,
    PREDICTION_TARGET_TICKERS
)
import time 

st.set_page_config(layout="wide", page_title="Crypto Dashboard Thông Minh")

# --- DANH SÁCH CÁC TICKER HỖ TRỢ REAL-TIME  ---
REALTIME_SUPPORTED_TICKERS_APP = ["BTC-USD", "ETH-USD", "BNB-USD"]

# --- Helper Functions ---
def fetch_realtime_ticks_api(ticker, minutes=30):
    try:
        response = requests.get(f"{API_BASE_URL}/crypto/{ticker}/realtime-ticks?minutes_history={minutes}")
        response.raise_for_status()
        data = response.json()
        
        raw_timestamps_for_debug = []
        if data:
            raw_timestamps_for_debug = [item.get('timestamp') for item in data[:5]]

        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp_original_str'] = df['timestamp'] 
            
            parsed_timestamps = []
            for ts_str in df['timestamp_original_str']:
                if not isinstance(ts_str, str):
                    parsed_timestamps.append(pd.NaT)
                    continue
                
                parsed_dt = pd.NaT
                try: 
                    parsed_dt = pd.to_datetime(ts_str, format="%Y-%m-%dT%H:%M:%S.%f%z")
                except ValueError:
                    try: 
                        parsed_dt = pd.to_datetime(ts_str, format="%Y-%m-%dT%H:%M:%S%z")
                    except ValueError:
                        try: 
                            parsed_dt = pd.to_datetime(ts_str) 
                        except ValueError:
                            pass 
                parsed_timestamps.append(parsed_dt)
            
            df['timestamp'] = pd.Series(parsed_timestamps, index=df.index)

            num_total = len(df['timestamp_original_str'])
            num_nat = df['timestamp'].isnull().sum()

            if num_nat > 0:
                print(f"Warning: For {ticker}, {num_nat}/{num_total} timestamps were unparseable and set to NaT.")
                problematic_original_values = df.loc[df['timestamp'].isnull(), 'timestamp_original_str'].head(5).tolist()
                print(f"  Problematic original values: {problematic_original_values}")
            
            if num_nat > 0:
                original_len = len(df)
                df = df.dropna(subset=['timestamp'])
                if len(df) < original_len:
                    print(f"Dropped {original_len - len(df)} rows with NaT timestamps for {ticker}.")
        
        return df
    except requests.exceptions.RequestException as e:
        print(f"Lỗi API real-time cho {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Lỗi không xác định trong fetch_realtime_ticks_api cho {ticker}: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()

def plot_realtime_data(df_rt, ticker):
    if df_rt.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_rt['timestamp'], 
        y=df_rt['price'],
        mode='lines',
        name='Giá Real-time'
    ))

    latest_price_val = "N/A"
    latest_timestamp_val = None
    latest_timestamp_str = "N/A"

    if not df_rt.empty:
        latest_price_val = df_rt['price'].iloc[-1]
        latest_timestamp_val = df_rt['timestamp'].iloc[-1] # Đây là đối tượng datetime

        # Định dạng lại timestamp để hiển thị
        if pd.notna(latest_timestamp_val): 
            try:
                latest_timestamp_str = latest_timestamp_val.strftime('%H:%M:%S')
            except AttributeError: 
                 latest_timestamp_str = "Lỗi giờ"


    price_display = f"${latest_price_val:,.2f}" if isinstance(latest_price_val, (int, float)) else "N/A"

    fig.update_layout(
        title=f'Giá Real-time cho {ticker.upper()} 30 phút qua - Giá cuối: {price_display} (thời điểm {latest_timestamp_str})', 
        xaxis_title='Thời gian',
        yaxis_title='Giá (USD)',
        uirevision='realtime_plot' 
    )
    return fig

def fetch_crypto_history(ticker, limit=90):
    try:
        response = requests.get(f"{API_BASE_URL}/crypto/{ticker}/history?limit={limit}")
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi lấy dữ liệu lịch sử cho {ticker}: {e}")
        return pd.DataFrame()

def fetch_future_predictions(ticker, horizon):
    try:
        response = requests.get(f"{API_BASE_URL}/crypto/{ticker}/predict-future?horizon={horizon}")
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi lấy dữ liệu dự đoán cho {ticker} (horizon {horizon}): {e}")
        return None

def plot_price_and_ta(df, ticker): 
    if df.empty:
        st.warning(f"Không có dữ liệu để vẽ biểu đồ cho {ticker}.")
        return
    df.columns = [col.lower() for col in df.columns]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Giá OHLC'))
    if 'sma_20' in df.columns and df['sma_20'].notna().any():
        fig.add_trace(go.Scatter(x=df['date'], y=df['sma_20'], mode='lines', name='SMA 20', line=dict(color='orange')))
    if 'sma_50' in df.columns and df['sma_50'].notna().any():
        fig.add_trace(go.Scatter(x=df['date'], y=df['sma_50'], mode='lines', name='SMA 50', line=dict(color='purple')))
    if 'ema_20' in df.columns and df['ema_20'].notna().any():
        fig.add_trace(go.Scatter(x=df['date'], y=df['ema_20'], mode='lines', name='EMA 20', line=dict(color='cyan')))
    bb_lower_col = next((col for col in df.columns if col.lower().startswith('bbl_')), None)
    bb_upper_col = next((col for col in df.columns if col.lower().startswith('bbu_')), None)
    if bb_lower_col and df[bb_lower_col].notna().any():
         fig.add_trace(go.Scatter(x=df['date'], y=df[bb_lower_col], mode='lines', name='BB Lower', line=dict(color='gray', dash='dash')))
    if bb_upper_col and df[bb_upper_col].notna().any():
         fig.add_trace(go.Scatter(x=df['date'], y=df[bb_upper_col], mode='lines', name='BB Upper', line=dict(color='gray', dash='dash'),
                                  fill='tonexty' if bb_lower_col else None, 
                                  fillcolor='rgba(128,128,128,0.1)'))
    fig.update_layout(title=f'Biểu đồ Giá và Phân tích Kỹ thuật cho {ticker.upper()}', xaxis_title='Ngày', yaxis_title='Giá (USD)', xaxis_rangeslider_visible=False, legend_title_text='Chỉ báo')
    st.plotly_chart(fig, use_container_width=True)
    rsi_col = next((col for col in df.columns if col.lower().startswith('rsi_')), None)
    if rsi_col and df[rsi_col].notna().any():
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['date'], y=df[rsi_col], mode='lines', name=rsi_col.upper()))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Quá mua (70)", annotation_position="bottom right")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Quá bán (30)", annotation_position="bottom right")
        fig_rsi.update_layout(title=f'Chỉ số {rsi_col.upper()} cho {ticker.upper()}', yaxis_title='RSI')
        st.plotly_chart(fig_rsi, use_container_width=True)
    macd_line_col = next((col for col in df.columns if col.lower().startswith('macd_') and not col.lower().endswith(('_signal', '_hist', 'h', 's'))), None)
    macd_signal_col = next((col for col in df.columns if col.lower().endswith(('_signal', 's')) and 'macd' in col.lower()), None)
    macd_hist_col = next((col for col in df.columns if col.lower().endswith(('_hist', 'h')) and 'macd' in col.lower()), None)
    if macd_line_col and macd_signal_col and macd_hist_col and \
       df[macd_line_col].notna().any() and df[macd_signal_col].notna().any() and df[macd_hist_col].notna().any():
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['date'], y=df[macd_line_col], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df['date'], y=df[macd_signal_col], mode='lines', name='Signal Line'))
        colors = ['green' if val >= 0 else 'red' for val in df[macd_hist_col]]
        fig_macd.add_trace(go.Bar(x=df['date'], y=df[macd_hist_col], name='Histogram', marker_color=colors))
        fig_macd.update_layout(title=f'Chỉ số MACD cho {ticker.upper()}', yaxis_title='Giá trị MACD')
        st.plotly_chart(fig_macd, use_container_width=True)


def plot_future_predictions(predictions_data, df_history): 
    if predictions_data is None or not predictions_data.get('predictions'):
        st.warning("Không có dữ liệu dự đoán để hiển thị.")
        return
    ticker = predictions_data['ticker']
    preds = predictions_data['predictions']
    if not preds: 
        st.warning(f"Không có điểm dự đoán nào cho {ticker}.")
        return
    df_preds = pd.DataFrame(preds)
    last_actual_date_str = predictions_data.get('last_actual_date')
    if last_actual_date_str:
        last_date = pd.to_datetime(last_actual_date_str)
    elif not df_history.empty and 'date' in df_history.columns:
        last_date = df_history['date'].max()
    else:
        st.error("Không thể xác định ngày bắt đầu cho dự đoán do thiếu dữ liệu lịch sử.")
        return
    future_dates = [last_date + timedelta(days=i) for i in df_preds['date_index']]
    fig = go.Figure()
    if not df_history.empty and 'date' in df_history.columns and 'close' in df_history.columns:
         history_to_plot = df_history.tail(90)
         fig.add_trace(go.Scatter(x=history_to_plot['date'], y=history_to_plot['close'], mode='lines', name='Giá Đóng cửa Thực tế', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=df_preds['predicted_price'], mode='lines+markers', name='Giá Dự đoán', line=dict(color='red', dash='dot')))
    fig.update_layout(title=f'Dự đoán Giá cho {ticker.upper()}', xaxis_title='Ngày', yaxis_title='Giá (USD)', legend_title_text='Loại dữ liệu')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Chi tiết Dự đoán:")
    st.dataframe(pd.DataFrame({'Ngày Dự Kiến': future_dates, 'Giá Dự Đoán (USD)': df_preds['predicted_price']}))


def ask_rag_api(question_text): 
    if not question_text.strip(): return {"error": "Câu hỏi không được để trống."}
    try:
        response = requests.post(f"{API_BASE_URL}/rag/ask", json={"question": question_text})
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"Lỗi khi gọi API RAG: {str(e)}"); return {"error": f"Lỗi API: {str(e)}"}
    except Exception as e: st.error(f"Lỗi không xác định khi hỏi RAG: {str(e)}"); return {"error": f"Lỗi không xác định: {str(e)}"}

def trigger_ingest_api():
    try:
        response = requests.post(f"{API_BASE_URL}/rag/ingest-data"); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"Lỗi khi kích hoạt ingest: {str(e)}"); return None
    except Exception as e:  st.error(f"Lỗi không xác định khi kích hoạt ingest: {str(e)}"); return None

def fetch_api_news(ticker=None, limit=10, page=1): 
    try:
        url = f"{API_BASE_URL}/news/latest?limit={limit}&page={page}"
        if ticker: url += f"&ticker={ticker}"
        response = requests.get(url); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"Lỗi khi lấy tin tức từ API: {str(e)}"); return []
    except Exception as e:  st.error(f"Lỗi không xác định khi lấy tin tức: {str(e)}"); return []


# --- Giao diện chính của Dashboard ---
st.title("📊 Dashboard Phân tích Crypto Thông Minh")

display_tickers_sidebar = sorted(list(set(DEFAULT_CRYPTO_TICKERS + PREDICTION_TARGET_TICKERS)))
selected_ticker_global = st.sidebar.selectbox(
    "Chọn Ticker Crypto:",
    display_tickers_sidebar,
    index=display_tickers_sidebar.index("BTC-USD") if "BTC-USD" in display_tickers_sidebar else 0
)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Trợ lý Crypto (RAG)")
if st.sidebar.button("Nạp lại/Cập nhật Cơ sở Tri thức Trợ lý"):
    with st.spinner("Đang gửi yêu cầu nạp lại dữ liệu cho trợ lý..."):
        ingest_result = trigger_ingest_api()
        if ingest_result and "message" in ingest_result:
            st.sidebar.success(ingest_result.get("message", "Yêu cầu đã được gửi!"))
        else:
            st.sidebar.error("Không thể gửi yêu cầu nạp lại hoặc không nhận được phản hồi hợp lệ.")

tab_titles = ["⏱️ Real-time", "📈 Lịch sử & TA", "🔮 Dự đoán", "💬 Hỏi Trợ lý", "📰 Tin tức"]
tabs = st.tabs(tab_titles)

# Tab Real-time
with tabs[0]:
    st.subheader(f"Theo dõi Giá Real-time")
    st.caption("Dữ liệu từ Binance WebSocket, cập nhật sau vài giây. Hiển thị 30 phút gần nhất.")

    default_rt_index = 0
    if selected_ticker_global in REALTIME_SUPPORTED_TICKERS_APP: 
        try:
            default_rt_index = REALTIME_SUPPORTED_TICKERS_APP.index(selected_ticker_global)
        except ValueError: pass

    selected_ticker_rt = st.selectbox(
        "Chọn Ticker Real-time:",
        REALTIME_SUPPORTED_TICKERS_APP, 
        index=default_rt_index,
        key="selectbox_realtime_ticker"
    )

    chart_placeholder_rt = st.empty()

    def update_realtime_display(ticker_to_plot): 
        df_rt_data = fetch_realtime_ticks_api(ticker_to_plot, minutes=30)
        if not df_rt_data.empty:
            fig_rt = plot_realtime_data(df_rt_data, ticker_to_plot)
            if fig_rt:
                chart_placeholder_rt.plotly_chart(fig_rt, use_container_width=True)
            else: # fig_rt is None
                chart_placeholder_rt.info(f"Không thể tạo biểu đồ real-time cho {ticker_to_plot}.")
        else:
            chart_placeholder_rt.info(f"Không có dữ liệu real-time cho {ticker_to_plot} trong 30 phút qua, hoặc đang chờ cập nhật...")

    if selected_ticker_rt:
        update_realtime_display(selected_ticker_rt)

    if st.button("Làm mới Real-time Ngay", key=f"rt_refresh_manual_{selected_ticker_rt}"):
        st.rerun()


# Tab Lịch sử & TA
with tabs[1]:
    if selected_ticker_global:
        st.subheader(f"Lịch sử & Phân tích Kỹ thuật: {selected_ticker_global.upper()}")
        days_history = st.slider("Số ngày lịch sử:", 30, 60, 90, 15, key=f"hist_slider_{selected_ticker_global}")
        df_history = fetch_crypto_history(selected_ticker_global, limit=days_history)
        if not df_history.empty:
            plot_price_and_ta(df_history, selected_ticker_global)
        else:
            st.info(f"Không có dữ liệu lịch sử cho {selected_ticker_global}.")
    else:
        st.info("Chọn một ticker từ sidebar.")


# Tab Dự đoán
with tabs[2]:
    if selected_ticker_global:
        if selected_ticker_global in PREDICTION_TARGET_TICKERS:
            st.subheader(f"Dự đoán Giá Đóng Cửa: {selected_ticker_global.upper()}")
            if len(AVAILABLE_PREDICTION_HORIZONS_DISPLAY) == 1:
                selected_horizon = AVAILABLE_PREDICTION_HORIZONS_DISPLAY[0]
            else:
                selected_horizon = st.selectbox(
                    "Khoảng thời gian dự đoán (ngày):",
                    AVAILABLE_PREDICTION_HORIZONS_DISPLAY,
                    index=AVAILABLE_PREDICTION_HORIZONS_DISPLAY.index(DEFAULT_PREDICTION_HORIZON_DISPLAY)
                            if DEFAULT_PREDICTION_HORIZON_DISPLAY in AVAILABLE_PREDICTION_HORIZONS_DISPLAY else 0,
                    key=f"horizon_select_{selected_ticker_global}"
                )
            if st.button(f"Xem Dự đoán {selected_horizon} ngày", key=f"predict_btn_{selected_ticker_global}_{selected_horizon}"):
                with st.spinner("Đang tải dự đoán..."):
                    predictions = fetch_future_predictions(selected_ticker_global, selected_horizon)
                df_hist_for_pred_plot = fetch_crypto_history(selected_ticker_global, limit=90)
                if predictions:
                    plot_future_predictions(predictions, df_hist_for_pred_plot)
                else:
                    st.error(f"Không thể tải dự đoán cho {selected_ticker_global}, horizon {selected_horizon} ngày.")
        else:
            st.info(f"Hiện chưa có mô hình dự đoán giá cho {selected_ticker_global.upper()}.")
    else:
        st.info("Chọn một ticker từ sidebar.")


# Tab Hỏi Trợ lý Crypto
with tabs[3]:
    st.subheader("💬 Hỏi Trợ lý Crypto")
    st.info("Tôi là trợ lý Crypto, tôi có thể giúp bạn hiểu và áp dụng tốt các kiến thức tiền ảo! Hãy thách thức tôi!!!")
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Xem nguồn tham khảo"):
                    for i, src in enumerate(message["sources"]):
                        st.caption(f"Nguồn {i+1}: {src.get('metadata',{}).get('source','N/A')}")
                        st.markdown(f"> _{src.get('content_preview')}_")
    if user_prompt := st.chat_input("Câu hỏi của bạn về crypto..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"): st.markdown(user_prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Trợ lý đang suy nghĩ..."):
                rag_response_data = ask_rag_api(user_prompt)
            if rag_response_data and not rag_response_data.get("error"):
                full_response_text = rag_response_data.get("answer", "Xin lỗi, tôi không tìm thấy câu trả lời.")
                sources_data = rag_response_data.get("sources")
            else:
                full_response_text = rag_response_data.get("error", "Đã có lỗi xảy ra với trợ lý.")
                sources_data = None
            message_placeholder.markdown(full_response_text)
            if sources_data:
                 with st.expander("Xem nguồn tham khảo"):
                    for i, src in enumerate(sources_data):
                        st.caption(f"Nguồn {i+1}: {src.get('metadata',{}).get('source','N/A')}")
                        st.markdown(f"> _{src.get('content_preview')}_")
        st.session_state.messages.append({"role": "assistant", "content": full_response_text, "sources": sources_data})

# Tab Tin tức & Tâm lý
with tabs[4]:
    if selected_ticker_global:
        st.subheader(f"Tin tức Crypto (Liên quan đến {selected_ticker_global.upper()})")
        num_news_items = st.slider("Số lượng tin tức:", 5, 50, 10, key=f"news_slider_{selected_ticker_global}")
        news_articles = fetch_api_news(ticker=selected_ticker_global, limit=num_news_items)
        if news_articles:
            for article in news_articles:
                col1, col2 = st.columns([4,1])
                with col1:
                    st.markdown(f"**[{article.get('title', 'N/A')}]({article.get('url', '#')})**")
                    st.caption(f"Nguồn: {article.get('domain', 'N/A')} | "
                               f"Đăng lúc: {pd.to_datetime(article.get('published_at')).strftime('%Y-%m-%d %H:%M') if article.get('published_at') else 'N/A'}")
                    if article.get('related_currencies'):
                        st.caption(f"Liên quan đến: {article['related_currencies']}")
                with col2:
                    score = article.get('sentiment_score')
                    label = article.get('sentiment_label', 'N/A')
                    if score is not None:
                        if label == 'positive': st.success(f"Tích cực ({score:.2f})")
                        elif label == 'negative': st.error(f"Tiêu cực ({score:.2f})")
                        else: st.info(f"Trung tính ({score:.2f})")
                    else: st.caption("Chưa có sentiment")
                st.divider()
        else:
            st.info("Không tìm thấy tin tức nào phù hợp.")
    else:
        st.info("Chọn một ticker từ sidebar.")
