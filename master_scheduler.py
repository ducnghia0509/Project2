# master_scheduler.py
import schedule
import time
import subprocess
import os
import sys
from datetime import datetime
from core.config import CRYPTO_PANIC_API_KEY, GOOGLE_API_KEY
from core.db_connect import engine
from sqlalchemy import text
# --- Cấu hình Đường dẫn ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXECUTABLE = sys.executable # Đường dẫn đến python đang chạy scheduler này

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# --- Hàm tiện ích để chạy script con ---
def run_script_module(module_path_from_root, log_file_name, script_args=None):
    log_full_path = os.path.join(LOGS_DIR, log_file_name)
    command = [PYTHON_EXECUTABLE, "-m", module_path_from_root]
    if script_args:
        command.extend(script_args)

    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time_str}] SCHEDULER: Attempting to run {module_path_from_root}...")

    try:
        with open(log_full_path, "a", encoding="utf-8") as logfile:
            logfile.write(f"\n--- Log at {current_time_str} for {module_path_from_root} ---\n")
            env_for_subprocess = os.environ.copy()
            env_for_subprocess["PYTHONIOENCODING"] = "utf-8"
            if CRYPTO_PANIC_API_KEY: # Chỉ thêm nếu key tồn tại trong config
                env_for_subprocess["CRYPTO_PANIC_API_KEY"] = CRYPTO_PANIC_API_KEY
            if GOOGLE_API_KEY:
                env_for_subprocess["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env_for_subprocess
            )
            stdout, stderr = process.communicate() 

            logfile.write(f"Return code: {process.returncode}\n")
            logfile.write("Stdout:\n")
            logfile.write(stdout)
            logfile.write("\nStderr:\n")
            logfile.write(stderr)
            logfile.write("\n--- End Log ---\n\n")

        if process.returncode == 0:
            print(f"[{current_time_str}] SCHEDULER: Successfully ran {module_path_from_root}")
        else:
            print(f"[{current_time_str}] SCHEDULER: Error running {module_path_from_root}. Check {log_file_name}. Return code: {process.returncode}")
            print(f"    Stderr: {stderr[:200]}...") # In một phần lỗi ra console

    except Exception as e:
        current_time_str_exc = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_full_path, "a", encoding="utf-8") as logfile:
            logfile.write(f"--- Scheduler Exception at {current_time_str_exc} for {module_path_from_root} ---\n")
            logfile.write(str(e) + "\n")
            logfile.write("--- End Exception ---\n\n")
        print(f"[{current_time_str_exc}] SCHEDULER: Exception while trying to run {module_path_from_root}: {e}")


# --- Định nghĩa Lập lịch ---
def task_update_price_data():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCHEDULER: Triggering Price Data Update...")
    run_script_module("data_processing.update_daily", "scheduler_update_price.log")

def task_fetch_news_data():
    run_script_module("news_handling.news_fetcher", "scheduler_fetch_news.log")

def task_analyze_sentiment():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCHEDULER: Triggering Sentiment Analysis...")
    run_script_module("analysis.sentiment_analyzer", "scheduler_analyze_sentiment.log")

def task_retrain_prediction_models():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCHEDULER: Triggering Prediction Models Retraining...")
    # Giả sử train_predict_models.py nằm trong prediction
    run_script_module("prediction.train_predict_models", "scheduler_retrain_models.log")

def task_cleanup_old_realtime_ticks():
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time_str}] SCHEDULER: Triggering Realtime Data Cleanup (older than 30 minutes)...")
    log_file_name = "scheduler_cleanup_realtime.log"
    log_full_path = os.path.join(LOGS_DIR, log_file_name)

    try:
        with engine.connect() as connection:
            query = text("DELETE FROM realtime_price_ticks WHERE timestamp < NOW() - INTERVAL '30 minutes'")
            result = connection.execute(query)
            connection.commit()
            deleted_rows = result.rowcount
            message = f"[{current_time_str}] SCHEDULER: Cleaned up {deleted_rows} realtime ticks older than 30 minutes." 
            print(message)
            with open(log_full_path, "a", encoding="utf-8") as logfile:
                logfile.write(message + "\n")

    except Exception as e:
        error_message = f"[{current_time_str}] SCHEDULER: Error cleaning up realtime ticks: {e}"
        print(error_message)
        with open(log_full_path, "a", encoding="utf-8") as logfile:
            logfile.write(error_message + "\n")

# --- Lập lịch cho các Tác vụ ---
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Master Scheduler started. Press Ctrl+C to exit.")

# Cập nhật dữ liệu giá mỗi ngày vào lúc 01:05 sáng
schedule.every().day.at("00:05").do(task_update_price_data).tag("data_update", "daily")

# Lấy tin tức 
schedule.every(15).minutes.do(task_fetch_news_data).tag("news_fetch", "minutely")
# Phân tích sentiment 
schedule.every(15).minutes.do(task_analyze_sentiment).tag("sentiment_analysis", "minutely")

# Huấn luyện lại mô hình vào Chủ nhật hàng tuần
schedule.every().sunday.at("02:00").do(task_retrain_prediction_models).tag("model_retrain", "weekly")

schedule.every(20).minutes.do(task_cleanup_old_realtime_ticks).tag("cleanup", "minutely")

# --- Vòng lặp chính của Scheduler ---
if __name__ == "__main__":
    try:
        while True:
            schedule.run_pending()
            time.sleep(600)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Master Scheduler stopped by user.")
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Master Scheduler encountered an unhandled exception: {e}")
    finally:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Master Scheduler shutting down.")
