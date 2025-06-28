# prediction/train_predict_models.py
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shutil
import gc 

from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import MinMaxScaler
from core.db_connect import engine 

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from .prediction_utils import inverse_transform_predictions
from core.config import (
    PREDICTION_TARGET_TICKERS, PREDICTION_OUTPUT_LENGTHS,
    PREDICTION_INPUT_FEATURES, PREDICTION_TARGET_FEATURE, PREDICTION_SEQUENCE_LENGTH, 
    PREPARED_DATA_DIR_BASE, TRAINED_MODELS_DIR_BASE,
    EPOCHS as CONFIG_EPOCHS,
    BATCH_SIZE as CONFIG_BATCH_SIZE,
    VALIDATION_SPLIT as CONFIG_VALIDATION_SPLIT, 
    LSTM_UNITS as CONFIG_LSTM_UNITS,
    DROPOUT_RATE as CONFIG_DROPOUT_RATE,
    LEARNING_RATE as CONFIG_LEARNING_RATE,
    PATIENCE_EARLY_STOPPING as CONFIG_PATIENCE_EARLY_STOPPING,
    PATIENCE_REDUCE_LR as CONFIG_PATIENCE_REDUCE_LR,
    PREDICTION_NUM_FEATURES, 
    get_db_table_name 
)

def get_specific_data_output_dir(base_dir, ticker, output_len):
    ticker_subdir = ticker.lower().replace('-','_')
    return os.path.join(base_dir, ticker_subdir, f"output_{output_len}")

def get_specific_model_output_dir(base_dir, ticker, output_len):
    ticker_subdir = ticker.lower().replace('-','_')
    return os.path.join(base_dir, ticker_subdir, f"output_{output_len}")

def load_data_from_db(ticker, engine_conn, feature_cols, seq_len, out_len):
    table_name = get_db_table_name(ticker) 
    print(f"[{ticker}/Out:{out_len}] Đang tải dữ liệu từ bảng DB: {table_name}...")
    try:
        sql_feature_cols = ', '.join([f'"{col}"' for col in feature_cols])
        query = f'SELECT "date", {sql_feature_cols} FROM {table_name} ORDER BY "date" ASC'
        df = pd.read_sql(query, engine_conn, parse_dates=['date'])

        if df.empty:
            print(f"[{ticker}/Out:{out_len}] Lỗi: Không tìm thấy dữ liệu trong bảng '{table_name}'.")
            return None
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
             print(f"[{ticker}/Out:{out_len}] Lỗi: Thiếu các cột sau: {missing_cols}")
             return None
        if len(df) < seq_len + out_len:
             print(f"[{ticker}/Out:{out_len}] Lỗi: Dữ liệu quá ngắn ({len(df)} dòng) cho seq_len={seq_len}, out_len={out_len}.")
             return None
        print(f"[{ticker}/Out:{out_len}] Tải thành công {len(df)} dòng dữ liệu từ DB.")
        return df
    except SQLAlchemyError as e:
        print(f"[{ticker}/Out:{out_len}] Lỗi khi tải dữ liệu từ bảng '{table_name}': {e}")
        return None

def preprocess_and_create_sequences(df, input_feature_list, target_feature, seq_length, out_length, ticker_info=""):
    print(f"[{ticker_info}] Chuẩn bị dữ liệu từ các cột: {input_feature_list}, Target: '{target_feature}' cho seq_len={seq_length}, out_len={out_length}")
    try:
        target_col_index = input_feature_list.index(target_feature)
    except ValueError:
        print(f"[{ticker_info}] Lỗi: Cột target '{target_feature}' không có trong danh sách input_feature_list.")
        return None, None, None, None, -1

    data = df[input_feature_list].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print(f"[{ticker_info}] Dữ liệu input đã được chuẩn hóa. Shape: {scaled_data.shape}")

    X, y = [], []
    for i in range(len(scaled_data) - seq_length - out_length + 1):
        input_seq = scaled_data[i:(i + seq_length), :]
        output_seq = scaled_data[(i + seq_length):(i + seq_length + out_length), target_col_index]
        X.append(input_seq)
        y.append(output_seq)

    X = np.array(X); y = np.array(y)
    print(f"[{ticker_info}] Đã tạo sequences: X shape: {X.shape}, y shape: {y.shape}")
    return X, y, scaler, scaled_data, target_col_index

def load_prepared_data_files(ticker, output_len, input_dir_base, num_expected_features):
    specific_input_dir = get_specific_data_output_dir(input_dir_base, ticker, output_len)
    if not os.path.exists(specific_input_dir):
        print(f"[{ticker}/Out:{output_len}] Thư mục dữ liệu chuẩn bị '{specific_input_dir}' không tồn tại.")
        return None, None, None, None, -1

    base_filename = os.path.join(specific_input_dir, "data")
    X_filename = f'{base_filename}_X.npy'; y_filename = f'{base_filename}_y.npy'
    scaler_filename = f'{base_filename}_scaler.pkl'; original_data_filename = f'{base_filename}_original_scaled.npy'
    target_idx_filename = f'{base_filename}_target_idx.txt'

    required_files = [X_filename, y_filename, scaler_filename, original_data_filename, target_idx_filename]
    if not all(os.path.exists(f) for f in required_files):
        print(f"[{ticker}/Out:{output_len}] Thiếu file dữ liệu trong '{specific_input_dir}'.")
        return None, None, None, None, -1

    print(f"[{ticker}/Out:{output_len}] Đang tải dữ liệu đã chuẩn bị từ '{specific_input_dir}'...")
    try:
        X = np.load(X_filename); y = np.load(y_filename)
        original_scaled = np.load(original_data_filename); scaler = pickle.load(open(scaler_filename, 'rb'))
        with open(target_idx_filename, 'r') as f:
            target_col_index = int(f.read())

        print(f"[{ticker}/Out:{output_len}] Tải dữ liệu từ file thành công. Target index: {target_col_index}")
        if scaler.n_features_in_ != num_expected_features:
             print(f"[{ticker}/Out:{output_len}] Lỗi: Scaler có {scaler.n_features_in_} features, cấu hình yêu cầu {num_expected_features}.")
             return None, None, None, None, -1
        return X, y, scaler, original_scaled, target_col_index
    except Exception as e:
        print(f"[{ticker}/Out:{output_len}] Lỗi khi tải dữ liệu từ file: {e}")
        return None, None, None, None, -1

def build_lstm_model(input_shape, output_length, lstm_units_cfg, dropout_rate_cfg):
    model = Sequential()
    model.add(LSTM(units=lstm_units_cfg, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate_cfg))
    model.add(LSTM(units=lstm_units_cfg // 2, return_sequences=False))
    model.add(Dropout(dropout_rate_cfg))
    model.add(Dense(units=output_length))
    model.summary()
    return model

def plot_predictions(ticker, output_len, original_scaled_data,
                     train_indices, train_predictions_close,
                     test_indices, test_predictions_close,
                     future_indices, future_predictions_close,
                     target_feature_name, target_feature_idx, num_total_features,
                     scaler_obj, save_dir=None):
    plt.figure(figsize=(18, 9))
    original_values_all_features = scaler_obj.inverse_transform(original_scaled_data)
    original_target_values = original_values_all_features[:, target_feature_idx]

    plt.plot(original_target_values, label=f'Giá thực tế ({target_feature_name.capitalize()})', color='blue', linewidth=1.5, alpha=0.7)
    if train_indices is not None and train_predictions_close is not None:
         plt.plot(train_indices, train_predictions_close, label=f'Dự đoán Train', color='orange', linewidth=1, alpha=0.8)
    if test_indices is not None and test_predictions_close is not None:
        plt.plot(test_indices, test_predictions_close, label=f'Dự đoán Test', color='green', linewidth=1, alpha=0.8, linestyle='-.')
    plt.plot(future_indices, future_predictions_close, label=f'Dự đoán {output_len} ngày tới', color='red', marker='o', markersize=4, linestyle='--', linewidth=1.5)

    plt.title(f'So sánh & Dự đoán cho {ticker} (Horizon: {output_len} ngày, Target: {target_feature_name.capitalize()})', fontsize=16)
    plt.xlabel('Số ngày'); plt.ylabel(f'Giá {target_feature_name.capitalize()}')
    plt.legend(); plt.grid(True); plt.tight_layout()

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plot_filename = os.path.join(save_dir, f"{ticker.lower().replace('-','_')}_pred_plot_out{output_len}.png")
        plt.savefig(plot_filename)
        print(f"Đã lưu biểu đồ dự đoán vào: {plot_filename}")
        plt.close()
    else:
        plt.show()


# --- Hàm chính thực thi ---
if __name__ == "__main__":

    for current_ticker in PREDICTION_TARGET_TICKERS: 
        for current_output_length in PREDICTION_OUTPUT_LENGTHS: 
            print(f"\n\n--- BẮT ĐẦU QUY TRÌNH CHO: {current_ticker} - HORIZON DỰ ĐOÁN: {current_output_length} NGÀY ---")

            current_target_col_index = -1

            specific_data_dir = get_specific_data_output_dir(PREPARED_DATA_DIR_BASE, current_ticker, current_output_length)
            specific_model_dir = get_specific_model_output_dir(TRAINED_MODELS_DIR_BASE, current_ticker, current_output_length)

            X_data, y_data, scaler, original_scaled_data, current_target_col_index = load_prepared_data_files(
                current_ticker, current_output_length, PREPARED_DATA_DIR_BASE, PREDICTION_NUM_FEATURES 
            )

            if X_data is None:
                print(f"[{current_ticker}/Out:{current_output_length}] Dữ liệu chuẩn bị từ file không có. Bắt đầu chuẩn bị từ DB...")
                if not os.path.exists(specific_data_dir):
                    os.makedirs(specific_data_dir)

                dataframe = load_data_from_db(current_ticker, engine, PREDICTION_INPUT_FEATURES, PREDICTION_SEQUENCE_LENGTH, current_output_length)
                if dataframe is not None:
                    X_data, y_data, scaler, original_scaled_data, current_target_col_index = preprocess_and_create_sequences(
                        dataframe, PREDICTION_INPUT_FEATURES, PREDICTION_TARGET_FEATURE, 
                        PREDICTION_SEQUENCE_LENGTH, current_output_length, 
                        ticker_info=f"{current_ticker}/Out:{current_output_length}"
                    )
                    if X_data is None:
                         print(f"[{current_ticker}/Out:{current_output_length}] Lỗi tiền xử lý. Bỏ qua."); continue

                    base_filename = os.path.join(specific_data_dir, "data")
                    try:
                        np.save(f'{base_filename}_X.npy', X_data)
                        np.save(f'{base_filename}_y.npy', y_data)
                        np.save(f'{base_filename}_original_scaled.npy', original_scaled_data)
                        with open(f'{base_filename}_scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
                        with open(f'{base_filename}_target_idx.txt', 'w') as f: f.write(str(current_target_col_index))
                        print(f"[{current_ticker}/Out:{current_output_length}] Đã lưu dữ liệu chuẩn bị vào '{specific_data_dir}'.")
                    except Exception as save_e:
                         print(f"[{current_ticker}/Out:{current_output_length}] Lỗi lưu file: {save_e}. Bỏ qua."); continue
                else:
                    print(f"[{current_ticker}/Out:{current_output_length}] Không thể tải dữ liệu từ DB. Bỏ qua."); continue
            
            if current_target_col_index == -1:
                try:
                    current_target_col_index = PREDICTION_INPUT_FEATURES.index(PREDICTION_TARGET_FEATURE) 
                except ValueError:
                    print(f"[{current_ticker}/Out:{current_output_length}] Lỗi: TARGET_FEATURE '{PREDICTION_TARGET_FEATURE}' không có trong PREDICTION_INPUT_FEATURES. Bỏ qua."); continue

            print(f"[{current_ticker}/Out:{current_output_length}] Dữ liệu sẵn sàng. Target feature index: {current_target_col_index}")

            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=CONFIG_VALIDATION_SPLIT, random_state=42, shuffle=False
            )

            model = build_lstm_model(
                input_shape=(PREDICTION_SEQUENCE_LENGTH, PREDICTION_NUM_FEATURES), 
                output_length=current_output_length,
                lstm_units_cfg=CONFIG_LSTM_UNITS, dropout_rate_cfg=CONFIG_DROPOUT_RATE 
            )

            optimizer = Adam(learning_rate=CONFIG_LEARNING_RATE) 
            model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())

            early_stopping = EarlyStopping(monitor='val_loss', patience=CONFIG_PATIENCE_EARLY_STOPPING, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=CONFIG_PATIENCE_REDUCE_LR, min_lr=1e-7, verbose=1)

            print(f"\n[{current_ticker}/Out:{current_output_length}] Bắt đầu huấn luyện...")
            history = model.fit(
                X_train, y_train,
                epochs=CONFIG_EPOCHS, batch_size=CONFIG_BATCH_SIZE, 
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr], verbose=1
            )
            print(f"[{current_ticker}/Out:{current_output_length}] Huấn luyện hoàn tất.")

            if not os.path.exists(specific_model_dir):
                os.makedirs(specific_model_dir)
            model_filename = f"model_lstm.h5"
            model_path = os.path.join(specific_model_dir, model_filename)
            try:
                model.save(model_path)
                print(f"[{current_ticker}/Out:{current_output_length}] Đã lưu model vào: {model_path}")
            except Exception as e:
                print(f"[{current_ticker}/Out:{current_output_length}] Lỗi khi lưu model: {e}")

            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Loss cho {current_ticker} (Output: {current_output_length} ngày)')
            plt.xlabel('Epoch'); plt.ylabel('Loss (Huber)'); plt.legend(); plt.grid(True)
            loss_plot_path = os.path.join(specific_model_dir, "training_loss_plot.png")
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"[{current_ticker}/Out:{current_output_length}] Đã lưu biểu đồ loss vào: {loss_plot_path}")

            predicted_train_scaled = model.predict(X_train)
            predicted_train_prices = inverse_transform_predictions(
                predicted_train_scaled, scaler, current_target_col_index, PREDICTION_NUM_FEATURES 
            )
            predicted_train_plot = predicted_train_prices[:, 0]
            train_predict_indices = np.arange(PREDICTION_SEQUENCE_LENGTH, PREDICTION_SEQUENCE_LENGTH + len(predicted_train_plot))

            predicted_test_scaled = model.predict(X_test)
            predicted_test_prices = inverse_transform_predictions(
                predicted_test_scaled, scaler, current_target_col_index, PREDICTION_NUM_FEATURES 
            )
            predicted_test_plot = predicted_test_prices[:, 0]
            test_predict_indices_start = PREDICTION_SEQUENCE_LENGTH + len(X_train) 
            test_predict_indices = np.arange(test_predict_indices_start, test_predict_indices_start + len(predicted_test_plot))

            last_sequence = original_scaled_data[-PREDICTION_SEQUENCE_LENGTH:, :] 
            last_sequence_reshaped = np.reshape(last_sequence, (1, PREDICTION_SEQUENCE_LENGTH, PREDICTION_NUM_FEATURES)) 
            predicted_future_scaled = model.predict(last_sequence_reshaped)
            predicted_future_prices = inverse_transform_predictions(
                predicted_future_scaled, scaler, current_target_col_index, PREDICTION_NUM_FEATURES 
            )[0]

            future_predict_indices_start = len(original_scaled_data)
            future_predict_indices = np.arange(future_predict_indices_start, future_predict_indices_start + current_output_length)

            plot_predictions(
                ticker=current_ticker, output_len=current_output_length,
                original_scaled_data=original_scaled_data,
                train_indices=train_predict_indices, train_predictions_close=predicted_train_plot,
                test_indices=test_predict_indices, test_predictions_close=predicted_test_plot,
                future_indices=future_predict_indices, future_predictions_close=predicted_future_prices,
                target_feature_name=PREDICTION_TARGET_FEATURE, target_feature_idx=current_target_col_index, 
                num_total_features=PREDICTION_NUM_FEATURES, scaler_obj=scaler, 
                save_dir=specific_model_dir
            )
            print(f"--- HOÀN TẤT CHO: {current_ticker} - HORIZON: {current_output_length} NGÀY ---")
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    print("\n\n--- === TẤT CẢ QUY TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT === ---")