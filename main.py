import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from sqlalchemy import create_engine
from datetime import datetime
import logging

# Đọc biến môi trường
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kết nối PostgreSQL
engine = create_engine(DATABASE_URL)

# Hàm đọc dữ liệu lịch sử
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM history ORDER BY created_at DESC LIMIT 100", engine)
        return df
    except Exception as e:
        logger.error(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

# Hàm lưu kết quả vào DB
def save_result(real, predicted):
    try:
        query = f"""
            INSERT INTO history (real_result, predicted_result, created_at)
            VALUES ('{real}', '{predicted}', '{datetime.now()}')
        """
        with engine.connect() as conn:
            conn.execute(query)
    except Exception as e:
        logger.error(f"Lỗi lưu dữ liệu: {e}")

# Hàm xử lý dự đoán kết quả tiếp theo
def extract_features(results):
    features = []
    for r in results:
        total = sum(map(int, r.split("-")))
        is_even = total % 2 == 0
        is_tai = total > 10
        is_bao = len(set(r.split("-"))) == 1
        features.append([total, int(is_even), int(is_tai), int(is_bao)])
    return np.array(features)

# Các mô hình máy học
def predict_with_models(X):
    rf = RandomForestClassifier()
    xgb = XGBClassifier()
    mlp = MLPClassifier()
    lstm_model = Sequential([
        LSTM(32, input_shape=(X.shape[1], 1)),
        Dense(4, activation='softmax')
    ])
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Tạo nhãn giả
    y = [random.choice(["Tài", "Xỉu", "Chẵn", "Lẻ"]) for _ in range(len(X))]

    # Huấn luyện mô hình
    rf.fit(X, y)
    xgb.fit(X, y)
    mlp.fit(X, y)
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
    y_lstm = tf.keras.utils.to_categorical([["Tài", "Xỉu", "Chẵn", "Lẻ"].index(label) for label in y], num_classes=4)
    lstm_model.fit(X_lstm, y_lstm, epochs=5, verbose=0)

    # Dự đoán với mẫu mới nhất
    last_sample = X[-1].reshape(1, -1)
    preds = [
        rf.predict(last_sample)[0],
        xgb.predict(last_sample)[0],
        mlp.predict(last_sample)[0],
        ["Tài", "Xỉu", "Chẵn", "Lẻ"][np.argmax(lstm_model.predict(last_sample.reshape((1, X.shape[1], 1)), verbose=0))]
    ]
    return preds

# Tổng hợp kết quả
def voting(preds):
    return max(set(preds), key=preds.count)

# Logic mô phỏng con người
def logic_suy_luan(recent):
    totals = [sum(map(int, r.split("-"))) for r in recent]
    chẵn_lẻ = ["Chẵn" if t % 2 == 0 else "Lẻ" for t in totals]
    tai_xiu = ["Tài" if t > 10 else "Xỉu" for t in totals]
    bao = [1 if r.split("-")[0] == r.split("-")[1] == r.split("-")[2] else 0 for r in recent]

    # Dự đoán nếu có 3 lần liên tiếp giống nhau
    if len(set(chẵn_lẻ[-3:])) == 1:
        du_doan_le = "Chẵn" if chẵn_lẻ[-1] == "Lẻ" else "Lẻ"
    else:
        du_doan_le = chẵn_lẻ[-1]

    if len(set(tai_xiu[-3:])) == 1:
        du_doan_tx = "Tài" if tai_xiu[-1] == "Xỉu" else "Xỉu"
    else:
        du_doan_tx = tai_xiu[-1]

    is_bao = any(bao[-5:])
    return du_doan_tx, du_doan_le, is_bao

# Gửi phản hồi
async def handle_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text.count("-") == 2:
        await update.message.reply_text("⚠️ Vui lòng nhập kết quả theo định dạng: 1-3-6")
        return

    real_result = text
    df = load_data()
    history = df["real_result"].tolist()[-20:] if not df.empty else []
    history.append(real_result)

    X = extract_features(history)
    preds = predict_with_models(X)
    final_vote = voting(preds)

    du_doan_tx, du_doan_le, bao = logic_suy_luan(history)
    save_result(real_result, f"{du_doan_tx}-{du_doan_le}")

    message = f"✅ Đã nhận kết quả: `{real_result}`\n\n"
    message += f"🔮 Dự đoán phiên tiếp theo:\n"
    message += f"- {du_doan_tx} - {du_doan_le}\n"
    message += f"- Dải điểm nên đánh: {'11–17' if du_doan_tx == 'Tài' else '4–10'}\n"
    if bao:
        message += "⚠️ *Cảnh báo: Có khả năng 'Bão'!*\n"
    message += f"\n📊 Tổng phiên đã lưu: {len(df)} | Đúng: ? | Sai: ?"

    await update.message.reply_text(message, parse_mode="Markdown")

# Lệnh bắt đầu
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Chào bạn! Gửi kết quả Tài Xỉu (ví dụ: `2-5-6`) để nhận dự đoán phiên tiếp theo.")

# Khởi tạo bot
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_result))
    print("Bot đang chạy...")
    app.run_polling()

if __name__ == "__main__":
    main()
