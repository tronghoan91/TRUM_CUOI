import os
import logging
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from joblib import dump, load
import psycopg2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timedelta

# --- Cấu hình biến môi trường & model ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_PATH = "sicbo_model.joblib"

# --- Tham số cho tự động phát hiện đổi thuật toán ---
MIN_ACCURACY = 0.5      # Ngưỡng tỉ lệ đúng tối thiểu (50%)
WINDOW_SIZE = 40        # Số ván gần nhất để đánh giá

logging.basicConfig(level=logging.INFO)

# --- Các hàm thao tác DB ---
def get_db_conn():
    return psycopg2.connect(DATABASE_URL)

def create_table():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id SERIAL PRIMARY KEY,
            input TEXT,
            prediction TEXT,
            actual TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """
    )
    conn.commit()
    cur.close()
    conn.close()

def insert_to_db(numbers, prediction, actual=None):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (input, prediction, actual) VALUES (%s, %s, %s)",
        (numbers, prediction, actual)
    )
    conn.commit()
    cur.close()
    conn.close()

def fetch_history(limit=500, with_actual=True):
    conn = get_db_conn()
    query = "SELECT id, input, prediction, actual, created_at FROM history"
    if with_actual:
        query += " WHERE actual IS NOT NULL"
    query += " ORDER BY id DESC LIMIT %s"
    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()
    return df

# --- Tiện ích xử lý dữ liệu ---
def extract_features(results):
    return [int(n) for n in results.split()]

def label_func(nums):
    total = sum(nums)
    return "Tài" if total >= 11 else "Xỉu"

# --- Model & ML ---
def train_and_save_model():
    df = fetch_history(2000)
    if df.empty:
        return None
    df = df[df["actual"].notnull()]
    if len(df) < 10:
        return None  # Cần ít nhất 10 mẫu thực tế để train
    X = np.array([extract_features(i) for i in df["input"]])
    y = df["actual"].values
    models = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("xgb", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")),
        ("mlp", MLPClassifier(max_iter=2000))
    ]
    ensemble = VotingClassifier(estimators=models, voting='hard')
    ensemble.fit(X, y)
    dump(ensemble, MODEL_PATH)
    return ensemble

def load_model():
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    return train_and_save_model()

def predict_with_model(model, input_data):
    X = np.array([extract_features(input_data)])
    return model.predict(X)[0]

# --- Tự động phát hiện đổi thuật toán ---
def detect_algo_change():
    df = fetch_history(WINDOW_SIZE)
    if len(df) < WINDOW_SIZE:
        return False
    acc = sum(df['prediction'] == df['actual']) / WINDOW_SIZE
    return acc < MIN_ACCURACY

def train_with_recent_data(n=100):
    df = fetch_history(n)
    df = df[df["actual"].notnull()]
    if len(df) < 10:
        return None
    X = np.array([extract_features(i) for i in df["input"]])
    y = df["actual"].values
    models = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("xgb", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")),
        ("mlp", MLPClassifier(max_iter=2000))
    ]
    ensemble = VotingClassifier(estimators=models, voting='hard')
    ensemble.fit(X, y)
    dump(ensemble, MODEL_PATH)
    return ensemble

# --- Thống kê & thông báo ---
def get_last_play_time():
    df = fetch_history(1, with_actual=False)
    if df.empty:
        return None
    return df["created_at"].iloc[0]

def time_diff_message(last_time):
    if last_time is None:
        return ""
    now = datetime.now(last_time.tzinfo)
    diff = now - last_time
    if diff > timedelta(hours=4):
        return ("⚠️ Lưu ý: Đã lâu bạn chưa nhập kết quả thực tế vào bot. "
                "Kết quả dự đoán chỉ có tính chất tham khảo, không đảm bảo đúng với từng thời điểm hoặc phiên chơi, "
                "đặc biệt nếu bạn chơi ở các thời điểm khác nhau.")
    return ""

def generate_response(prediction, input_text, stats, time_msg):
    nums = list(map(int, input_text.split()))
    total = sum(nums)
    tai_xiu = "Tài" if total >= 11 else "Xỉu"
    chan_le = "Chẵn" if total % 2 == 0 else "Lẻ"
    bao = "🎲 BÃO! Ba số giống nhau!" if len(set(nums)) == 1 else ""
    response = (
        f"🎯 Dự đoán: {prediction}\n"
        f"🔢 Tổng: {total} → {tai_xiu} - {chan_le}\n"
        f"{bao}\n\n"
        f"📊 Thống kê gần đây:\n"
        f"✔️ Đúng: {stats['correct']} | ❌ Sai: {stats['wrong']} | 🎯 Tỉ lệ: {stats['accuracy']}%\n"
    )
    if time_msg:
        response += "\n" + time_msg
    return response.strip()

def calculate_stats():
    df = fetch_history(50)
    correct = sum(df['prediction'] == df['actual'])
    total = len(df)
    wrong = total - correct
    acc = round(correct / total * 100, 2) if total > 0 else 0
    return {"correct": correct, "wrong": wrong, "accuracy": acc}

# --- Handler Bot ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Gửi 3 số kết quả gần nhất để nhận dự đoán Tài/Xỉu (VD: 1 3 2).\n"
        "Hãy gửi liên tục kết quả của các phiên để bot tự học và ngày càng chính xác nhé!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not re.match(r"^\d+ \d+ \d+$", text):
        await update.message.reply_text("⚠️ Vui lòng nhập đúng định dạng: 3 số cách nhau bằng khoảng trắng (VD: 1 2 3)")
        return

    # Gán nhãn thực tế cho lượt chơi trước đó (nếu có)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, input FROM history WHERE actual IS NULL ORDER BY id DESC LIMIT 1")
    last_entry = cur.fetchone()
    cur.close()
    if last_entry:
        last_id, last_input = last_entry
        actual_label = label_func(extract_features(text))
        conn2 = get_db_conn()
        cur2 = conn2.cursor()
        cur2.execute("UPDATE history SET actual = %s WHERE id = %s", (actual_label, last_id))
        conn2.commit()
        cur2.close()
        conn2.close()
        # Sau mỗi lần có dữ liệu thực tế, train lại mô hình
        train_and_save_model()

    # Phát hiện đổi thuật toán, nếu có thì chỉ train lại bằng dữ liệu gần nhất và cảnh báo
    algo_changed = detect_algo_change()
    if algo_changed:
        train_with_recent_data(WINDOW_SIZE * 2)
        await update.message.reply_text(
            f"⚠️ BOT phát hiện tỉ lệ dự đoán đúng giảm mạnh! Game có thể đã đổi thuật toán. "
            f"BOT sẽ tự động học lại dựa trên {WINDOW_SIZE * 2} ván gần nhất để thích ứng sóng mới!"
        )

    # Dự đoán kết quả
    model = load_model()
    if model is not None:
        prediction = predict_with_model(model, text)
    else:
        prediction = label_func(extract_features(text))

    insert_to_db(text, prediction, actual=None)
    stats = calculate_stats()
    time_msg = time_diff_message(get_last_play_time())
    response = generate_response(prediction, text, stats, time_msg)
    await update.message.reply_text(response)

# --- Main ---
def main():
    create_table()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
