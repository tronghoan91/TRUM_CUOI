import os
import logging
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from joblib import dump, load
import psycopg2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timedelta
import threading
from flask import Flask

# ----- Giữ port cho Render/UptimeRobot -----
def start_flask():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "Bot is alive!", 200

    @app.route('/healthz')
    def health():
        return "OK", 200

    app.run(host='0.0.0.0', port=10000)

threading.Thread(target=start_flask, daemon=True).start()
# -------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/sicbo_model.joblib")
BAO_MODEL_PATH = os.getenv("BAO_MODEL_PATH", "/tmp/bao_model.joblib")

MIN_ACCURACY = 0.5
WINDOW_SIZE = 40
FEATURE_WINDOW = 3  # Lấy 3 phiên gần nhất

logging.basicConfig(level=logging.INFO)

def get_db_conn():
    return psycopg2.connect(DATABASE_URL)

def create_table():
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id SERIAL PRIMARY KEY,
                    input TEXT,
                    prediction TEXT,
                    actual TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()

def insert_to_db(numbers, prediction, actual=None):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO history (input, prediction, actual) VALUES (%s, %s, %s)",
                (numbers, prediction, actual)
            )
            conn.commit()

def fetch_history(limit=500, with_actual=True):
    with get_db_conn() as conn:
        query = "SELECT id, input, prediction, actual, created_at FROM history"
        if with_actual:
            query += " WHERE actual IS NOT NULL"
        query += " ORDER BY id DESC LIMIT %s"
        df = pd.read_sql(query, conn, params=(limit,))
    return df

def extract_features(nums):
    features = []
    features.extend(nums)  # 3 số gốc
    features.append(sum(nums))
    features.append(max(nums))
    features.append(min(nums))
    features.append(np.std(nums))
    features.append(np.mean(nums))
    features.append(1 if len(set(nums)) == 1 else 0)  # bão
    features.append(1 if sum(nums) % 2 == 0 else 0)   # chẵn lẻ
    features.append(1 if sum(nums) >= 11 else 0)      # tài/xỉu
    return features

def get_window_features(history_inputs):
    features = []
    for nums in history_inputs:
        features += extract_features(nums)
    # Fill 0 nếu thiếu data
    for _ in range(FEATURE_WINDOW - len(history_inputs)):
        features += [0] * 10
    return features

def label_func(nums):
    total = sum(nums)
    return "Tài" if total >= 11 else "Xỉu"

def is_bao(nums):
    return int(len(set(nums)) == 1)

def train_and_save_model():
    df = fetch_history(2000)
    if df.empty:
        return None
    df = df[df["actual"].notnull()]
    if len(df) < 10:
        return None
    X, y = [], []
    for i in range(len(df)):
        history_inputs = []
        for j in range(FEATURE_WINDOW):
            if i-j < 0:
                continue
            nums = [int(x) for x in df.iloc[i-j]["input"].split()]
            history_inputs.insert(0, nums)
        X.append(get_window_features(history_inputs))
        y.append(df.iloc[i]["actual"])
    models = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("xgb", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")),
        ("mlp", MLPClassifier(max_iter=2000)),
        ("lr", LogisticRegression(max_iter=1000))
    ]
    ensemble = VotingClassifier(estimators=models, voting='hard')
    ensemble.fit(X, y)
    dump(ensemble, MODEL_PATH)
    return ensemble

def load_model():
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    return train_and_save_model()

def predict_with_model(model, input_data, prev_inputs):
    history_inputs = prev_inputs[-(FEATURE_WINDOW-1):] + [input_data]
    features = get_window_features(history_inputs)
    X = np.array([features])
    return model.predict(X)[0], features

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
    X, y = [], []
    for i in range(len(df)):
        history_inputs = []
        for j in range(FEATURE_WINDOW):
            if i-j < 0:
                continue
            nums = [int(x) for x in df.iloc[i-j]["input"].split()]
            history_inputs.insert(0, nums)
        X.append(get_window_features(history_inputs))
        y.append(df.iloc[i]["actual"])
    models = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("xgb", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")),
        ("mlp", MLPClassifier(max_iter=2000)),
        ("lr", LogisticRegression(max_iter=1000))
    ]
    ensemble = VotingClassifier(estimators=models, voting='hard')
    ensemble.fit(X, y)
    dump(ensemble, MODEL_PATH)
    return ensemble

def train_bao_model():
    df = fetch_history(2000)
    if df.empty or len(df) < 20:
        return None
    X, y = [], []
    for i in range(len(df)):
        history_inputs = []
        for j in range(FEATURE_WINDOW):
            if i-j < 0:
                continue
            nums = [int(x) for x in df.iloc[i-j]["input"].split()]
            history_inputs.insert(0, nums)
        X.append(get_window_features(history_inputs))
        label = is_bao([int(x) for x in df.iloc[i]["input"].split()])
        y.append(label)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    dump(model, BAO_MODEL_PATH)
    return model

def load_bao_model():
    if os.path.exists(BAO_MODEL_PATH):
        return load(BAO_MODEL_PATH)
    return train_bao_model()

def predict_bao_prob(model, input_data, prev_inputs):
    history_inputs = prev_inputs[-(FEATURE_WINDOW-1):] + [input_data]
    features = get_window_features(history_inputs)
    X = np.array([features])
    prob = model.predict_proba(X)[0][1]
    return prob

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
        return ("⚠️ Đã lâu bạn chưa nhập kết quả thực tế vào bot. Kết quả dự đoán chỉ mang tính tham khảo.")
    return ""

def generate_response(prediction, input_text, stats, time_msg, explain_msg="", bao_warn=""):
    nums = list(map(int, input_text.split()))
    total = sum(nums)
    tai_xiu = "Tài" if total >= 11 else "Xỉu"
    chan_le = "Chẵn" if total % 2 == 0 else "Lẻ"
    bao = "🎲 BÃO! Ba số giống nhau!" if len(set(nums)) == 1 else ""
    # Gọn gàng cho thực chiến
    response = (
        f"🎯 {prediction}\n"
        f"🔢 Tổng: {total} ({tai_xiu} - {chan_le})\n"
        f"{bao}\n"
        f"{explain_msg}\n"
        f"{bao_warn}\n"
        f"✔️ Đúng: {stats['correct']} | ❌ Sai: {stats['wrong']} | 🎯 {stats['accuracy']}%"
    )
    if time_msg:
        response += f"\n{time_msg}"
    return response.strip()

def calculate_stats():
    df = fetch_history(50)
    correct = sum(df['prediction'] == df['actual'])
    total = len(df)
    wrong = total - correct
    acc = round(correct / total * 100, 2) if total > 0 else 0
    return {"correct": correct, "wrong": wrong, "accuracy": acc}

def explain_prediction(features, input_data, prev_inputs):
    last_sum = sum(input_data)
    if last_sum >= 11:
        xu_huong = "Tổng cao"
    else:
        xu_huong = "Tổng thấp"
    msg = f"📈 {xu_huong}"
    if prev_inputs:
        prev_tai = sum(1 for nums in prev_inputs if sum(nums) >= 11)
        if prev_tai > len(prev_inputs)//2:
            msg += ", gần đây đa phần ra Tài."
        else:
            msg += ", gần đây đa phần ra Xỉu."
    return msg

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Gửi 3 số kết quả gần nhất để nhận dự đoán (ví dụ: 456 hoặc 4 5 6). Gõ /backup để xuất lịch sử ra file."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    # Chuẩn hóa input: '123' hoặc '1 2 3' đều được
    if re.match(r"^\d{3}$", text):
        numbers = [int(x) for x in text]
        input_str = f"{numbers[0]} {numbers[1]} {numbers[2]}"
    elif re.match(r"^\d+ \d+ \d+$", text):
        numbers = [int(x) for x in text.split()]
        input_str = text
    else:
        await update.message.reply_text("⚠️ Vui lòng nhập 3 số liền nhau (VD: 345) hoặc 3 số cách nhau bằng dấu cách (VD: 3 4 5).")
        return

    # Gán nhãn thực tế cho lượt chơi trước đó (nếu có)
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, input FROM history WHERE actual IS NULL ORDER BY id DESC LIMIT 1")
            last_entry = cur.fetchone()
    if last_entry:
        last_id, last_input = last_entry
        actual_label = label_func(numbers)
        with get_db_conn() as conn2:
            with conn2.cursor() as cur2:
                cur2.execute("UPDATE history SET actual = %s WHERE id = %s", (actual_label, last_id))
                conn2.commit()
        train_and_save_model()
        train_bao_model()

    # Phát hiện đổi thuật toán
    algo_changed = detect_algo_change()
    if algo_changed:
        train_with_recent_data(WINDOW_SIZE * 2)
        train_bao_model()
        await update.message.reply_text(
            f"⚠️ BOT phát hiện tỉ lệ dự đoán đúng giảm mạnh! Game có thể đã đổi thuật toán. "
            f"BOT sẽ tự động học lại với sóng mới!"
        )

    # Lấy các phiên trước cho feature chuỗi
    df_hist = fetch_history(FEATURE_WINDOW-1, with_actual=False)
    prev_inputs = []
    if not df_hist.empty:
        prev_inputs = [[int(n) for n in s.split()] for s in reversed(df_hist["input"].tolist())]

    model = load_model()
    bao_model = load_bao_model()
    input_data = numbers
    if model is not None:
        prediction, features = predict_with_model(model, input_data, prev_inputs)
    else:
        prediction = label_func(input_data)
        features = extract_features(input_data)

    insert_to_db(input_str, prediction, actual=None)
    stats = calculate_stats()
    time_msg = time_diff_message(get_last_play_time())
    explain_msg = explain_prediction(features, input_data, prev_inputs)

    # Dự đoán "bão" phiên tới
    bao_warn = ""
    if bao_model:
        bao_prob = predict_bao_prob(bao_model, input_data, prev_inputs)
        if bao_prob > 0.08:
            bao_warn = f"⚡️ Dự báo: Khả năng ra BÃO phiên tới cao bất thường! (Xác suất ~{bao_prob:.1%})"

    response = generate_response(prediction, input_str, stats, time_msg, explain_msg, bao_warn)
    # Cảnh báo data ít
    if stats['correct'] + stats['wrong'] < 15:
        response += "\n⚠️ Dữ liệu còn ít, chỉ nên tham khảo!"
    await update.message.reply_text(response)

async def backup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000, with_actual=False)
    if df.empty:
        await update.message.reply_text("⚠️ Không có dữ liệu để backup.")
        return
    path = "/tmp/sicbo_history_backup.csv"
    df.to_csv(path, index=False)
    await update.message.reply_document(document=open(path, "rb"), filename="sicbo_history_backup.csv")

def main():
    create_table()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("backup", backup))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
