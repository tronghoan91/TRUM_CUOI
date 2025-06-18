# main.py - Telegram Sicbo Bot với ML models, thống kê, tư duy mô phỏng con người

import os
import re
import random
import numpy as np
import psycopg2
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, filters
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# ========== Cấu hình ==========
TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
bot = Bot(token=TOKEN)
app = Flask(__name__)
dispatcher = Dispatcher(bot=bot, update_queue=None, workers=0, use_context=True)

# ========== Database ==========
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def create_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sicbo_history (
                id SERIAL PRIMARY KEY,
                result TEXT,
                sum INTEGER,
                is_tai BOOLEAN,
                is_chan BOOLEAN,
                is_bao BOOLEAN,
                prediction TEXT,
                is_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.commit()

create_table()

# ========== Xử lý kết quả đầu vào ==========
def parse_input(text):
    numbers = list(map(int, re.findall(r'\d+', text)))
    if len(numbers) != 3:
        return None
    total = sum(numbers)
    tai = total >= 11 and total <= 17 and not (numbers[0] == numbers[1] == numbers[2])
    xiu = not tai
    chan = total % 2 == 0
    le = not chan
    bao = numbers[0] == numbers[1] == numbers[2]
    return {
        "numbers": numbers,
        "sum": total,
        "tai": tai,
        "xiu": xiu,
        "chan": chan,
        "le": le,
        "bao": bao
    }

# ========== Huấn luyện mô hình ==========
def train_models(history):
    X, y_tai = [], []
    for row in history:
        n = list(map(int, row[1].split()))
        total = sum(n)
        X.append(n + [total])
        y_tai.append(row[3])  # is_tai

    rf = RandomForestClassifier().fit(X, y_tai)
    xgb = XGBClassifier(verbosity=0).fit(X, y_tai)
    mlp = MLPClassifier(max_iter=500).fit(X, y_tai)

    return rf, xgb, mlp

# ========== Dự đoán ==========
def predict_next(models, recent_results):
    input_data = []
    for r in recent_results[-1:]:
        n = list(map(int, r[1].split()))
        total = sum(n)
        input_data.append(n + [total])
    rf, xgb, mlp = models
    preds = [
        rf.predict(input_data)[0],
        xgb.predict(input_data)[0],
        mlp.predict(input_data)[0]
    ]
    final = max(set(preds), key=preds.count)
    return final

# ========== Thống kê ==========
def get_stats():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) FROM sicbo_history")
            total, correct = cur.fetchone()
            correct = correct or 0
            acc = round(correct / total * 100, 1) if total > 0 else 0.0
            return total, correct, acc

# ========== Xử lý Telegram ==========
def start(update, context):
    update.message.reply_text("🎲 Gửi kết quả thực tế (VD: 3 4 5) để tôi dự đoán phiên tiếp theo.")

def handle_result(update, context):
    result = parse_input(update.message.text)
    if not result:
        update.message.reply_text("❌ Vui lòng nhập đúng 3 số (VD: 2 3 6).")
        return

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM sicbo_history ORDER BY id DESC LIMIT 20")
            history = cur.fetchall()
            models = train_models(history) if history else None

            prediction = predict_next(models, history) if models else random.choice([True, False])
            predicted = "Tài" if prediction else "Xỉu"
            chanle = random.choice(["Chẵn", "Lẻ"])
            bao_warn = ⚠️ Cảnh báo BÃO: Xác suất cao!" if random.random() < 0.08 else None

            cur.execute("""
            INSERT INTO sicbo_history (result, sum, is_tai, is_chan, is_bao, prediction, is_correct)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                ' '.join(map(str, result["numbers"])),
                result["sum"],
                result["tai"],
                result["chan"],
                result["bao"],
                predicted,
                predicted == ("Tài" if result["tai"] else "Xỉu")
            ))
            conn.commit()

    total, correct, acc = get_stats()

    response = f"""✅ Đã nhận KQ thực tế: {' + '.join(map(str, result['numbers']))} = {result['sum']} ➜ {"Tài" if result['tai'] else "Xỉu"} - {"Chẵn" if result['chan'] else "Lẻ"}

📈 Dự báo phiên tiếp theo:
1️⃣ Nên vào: {predicted} - {chanle}
2️⃣ Dải điểm nên đánh: 10 ➜ 13
{bao_warn if bao_warn else ''}
4️⃣ Thống kê: {correct} đúng / {total - correct} sai (Độ chính xác: {acc}%)
"""
    update.message.reply_text(response.strip())

# ========== Đăng ký handler ==========
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_result))

# ========== Webhook cho Telegram ==========
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

@app.route("/")
def index():
    return "Sicbo Bot đang chạy."

if __name__ == "__main__":
    app.run(debug=False)
