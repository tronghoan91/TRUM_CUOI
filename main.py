import os
import json
import logging
import random
import numpy as np
import psycopg2
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# --- Cấu hình ---
TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")

app = Flask(__name__)
application = Application.builder().token(TOKEN).build()

# --- Kết nối PostgreSQL ---
def insert_result(result, prediction, is_correct):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id SERIAL PRIMARY KEY,
                result TEXT,
                prediction TEXT,
                is_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("INSERT INTO history (result, prediction, is_correct) VALUES (%s, %s, %s)",
                    (result, prediction, is_correct))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("DB error:", e)

def get_history_stats():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) FROM history;")
        total, correct = cur.fetchone()
        cur.close()
        conn.close()
        return total, correct or 0
    except:
        return 0, 0

# --- Hàm dự đoán ---
def predict_next(result_history):
    # Dự đoán điểm
    last_15 = result_history[-15:] if len(result_history) >= 15 else result_history
    total_points = [sum(map(int, list(x))) for x in last_15]
    avg_point = np.mean(total_points)
    suggest_range = (max(3, int(avg_point - 2)), min(18, int(avg_point + 2)))

    # Dự đoán tài/xỉu, chẵn/lẻ
    next_tai_xiu = "Tài" if avg_point >= 10.5 else "Xỉu"
    next_chan_le = "Chẵn" if int(avg_point) % 2 == 0 else "Lẻ"

    # Dự đoán bão (3 số giống nhau)
    triple_chance = sum([1 for r in last_15 if len(set(r)) == 1]) / len(last_15)
    is_storm = triple_chance >= 0.2

    return {
        "tai_xiu": next_tai_xiu,
        "chan_le": next_chan_le,
        "range": suggest_range,
        "storm": is_storm
    }

# --- Bot handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if text.isdigit() and len(text) == 3:
        history = context.bot_data.get("history", [])
        history.append(text)
        context.bot_data["history"] = history[-20:]

        prediction = predict_next(history)

        # Giả sử đúng nếu tổng điểm > 10 thì là "Tài"
        sum_now = sum(map(int, list(text)))
        real_tai_xiu = "Tài" if sum_now >= 11 else "Xỉu"
        is_correct = real_tai_xiu == prediction["tai_xiu"]

        insert_result(text, prediction["tai_xiu"], is_correct)
        total, correct = get_history_stats()
        percent = round(correct / total * 100, 2) if total else 0

        response = f"✅ Đã nhận kết quả: {text}\n"
        response += f"📊 Dự đoán phiên tiếp theo:\n"
        response += f"1. Nên vào: {prediction['tai_xiu']} - {prediction['chan_le']}\n"
        response += f"2. Dải điểm nên đánh: {prediction['range'][0]} → {prediction['range'][1]}\n"
        if prediction['storm']:
            response += f"3. ⚠️ Cảnh báo: Có khả năng BÃO (3 số giống nhau)\n"
        response += f"4. Tổng phiên đã dự đoán: {correct}/{total} đúng ({percent}%)"

        await update.message.reply_text(response)
    else:
        await update.message.reply_text("❗ Hãy nhập 3 chữ số kết quả (ví dụ: 234) để dự đoán phiên tiếp theo.")

# --- Đăng webhook ---
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update_data = request.get_json(force=True)
    update = Update.de_json(update_data, application.bot)
    application.update_queue.put_nowait(update)
    return "ok"

@app.route("/")
def home():
    return "Bot Sicbo Online đang chạy..."

if __name__ == '__main__':
    application.run_polling()  # Dành cho local test
