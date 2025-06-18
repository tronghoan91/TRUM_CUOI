import os
import json
import logging
import random
import psycopg2
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, MessageHandler, Filters

# ========== ENV ==========
TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")

# ========== DB INIT ==========
def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id SERIAL PRIMARY KEY,
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_result_to_db(result):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("INSERT INTO history (result) VALUES (%s)", (result,))
    conn.commit()
    cur.close()
    conn.close()

def get_recent_results(limit=20):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT result FROM history ORDER BY id DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in rows]

# ========== DATA LOGIC ==========
def extract_features(result_str):
    dice = [int(x) for x in result_str.strip()]
    total = sum(dice)
    tai_xiu = 1 if total >= 11 else 0
    chan_le = 1 if total % 2 == 0 else 0
    bao = 1 if dice[0] == dice[1] == dice[2] else 0
    return dice + [total, tai_xiu, chan_le, bao]

def prepare_training_data(results):
    X, y_tai_xiu, y_chan_le, y_bao = [], [], [], []
    for res in results:
        feats = extract_features(res)
        X.append(feats[:3])  # Dùng 3 xúc xắc làm input
        y_tai_xiu.append(feats[5])
        y_chan_le.append(feats[6])
        y_bao.append(feats[7])
    return np.array(X), np.array(y_tai_xiu), np.array(y_chan_le), np.array(y_bao)

def predict_next(X, y):
    if len(set(y)) == 1:
        return y[0]
    model_rf = RandomForestClassifier().fit(X, y)
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X, y)
    rf_pred = model_rf.predict(X[-1].reshape(1, -1))[0]
    xgb_pred = model_xgb.predict(X[-1].reshape(1, -1))[0]
    return int(round((rf_pred + xgb_pred) / 2))

def suggest_range(dice_predict):
    # Gợi ý dải điểm dựa vào các lần trước
    avg = np.mean([sum([int(x) for x in res]) for res in dice_predict])
    return int(avg - 2), int(avg + 2)

# ========== BOT ==========
app = Flask(__name__)
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot, None, workers=0)

stats = {"total": 0, "correct": 0}

def handle_message(update: Update, context):
    global stats
    user_input = update.message.text.strip()
    if not (len(user_input) == 3 and user_input.isdigit()):
        update.message.reply_text("Vui lòng nhập đúng 3 số kết quả (VD: 234)")
        return

    save_result_to_db(user_input)
    history = get_recent_results()

    if len(history) < 5:
        update.message.reply_text("Cần ít nhất 5 kết quả để dự đoán. Hãy tiếp tục gửi thêm.")
        return

    X, y_tai_xiu, y_chan_le, y_bao = prepare_training_data(history[::-1])

    tai_xiu_pred = predict_next(X, y_tai_xiu)
    chan_le_pred = predict_next(X, y_chan_le)
    bao_pred = predict_next(X, y_bao)

    d_from, d_to = suggest_range(history[:10])

    latest = extract_features(user_input)
    tai_xiu_actual = latest[5]
    chan_le_actual = latest[6]
    bao_actual = latest[7]

    # thống kê đúng/sai
    if stats["total"] >= 1:
        if tai_xiu_pred == tai_xiu_actual:
            stats["correct"] += 1
    stats["total"] += 1
    acc = round(stats["correct"] / stats["total"] * 100, 1)

    msg = f"""✅ Đã nhận KQ thực tế: {user_input}
🔮 Dự báo phiên tiếp theo:
   - Nên vào: {'Tài' if tai_xiu_pred else 'Xỉu'} - {'Chẵn' if chan_le_pred else 'Lẻ'}
   - Dải điểm nên đánh: {d_from}–{d_to}
{"🌪️ Bão: Dễ xảy ra!" if bao_pred else ""}
📊 Dự đoán: {stats['correct']} đúng / {stats['total']} tổng ({acc}%)
"""
    update.message.reply_text(msg.strip())

dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

@app.route("/", methods=["GET"])
def index():
    return "Sicbo Bot is running."

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
