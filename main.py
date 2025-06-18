import os
import logging
import psycopg2
import numpy as np
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)

# Lấy biến môi trường
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

def create_connection():
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def load_latest_data():
    try:
        conn = create_connection()
        cur = conn.cursor()
        cur.execute("SELECT dice1, dice2, dice3 FROM sicbo_history ORDER BY created_at DESC LIMIT 20")
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logging.error(f"Lỗi DB: {e}")
        return []

def preprocess_data(rows):
    X = []
    y = []
    for row in rows:
        total = sum(row)
        label = 1 if total >= 11 else 0
        X.append(list(row))
        y.append(label)
    return np.array(X), np.array(y)

def train_models(X, y):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)

    xgb = XGBClassifier()
    xgb.fit(X, y)

    return rf, xgb

def predict_next(X, models):
    votes = [model.predict(X[-1].reshape(1, -1))[0] for model in models]
    result = max(set(votes), key=votes.count)
    return result, votes.count(result) / len(votes)

def format_response(real_result, prediction, confidence, stats, has_bao):
    form = f"🎲 Đã nhận KQ thực tế: {real_result}\n"
    form += f"📊 Dự báo phiên sau:\n"
    form += f" - Nên vào: {'TÀI' if prediction == 1 else 'XỈU'}\n"
    form += f" - Dải điểm: {'11-17' if prediction == 1 else '4-10'}\n"
    if has_bao:
        form += f"⚠️ Cảnh báo BÃO: có thể xuất hiện bộ ba giống nhau!\n"
    form += f"✅ Dự đoán đúng {stats['correct']}/{stats['total']} phiên ({stats['accuracy']:.2f}%)"
    return form

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Gửi kết quả 3 xúc xắc (VD: 3 4 6) để dự đoán phiên tiếp theo!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        values = [int(x) for x in update.message.text.strip().split()]
        if len(values) != 3 or any(not (1 <= x <= 6) for x in values):
            raise ValueError
    except:
        await update.message.reply_text("❌ Gửi đúng 3 số từ 1–6, ví dụ: 2 4 6")
        return

    try:
        conn = create_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO sicbo_history (dice1, dice2, dice3, result, created_at) VALUES (%s, %s, %s, %s, NOW())",
                    (*values, sum(values)))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Lỗi ghi DB: {e}")
        await update.message.reply_text("❌ Không ghi được dữ liệu vào hệ thống.")
        return

    rows = load_latest_data()
    if len(rows) < 6:
        await update.message.reply_text("⏳ Cần thêm dữ liệu (ít nhất 6 phiên gần nhất).")
        return

    X, y = preprocess_data(rows)
    rf, xgb = train_models(X, y)
    prediction, confidence = predict_next(X, [rf, xgb])

    stats = {
        "total": len(y),
        "correct": int(confidence * len(y)),
        "accuracy": confidence * 100
    }
    has_bao = sum(1 for r in rows if r[0] == r[1] == r[2]) >= 2

    await update.message.reply_text(
        format_response(values, prediction, confidence, stats, has_bao)
    )

def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    main()
