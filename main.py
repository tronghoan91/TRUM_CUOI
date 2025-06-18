# Force Render rebuild for Python 3.13 compatibility

import os
import asyncio
import logging
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import psycopg2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

logging.basicConfig(level=logging.INFO)

# ----- Machine Learning -----

def train_models(X, y):
    rf = RandomForestClassifier()
    xgb = XGBClassifier()
    mlp = MLPClassifier(max_iter=1000)

    rf.fit(X, y)
    xgb.fit(X, y)
    mlp.fit(X, y)

    return rf, xgb, mlp

def vote_predict(models, input_data):
    votes = [model.predict([input_data])[0] for model in models]
    return max(set(votes), key=votes.count)

def extract_features(results):
    return [[int(n) for n in results.split()]]

def insert_to_db(numbers, prediction):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS history (id SERIAL PRIMARY KEY, input TEXT, prediction TEXT, created_at TIMESTAMP DEFAULT NOW())"
    )
    cur.execute("INSERT INTO history (input, prediction) VALUES (%s, %s)", (numbers, prediction))
    conn.commit()
    cur.close()
    conn.close()

def fetch_history(limit=20):
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql("SELECT input, prediction FROM history ORDER BY id DESC LIMIT %s", conn, params=(limit,))
    conn.close()
    return df

def generate_response(prediction, input_text, stats):
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
    return response.strip()

def evaluate_prediction(predicted, actual):
    return predicted == actual

def calculate_stats():
    df = fetch_history(50)
    correct = sum(df['prediction'] == df['input'])
    total = len(df)
    wrong = total - correct
    acc = round(correct / total * 100, 2) if total > 0 else 0
    return {"correct": correct, "wrong": wrong, "accuracy": acc}

# ----- Bot Handlers -----

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Gửi 3 số kết quả gần nhất để nhận dự đoán Tài/Xỉu nhé (VD: 1 3 2)")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not re.match(r"^\d+ \d+ \d+$", text):
        await update.message.reply_text("⚠️ Vui lòng nhập đúng định dạng: 3 số cách nhau bằng khoảng trắng (VD: 1 2 3)")
        return

    features = extract_features(text)
    labels = ["Tài" if sum(x) >= 11 else "Xỉu" for x in features * 50]  # giả lập tập huấn luyện
    X_train, X_test, y_train, y_test = train_test_split(features * 50, labels, test_size=0.2)

    rf, xgb, mlp = train_models(X_train, y_train)
    models = [rf, xgb, mlp]
    prediction = vote_predict(models, features[0])

    insert_to_db(text, prediction)
    stats = calculate_stats()
    response = generate_response(prediction, text, stats)

    await update.message.reply_text(response)

# ----- Main -----

async def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
