import os
import asyncio
import re
import logging
import psycopg2
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data từ DATABASE_URL
async def load_data():
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM history ORDER BY id DESC LIMIT 1000", conn)
    conn.close()
    return df

# Tiền xử lý dữ liệu
async def preprocess(df):
    df[['a', 'b', 'c']] = df['result'].str.extract(r'(\d)(\d)(\d)').astype(int)
    df['sum'] = df[['a', 'b', 'c']].sum(axis=1)
    df['tai_xiu'] = df['sum'].apply(lambda x: 1 if x >= 11 else 0)
    df['chan_le'] = df['sum'] % 2
    return df[['a', 'b', 'c', 'tai_xiu', 'chan_le']]

# Huấn luyện mô hình
async def train_models(df):
    X = df[['a', 'b', 'c']]
    y_tx = df['tai_xiu']
    y_cl = df['chan_le']
    rf = RandomForestClassifier().fit(X, y_tx)
    xgb = XGBClassifier().fit(X, y_tx)
    mlp = MLPClassifier(max_iter=500).fit(X, y_tx)
    return rf, xgb, mlp

# Dự đoán kết quả
async def predict(models, numbers):
    X_input = np.array(numbers).reshape(1, -1)
    votes = [model.predict(X_input)[0] for model in models]
    tai_xiu = round(np.mean(votes))
    chan_le = numbers[0] + numbers[1] + numbers[2]
    chan_le = chan_le % 2
    total = sum(numbers)
    storm_prob = votes.count(tai_xiu) / len(votes)
    return tai_xiu, chan_le, total, storm_prob

# Lưu dữ liệu mới vào database
async def save_result(result):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO history (result) VALUES (%s)", (result,))
    conn.commit()
    conn.close()

# Command start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Gửi kết quả 3 số liền nhau, tôi sẽ dự đoán Tài/Xỉu - Chẵn/Lẻ cho phiên tiếp theo.")

# Xử lý tin nhắn chứa kết quả
async def handle_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    match = re.fullmatch(r"\d{3}", text)
    if not match:
        await update.message.reply_text("Vui lòng nhập đúng định dạng 3 chữ số liền nhau. Ví dụ: 123")
        return

    numbers = [int(d) for d in text]
    await save_result(text)

    df = await load_data()
    df = await preprocess(df)
    models = await train_models(df)
    tai_xiu, chan_le, total, storm_prob = await predict(models, numbers)

    tx_text = "Tài" if tai_xiu else "Xỉu"
    cl_text = "Chẵn" if chan_le == 0 else "Lẻ"
    bao_text = "\n⚠️ Cảnh báo BÃO: Xác suất cao!" if storm_prob > 0.75 and numbers[0] == numbers[1] == numbers[2] else ""

    await update.message.reply_text(
        f"✅ Đã ghi nhận kết quả: {text}\n"
        f"🔮 Dự đoán phiên tới: {tx_text} - {cl_text}\n"
        f"🎯 Tổng: {total} → Dải nên đánh: {max(4, total-1)} - {min(17, total+1)}{bao_text}"
    )

# Main async app
async def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_result))
    await app.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
