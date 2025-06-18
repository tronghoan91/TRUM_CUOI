import os
import asyncio
import logging
import random
import psycopg2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# Thiết lập logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

model_rf = RandomForestClassifier()
model_xgb = XGBClassifier()
model_mlp = MLPClassifier(max_iter=500)

history = []

def extract_features(sequence):
    return [
        sum(sequence),
        max(sequence),
        min(sequence),
        sequence.count(3),
        len(set(sequence)),
        int(sequence == sorted(sequence)),
    ]

def label_result(nums):
    total = sum(nums)
    tai_xiu = "Tài" if total >= 11 else "Xỉu"
    chan_le = "Chẵn" if total % 2 == 0 else "Lẻ"
    is_bao = int(nums[0] == nums[1] == nums[2])
    return tai_xiu, chan_le, is_bao

def prepare_data():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS results (
        id SERIAL PRIMARY KEY,
        num1 INT, num2 INT, num3 INT,
        tai_xiu TEXT, chan_le TEXT, is_bao INT
    )""")
    cur.execute("SELECT num1, num2, num3, tai_xiu, chan_le, is_bao FROM results ORDER BY id DESC LIMIT 1000")
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 20:
        return None, None

    X, y_tai_xiu, y_chan_le, y_bao = [], [], [], []
    for row in rows:
        nums = list(map(int, row[:3]))
        X.append(extract_features(nums))
        y_tai_xiu.append(1 if row[3] == "Tài" else 0)
        y_chan_le.append(1 if row[4] == "Chẵn" else 0)
        y_bao.append(row[5])

    return np.array(X), {
        "tai_xiu": np.array(y_tai_xiu),
        "chan_le": np.array(y_chan_le),
        "bao": np.array(y_bao)
    }

def train_models():
    X, y_dict = prepare_data()
    if X is None:
        return False
    model_rf.fit(X, y_dict["tai_xiu"])
    model_xgb.fit(X, y_dict["chan_le"])
    model_mlp.fit(X, y_dict["bao"])
    return True

def predict_next(nums):
    x = np.array([extract_features(nums)])
    tai = model_rf.predict(x)[0]
    chan = model_xgb.predict(x)[0]
    bao_prob = model_mlp.predict_proba(x)[0][1]
    result = f"Dự đoán: {'Tài' if tai else 'Xỉu'} - {'Chẵn' if chan else 'Lẻ'}\n"
    result += f"✨ Gợi ý: Điểm {'cao' if tai else 'thấp'} + {'số chẵn' if chan else 'số lẻ'}\n"
    if bao_prob > 0.8:
        result += f"🚨 Cảnh báo: Xác suất bão cao ({bao_prob*100:.1f}%)\n"
    return result

def save_result_to_db(nums):
    tai_xiu, chan_le, is_bao = label_result(nums)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("INSERT INTO results (num1, num2, num3, tai_xiu, chan_le, is_bao) VALUES (%s, %s, %s, %s, %s, %s)",
                (nums[0], nums[1], nums[2], tai_xiu, chan_le, is_bao))
    conn.commit()
    conn.close()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Chào mừng bạn đến với bot dự đoán Tài Xỉu Sicbo!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text.replace(" ", "").isdigit():
        await update.message.reply_text("Vui lòng nhập 3 số liền nhau, ví dụ: 1 3 2")
        return
    nums = list(map(int, text.strip().split()))
    if len(nums) != 3:
        await update.message.reply_text("Cần 3 số. Nhập ví dụ: 2 5 6")
        return

    save_result_to_db(nums)
    ok = train_models()
    if not ok:
        await update.message.reply_text("Chưa đủ dữ liệu để huấn luyện. Tiếp tục nhập dữ liệu.")
        return

    result = predict_next(nums)
    await update.message.reply_text(f"📉 Đã ghi nhận kết quả: {nums}\n" + result)

async def main():
    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await app.updater.idle()

if __name__ == "__main__":
    asyncio.run(main())
