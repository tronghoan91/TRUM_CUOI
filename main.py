import os
import random
import logging
import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đọc biến môi trường
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Kết nối PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sicbo_results (
            id SERIAL PRIMARY KEY,
            result TEXT,
            prediction TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    cursor.close()
    conn.close()

create_table()

# Lưu kết quả thực tế và dự đoán
def save_result(actual: str, prediction: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sicbo_results (result, prediction) VALUES (%s, %s)",
        (actual, prediction)
    )
    conn.commit()
    cursor.close()
    conn.close()

# Hàm phân tích kết quả từ chuỗi input

def parse_result(text):
    try:
        parts = list(map(int, text.strip().split(" ")))
        if len(parts) != 3 or not all(1 <= x <= 6 for x in parts):
            return None
        return parts
    except:
        return None

# Tính toán dự đoán dựa trên lịch sử

def predict_next():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM sicbo_results ORDER BY id DESC LIMIT 20", conn)
    conn.close()

    if len(df) < 15:
        return "Chưa đủ dữ liệu để dự đoán."

    results = [list(map(int, row.split())) for row in df['result']]
    X = [r[:-1] for r in results[:-1]]
    y = [sum(r) for r in results[1:]]

    models = [
        RandomForestClassifier(n_estimators=50),
        XGBClassifier(n_estimators=50, verbosity=0),
        MLPClassifier(max_iter=300)
    ]
    votes = []
    for model in models:
        try:
            model.fit(X, y)
            pred = model.predict([results[-1][:-1]])[0]
            votes.append(pred)
        except Exception as e:
            logger.warning(f"Model error: {e}")

    final_pred = int(sum(votes) / len(votes)) if votes else random.randint(8, 13)
    tai_xiu = "Tài" if final_pred >= 11 else "Xỉu"
    chan_le = "Chẵn" if final_pred % 2 == 0 else "Lẻ"

    bao = results[-1][0] == results[-1][1] == results[-1][2]
    bao_text = "\n⚠️ Có khả năng BÃO!" if bao else ""

    return f"🎲 Dự đoán tiếp theo:\n- {tai_xiu} - {chan_le}\n- Tổng điểm dự đoán: {final_pred}\n{bao_text}"

# Handler chính
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    result = parse_result(text)
    if result:
        prediction_text = predict_next()
        save_result(" ".join(map(str, result)), prediction_text)
        await update.message.reply_text(
            f"✅ Đã ghi nhận kết quả: {' '.join(map(str, result))}\n{prediction_text}"
        )
    else:
        await update.message.reply_text("Vui lòng nhập kết quả 3 viên xúc xắc, cách nhau bằng dấu cách. Ví dụ: 2 5 6")

# Lệnh bắt đầu
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Gửi kết quả 3 viên xúc xắc (VD: 1 3 6) để bot dự đoán phiên tiếp theo!")

# Khởi tạo bot
async def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await app.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
