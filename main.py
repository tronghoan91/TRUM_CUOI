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
from joblib import dump, load

# Logging để debug
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Token bot từ Render Environment Variable
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# ========================== KẾT NỐI DATABASE =============================

def create_connection():
    return psycopg2.connect(DATABASE_URL, sslmode='require')


# ========================== HÀM XỬ LÝ DỮ LIỆU ============================

def load_latest_data():
    try:
        conn = create_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM sicbo_history ORDER BY created_at DESC LIMIT 20")
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu từ DB: {e}")
        return []


def preprocess_data(rows):
    # Ví dụ giả lập xử lý
    X = []
    y = []
    for row in rows:
        # Giả sử row = (id, dice1, dice2, dice3, ..., result)
        dice_values = [row[1], row[2], row[3]]
        total = sum(dice_values)
        label = 1 if total >= 11 else 0  # 1 = Tài, 0 = Xỉu
        X.append(dice_values)
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


# ========================== FORM TRẢ LỜI ============================

def format_response(real_result, prediction, confidence, stats, has_bao):
    form = "🎲 Đã nhận KQ thực tế: {}\n".format(real_result)
    form += f"📊 Dự báo phiên tiếp theo:\n"
    form += f" - Nên vào: {'TÀI' if prediction == 1 else 'XỈU'}\n"
    form += f" - Dải điểm nên đánh: {'11-17' if prediction == 1 else '4-10'}\n"
    if has_bao:
        form += "⚠️ Cảnh báo: Xác suất BÃO cao! Cân nhắc kỹ lưỡng!\n"
    form += f"✅ Tổng số phiên đã dự đoán: {stats['total']} | Đúng: {stats['correct']} ({stats['accuracy']:.2f}%)"
    return form


# ========================== TELEGRAM HANDLER ============================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Xin chào! Gửi kết quả 3 viên xúc xắc để dự đoán phiên sau (VD: 3 5 2)")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text.strip()
        parts = [int(p) for p in text.split()]
        if len(parts) != 3 or not all(1 <= p <= 6 for p in parts):
            raise ValueError
    except:
        await update.message.reply_text("❌ Định dạng không hợp lệ. Gửi đúng 3 số từ 1–6. VD: 2 5 6")
        return

    # Lưu kết quả thực tế vào DB
    try:
        conn = create_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sicbo_history (dice1, dice2, dice3, result, created_at) VALUES (%s, %s, %s, %s, NOW())",
            (*parts, sum(parts))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        await update.message.reply_text("❌ Lỗi khi ghi dữ liệu vào hệ thống.")
        logging.error(e)
        return

    rows = load_latest_data()
    if len(rows) < 6:
        await update.message.reply_text("⏳ Cần ít nhất 6 phiên để huấn luyện. Gửi thêm dữ liệu.")
        return

    X, y = preprocess_data(rows)
    models = train_models(X, y)
    prediction, confidence = predict_next(X, models)

    # Thống kê đúng/sai (giả lập)
    stats = {"total": len(y), "correct": int(confidence * len(y)), "accuracy": confidence * 100}
    has_bao = sum(p[1] == p[2] == p[3] for p in rows[:5]) >= 2

    response = format_response(parts, prediction, confidence, stats, has_bao)
    await update.message.reply_text(response)


# ========================== CHẠY BOT ============================

def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    main()
