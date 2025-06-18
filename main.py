import os
import asyncio
import logging
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kết nối DB
def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id SERIAL PRIMARY KEY,
                input TEXT NOT NULL,
                tai_xiu TEXT,
                chan_le TEXT,
                is_bao BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    conn.close()

# Lưu lịch sử
def save_history(inp, tai_xiu, chan_le, is_bao):
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO history (input, tai_xiu, chan_le, is_bao)
            VALUES (%s, %s, %s, %s)
        """, (inp, tai_xiu, chan_le, is_bao))
        conn.commit()
    conn.close()

# Lấy dữ liệu gần nhất để train
def load_history(n=20):
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql_query(f"""
        SELECT * FROM history ORDER BY created_at DESC LIMIT {n}
    """, conn)
    conn.close()
    return df[::-1]  # đảo ngược để đúng thứ tự thời gian

# Tách đặc trưng từ chuỗi số
def extract_features(text):
    nums = [int(x) for x in text if x.isdigit()]
    if len(nums) != 3:
        raise ValueError("Vui lòng nhập đúng 3 số (vd: 123 hoặc 1 2 3)")
    total = sum(nums)
    features = {
        "total": total,
        "max": max(nums),
        "min": min(nums),
        "unique": len(set(nums)),
        "same": 1 if len(set(nums)) == 1 else 0
    }
    return features

# Dự đoán từ mô hình
def predict_sicbo(input_text):
    df = load_history(100)
    if len(df) < 10:
        return "Chưa đủ dữ liệu để dự đoán. Vui lòng nhập thêm kết quả thực tế."

    X = pd.DataFrame([extract_features(x) for x in df["input"]])
    y_tai_xiu = df["tai_xiu"]
    y_chan_le = df["chan_le"]

    le_tx = LabelEncoder().fit(y_tai_xiu)
    le_cl = LabelEncoder().fit(y_chan_le)

    X_train, X_test, y_tx_train, y_tx_test = train_test_split(X, le_tx.transform(y_tai_xiu), test_size=0.2, random_state=42)
    _, _, y_cl_train, y_cl_test = train_test_split(X, le_cl.transform(y_chan_le), test_size=0.2, random_state=42)

    rf = RandomForestClassifier().fit(X_train, y_tx_train)
    xgb = XGBClassifier().fit(X_train, y_tx_train)
    mlp = MLPClassifier(max_iter=500).fit(X_train, y_tx_train)

    input_feat = pd.DataFrame([extract_features(input_text)])

    # Tài/Xỉu prediction
    preds = [rf.predict(input_feat)[0], xgb.predict(input_feat)[0], mlp.predict(input_feat)[0]]
    vote_tx = max(set(preds), key=preds.count)
    tai_xiu = le_tx.inverse_transform([vote_tx])[0]

    # Chẵn/Lẻ prediction
    rf2 = RandomForestClassifier().fit(X_train, y_cl_train)
    chan_le = le_cl.inverse_transform(rf2.predict(input_feat))[0]

    total = extract_features(input_text)["total"]
    is_bao = extract_features(input_text)["same"] == 1

    # Lưu lịch sử
    save_history(input_text, tai_xiu, chan_le, is_bao)

    return (
        f"🎲 Kết quả dự đoán:\n"
        f"• Tài/Xỉu: {tai_xiu}\n"
        f"• Chẵn/Lẻ: {chan_le}\n"
        f"• Tổng điểm: {total}\n"
        f"{'⚠️ CẢNH BÁO: BÃO (3 số giống nhau)' if is_bao else ''}"
    )

# Nhập kết quả thật
async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().replace(" ", "")
    if not text.isdigit() or len(text) != 3:
        await update.message.reply_text("❌ Vui lòng nhập 3 chữ số liền nhau (VD: 123)")
        return
    result = predict_sicbo(text)
    await update.message.reply_text(result)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Chào mừng đến với bot dự đoán Tài Xỉu!\n"
        "Gửi 3 số kết quả gần nhất (VD: 123 hoặc 4 2 6) để nhận dự đoán phiên tiếp theo."
    )

async def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_input))
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
