import os
import logging
import pandas as pd
import psycopg2
from collections import Counter
from flask import Flask
import threading
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- Flask giữ port tránh sleep ---
def start_flask():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "Sicbo Bot is alive!", 200

    @app.route('/healthz')
    def health():
        return "OK", 200

    app.run(host='0.0.0.0', port=10000)

threading.Thread(target=start_flask, daemon=True).start()
# -----------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
if not BOT_TOKEN or not DATABASE_URL:
    raise Exception("Bạn cần set cả BOT_TOKEN và DATABASE_URL ở biến môi trường!")

logging.basicConfig(level=logging.INFO)

# === Fetch lịch sử từ database cũ ===
def fetch_history(limit=10000):
    with psycopg2.connect(DATABASE_URL) as conn:
        query = """
        SELECT id, input, prediction, actual, created_at 
        FROM history 
        WHERE actual IS NOT NULL 
        ORDER BY id ASC LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
    return df

# === Hàm tính toán thông minh & trả lời ngắn gọn ===
def bot_smart_reply(df, best_totals, prediction=None):
    total = len(df)
    correct = sum(df['prediction'] == df['actual'])
    wrong = total - correct
    accuracy = round(correct/total*100, 2) if total > 0 else 0
    acc = correct/total if total > 0 else 0

    # Phân tích trend
    actuals = df['actual'].tolist()
    streak = 1
    last = actuals[-1] if actuals else None
    for v in reversed(actuals[:-1]):
        if v == last:
            streak += 1
        else:
            break
    flip_count = sum([actuals[i]!=actuals[i-1] for i in range(1, len(actuals))]) if total > 1 else 0
    flip_rate = flip_count/(len(actuals)-1) if total > 1 else 0

    # Quyết định trả lời linh động:
    msg_predict = ""
    note = ""
    if total < 10:
        msg_predict = "Chưa đủ dữ liệu để dự đoán chắc chắn."
        note = "Hãy nhập thêm kết quả thực tế để học và phân tích chính xác hơn."
    elif acc < 0.48 or flip_rate > 0.75 or streak <= 2 and acc < 0.52:
        msg_predict = "DỰ ĐOÁN: Phiên này nên nghỉ vì xác suất đúng thấp hoặc cầu nhiễu."
        note = "Thị trường nguy hiểm. Đề nghị quan sát hoặc nghỉ."
    elif streak >= 4:
        msg_predict = f"DỰ ĐOÁN: {last.upper()} (trend rõ, {streak} phiên liên tiếp)"
        note = f"Trend mạnh, có thể theo {last.title()} nhưng kiểm soát vốn!"
    elif prediction:
        msg_predict = f"DỰ ĐOÁN: {prediction}"
        note = "Cầu bình thường, chưa rõ trend mạnh, vào nhẹ hoặc quan sát thêm."
    else:
        msg_predict = "Không rõ trend, nên quan sát thêm!"
        note = "Bạn nên nhập thêm nhiều phiên thực tế để phân tích tốt hơn."

    msg = (
        f"{msg_predict}\n"
        f"Dải tổng nên đánh: {', '.join(str(x) for x in best_totals)}\n"
        f"Số phiên đã ghi nhận: {total} (Đúng: {correct} | Sai: {wrong} | Chính xác: {accuracy}%)\n\n"
        f"Phân tích: {note}"
    )
    return msg

# === Dự đoán dải tổng nên đánh ===
def suggest_best_totals(df):
    # Thống kê tổng thực tế gần nhất để pick dải có xác suất cao
    if df.empty:
        return [11, 12, 13]
    total_list = [sum(int(x) for x in s.split()) for s in df['input']]
    c = Counter(total_list)
    common = [k for k, v in c.most_common(3)]
    # Nếu ít lịch sử, trả lại default tài/xỉu trung bình
    if len(common) < 3:
        common = [11, 12, 13]
    return common

# === Hàm xử lý message Telegram ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df = fetch_history(1000)
    except Exception as e:
        await update.message.reply_text(f"Lỗi kết nối database: {e}")
        return

    # Dự đoán dải tổng và tài/xỉu
    best_totals = suggest_best_totals(df)
    prediction = None
    if best_totals:
        prediction = "Tài" if best_totals[0] >= 11 else "Xỉu"
    else:
        prediction = None

    msg = bot_smart_reply(df, best_totals, prediction)
    await update.message.reply_text(msg)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Gửi bất kỳ tin nhắn nào để nhận phân tích nhanh, dự đoán, dải tổng nên đánh")

# === MAIN ===
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
