import os
import logging
import pandas as pd
import psycopg2
from collections import Counter
from flask import Flask
import threading
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

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

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
if not BOT_TOKEN or not DATABASE_URL:
    raise Exception("Bạn cần set cả BOT_TOKEN và DATABASE_URL ở biến môi trường!")
logging.basicConfig(level=logging.INFO)

def get_db_conn():
    return psycopg2.connect(DATABASE_URL)

def fetch_history(limit=10000):
    with get_db_conn() as conn:
        query = """
        SELECT id, input, prediction, actual, created_at 
        FROM history 
        WHERE actual IS NOT NULL 
        ORDER BY id ASC LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
    return df

def get_history_count():
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM history WHERE actual IS NOT NULL")
            return cur.fetchone()[0]

def bot_smart_reply(df, best_totals, prediction=None):
    total = len(df)
    correct = sum(df['prediction'] == df['actual'])
    wrong = total - correct
    accuracy = round(correct/total*100, 2) if total > 0 else 0
    acc = correct/total if total > 0 else 0
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
    msg_predict = ""
    note = ""
    if total < 10:
        msg_predict = "Chưa đủ dữ liệu để dự đoán chắc chắn."
        note = "Hãy nhập thêm kết quả thực tế để BOT học và phân tích chính xác hơn."
    elif acc < 0.48 or flip_rate > 0.75 or streak <= 2 and acc < 0.52:
        msg_predict = "DỰ ĐOÁN: Phiên này nên nghỉ vì xác suất đúng thấp hoặc cầu nhiễu."
        note = "Thị trường nguy hiểm, accuracy thấp, flip nhiều. Đề nghị quan sát hoặc nghỉ."
    elif streak >= 4:
        msg_predict = f"DỰ ĐOÁN: {last.upper()} (trend rõ, {streak} phiên liên tiếp)"
        note = f"Trend mạnh, có thể theo {last.title()} nhưng kiểm soát vốn!"
    elif prediction:
        msg_predict = f"DỰ ĐOÁN: {prediction}"
        note = "Cầu bình thường, chưa rõ trend mạnh, vào nhẹ hoặc quan sát thêm."
    else:
        msg_predict = "Không rõ trend, nên quan sát thêm!"
        note = "Bạn nên nhập thêm nhiều phiên thực tế để BOT phân tích tốt hơn."
    msg = (
        f"{msg_predict}\n"
        f"Dải tổng nên đánh: {', '.join(str(x) for x in best_totals)}\n"
        f"Số phiên đã ghi nhận: {total} (Đúng: {correct} | Sai: {wrong} | Chính xác: {accuracy}%)\n\n"
        f"Phân tích: {note}"
    )
    return msg

def suggest_best_totals(df):
    if df.empty:
        return [11, 12, 13]
    total_list = [sum(int(x) for x in s.split()) for s in df['input']]
    c = Counter(total_list)
    common = [k for k, v in c.most_common(3)]
    if len(common) < 3:
        common = [11, 12, 13]
    return common

def get_stats_message(df):
    total = len(df)
    correct = sum(df['prediction'] == df['actual'])
    wrong = total - correct
    accuracy = round(correct/total*100, 2) if total > 0 else 0
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
    return (
        f"THỐNG KÊ DỮ LIỆU:\n"
        f"Số phiên đã ghi nhận: {total}\n"
        f"Đúng: {correct} | Sai: {wrong} | Chính xác: {accuracy}%\n"
        f"Chuỗi {last or '-'} hiện tại: {streak} phiên\n"
        f"Tỷ lệ đổi cầu (flip rate): {round(flip_rate*100, 2)}%\n"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df = fetch_history(1000)
    except Exception as e:
        await update.message.reply_text(f"Lỗi kết nối database: {e}")
        return
    best_totals = suggest_best_totals(df)
    prediction = None
    if best_totals:
        prediction = "Tài" if best_totals[0] >= 11 else "Xỉu"
    msg = bot_smart_reply(df, best_totals, prediction)
    await update.message.reply_text(msg)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Xin chào! Đây là Sicbo RealBot.\n"
        "Các lệnh bạn có thể dùng:\n"
        "/start - Xem giới thiệu và lệnh\n"
        "/stats - Xem thống kê dữ liệu, chuỗi thắng/thua, flip rate\n"
        "/count - Đếm tổng số phiên đã nhập và lưu\n"
        "/reset - Reset toàn bộ lịch sử (Cẩn thận, không thể khôi phục)\n"
        "/backup - Xuất file lịch sử ra CSV\n"
        "Hoặc chỉ cần gửi bất kỳ tin nhắn nào để nhận dự đoán, phân tích, tổng nên đánh!"
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df = fetch_history(1000)
        msg = get_stats_message(df)
    except Exception as e:
        msg = f"Lỗi: {e}"
    await update.message.reply_text(msg)

async def count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        count = get_history_count()
        await update.message.reply_text(f"Đã có tổng cộng {count} phiên dữ liệu được lưu.")
    except Exception as e:
        await update.message.reply_text(f"Lỗi: {e}")

async def backup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000)
    if df.empty:
        await update.message.reply_text("Không có dữ liệu để backup.")
        return
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    path = f"/tmp/sicbo_history_backup_{now_str}.csv"
    df.to_csv(path, index=False)
    await update.message.reply_document(document=open(path, "rb"), filename=f"sicbo_history_backup_{now_str}.csv")

reset_confirm = {}

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if reset_confirm.get(user_id):
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM history")
                conn.commit()
        reset_confirm[user_id] = False
        await update.message.reply_text("Đã reset toàn bộ lịch sử!")
    else:
        reset_confirm[user_id] = True
        await update.message.reply_text(
            "Bạn có chắc chắn muốn xóa toàn bộ lịch sử không?\n"
            "Gõ lại /reset lần nữa để xác nhận."
        )

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("count", count))
    app.add_handler(CommandHandler("backup", backup))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
