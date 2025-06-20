import os
import logging
import pandas as pd
import psycopg2
from collections import Counter
from flask import Flask
import threading
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime
import re

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

def create_table():
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id SERIAL PRIMARY KEY,
                    input TEXT,
                    actual TEXT,
                    bot_predict TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()

def alter_table_add_column_bot_predict():
    # An toàn, không lỗi nếu đã có cột
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE history ADD COLUMN IF NOT EXISTS bot_predict TEXT;")
            conn.commit()

def fetch_history_all(limit=10000):
    with get_db_conn() as conn:
        query = """
        SELECT id, input, actual, bot_predict, created_at 
        FROM history 
        ORDER BY id ASC LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
    return df

def get_history_count():
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM history")
            return cur.fetchone()[0]

def insert_to_history(input_str, actual, bot_predict=None):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO history (input, actual, bot_predict, created_at) VALUES (%s, %s, %s, %s)",
                (input_str, actual, bot_predict, datetime.now())
            )
            conn.commit()

def analyze_trend_and_predict(df_with_actual):
    note = ""
    prediction = None
    if len(df_with_actual) >= 8:
        actuals = df_with_actual['actual'].tolist()
        streak = 1
        last = actuals[-1]
        for v in reversed(actuals[:-1]):
            if v == last:
                streak += 1
            else:
                break
        flip_count = sum([actuals[i]!=actuals[i-1] for i in range(1, len(actuals))])
        flip_rate = flip_count/(len(actuals)-1) if len(actuals)>1 else 0
        acc = sum(df_with_actual['bot_predict']==df_with_actual['actual'])/len(df_with_actual) if 'bot_predict' in df_with_actual else 0
        # Linh hoạt nhưng kiểm soát rủi ro!
        if acc < 0.48 or flip_rate > 0.8 or (streak <= 1 and acc < 0.5):
            note = "⚠️ Cầu nhiễu, tỉ lệ đúng thấp. Nên nghỉ hoặc chỉ quan sát."
            prediction = None
        elif streak >= 3:
            note = f"🔥 Trend rõ: {last.upper()} {streak} phiên liên tiếp! Nên theo trend này."
            prediction = last
        elif acc >= 0.5 and flip_rate < 0.8:
            note = "💡 Cầu bình thường, có thể vào nhẹ thăm dò theo xác suất gần đây."
            prediction = "Tài" if df_with_actual['actual'].value_counts().get("Tài", 0) >= df_with_actual['actual'].value_counts().get("Xỉu", 0) else "Xỉu"
        else:
            note = "Chưa đủ dữ liệu thực tế để phân tích trend."
            prediction = None
    else:
        note = "Chưa đủ dữ liệu thực tế để phân tích trend."
        prediction = None
    return prediction, note

def suggest_best_totals_by_prediction(df_with_actual, prediction, n_last=40, min_ratio=0.5):
    if prediction not in ("Tài", "Xỉu") or df_with_actual.empty:
        return [], "Không có dự đoán, không nên đánh dải tổng."
    recent = df_with_actual.tail(n_last)
    totals = [sum(int(x) for x in s.split()) for s in recent['input']]
    if prediction == "Tài":
        eligible = [t for t in range(11, 19)]
    else:
        eligible = [t for t in range(3, 11)]
    total_counts = Counter([t for t in totals if t in eligible])
    if not total_counts:
        return [], "Cầu tổng quá nhiễu, không nên đánh tổng phiên này."
    sorted_totals = [t for t, _ in total_counts.most_common()]
    # Cộng xác suất cho tới khi đủ min_ratio
    cumulative, dsum, best = 0, sum(total_counts.values()), []
    for t in sorted_totals:
        cumulative += total_counts[t]
        best.append(t)
        if dsum and cumulative / dsum >= min_ratio:
            break
    if len(best) < 2:
        return [], "⚠️ Cầu tổng đang out trend, không nên đánh tổng phiên này."
    return sorted(best), ""

def reply_summary(df_all, df_with_actual, best_totals, prediction, trend_note, total_note):
    tong = len(df_all)
    df_predict = df_with_actual[df_with_actual['bot_predict'].isin(["Tài", "Xỉu"])]
    so_du_doan = len(df_predict)
    dung = sum(df_predict['bot_predict'] == df_predict['actual'])
    sai = so_du_doan - dung
    tile = round((dung/so_du_doan)*100, 2) if so_du_doan else 0

    msg = (
        f"Số phiên đã lưu: {tong}\n"
        f"Số phiên đã dự đoán (Tài/Xỉu): {so_du_doan} (Đúng: {dung} | Sai: {sai} | Tỉ lệ đúng: {tile}%)\n"
        f"Dự đoán phiên này: {prediction or '-'}\n"
        f"Dải tổng nên đánh: {', '.join(str(x) for x in best_totals) if best_totals else '-'}\n"
        f"{total_note}\n"
        f"{trend_note}"
    )
    return msg

reset_confirm = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    create_table()
    alter_table_add_column_bot_predict()
    m = re.match(r"^(\d{3})$", text)
    m2 = re.match(r"^(\d+)\s+(\d+)\s+(\d+)$", text)
    if m or m2:
        if m:
            numbers = [int(x) for x in m.group(1)]
        else:
            numbers = [int(m2.group(1)), int(m2.group(2)), int(m2.group(3))]
        input_str = f"{numbers[0]} {numbers[1]} {numbers[2]}"
        total = sum(numbers)
        actual = "Tài" if total >= 11 else "Xỉu"

        # Lấy lịch sử trước khi insert
        df_all = fetch_history_all(10000)
        df_with_actual = df_all[df_all['actual'].notnull()]

        # Phân tích trend và quyết định prediction linh động
        prediction, trend_note = analyze_trend_and_predict(df_with_actual)
        # Nếu bot quyết định dự đoán thì ghi vào cột bot_predict
        insert_to_history(input_str, actual, prediction)

        # Thống kê lại sau khi thêm
        df_all = fetch_history_all(10000)
        df_with_actual = df_all[df_all['actual'].notnull()]
        best_totals, total_note = suggest_best_totals_by_prediction(df_with_actual, prediction)
        msg = reply_summary(df_all, df_with_actual, best_totals, prediction, trend_note, total_note)
        await update.message.reply_text(msg)
        return

    await update.message.reply_text(
        "Vui lòng nhập dãy 3 số kết quả thực tế (ví dụ: 4 5 6 hoặc 456)."
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Xin chào! Đây là Sicbo RealBot.\n"
        "Các lệnh bạn có thể dùng:\n"
        "/start - Xem giới thiệu và lệnh\n"
        "/stats - Xem thống kê dữ liệu, trend, chuỗi thắng/thua\n"
        "/count - Đếm tổng số phiên đã nhập\n"
        "/reset - Reset toàn bộ lịch sử (Cẩn thận, không thể khôi phục)\n"
        "/backup - Xuất file lịch sử ra CSV\n"
        "Gửi 3 số kết quả (ví dụ: 3 5 6 hoặc 356), bot sẽ tự tính toán, thống kê, phân tích trend và dự đoán siêu linh động!"
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    create_table()
    alter_table_add_column_bot_predict()
    df_all = fetch_history_all(10000)
    df_with_actual = df_all[df_all['actual'].notnull()]
    prediction, trend_note = analyze_trend_and_predict(df_with_actual)
    best_totals, total_note = suggest_best_totals_by_prediction(df_with_actual, prediction)
    msg = reply_summary(df_all, df_with_actual, best_totals, prediction, trend_note, total_note)
    await update.message.reply_text(msg)

async def backup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    create_table()
    alter_table_add_column_bot_predict()
    df = fetch_history_all(10000)
    if df.empty:
        await update.message.reply_text("Không có dữ liệu để backup.")
        return
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    path = f"/tmp/sicbo_history_backup_{now_str}.csv"
    df.to_csv(path, index=False)
    await update.message.reply_document(document=open(path, "rb"), filename=f"sicbo_history_backup_{now_str}.csv")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if reset_confirm.get(user_id):
        create_table()
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

async def count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    create_table()
    alter_table_add_column_bot_predict()
    count = get_history_count()
    await update.message.reply_text(f"Đã có tổng cộng {count} phiên dữ liệu được lưu.")

def main():
    create_table()
    alter_table_add_column_bot_predict()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("backup", backup))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("count", count))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
