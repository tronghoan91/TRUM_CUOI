import os
import logging
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from joblib import dump, load
import psycopg2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime
import threading
from flask import Flask
import asyncio

# ==== Flask giữ port tránh sleep ====
def start_flask():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "Bot is alive!", 200

    @app.route('/healthz')
    def health():
        return "OK", 200

    app.run(host='0.0.0.0', port=10000)

threading.Thread(target=start_flask, daemon=True).start()
# ====================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/sicbo_model.joblib")
BAO_MODEL_PATH = os.getenv("BAO_MODEL_PATH", "/tmp/bao_model.joblib")
TOTAL_MODEL_PATH = os.getenv("TOTAL_MODEL_PATH", "/tmp/total_model.joblib")

MIN_ACCURACY = 0.5
WINDOW_SIZE = 40
FEATURE_WINDOW = 3  # Số phiên gần nhất dùng làm feature

# --- Tối ưu tốc độ retrain ---
RETRAIN_EVERY_N = 10
retrain_counter = 0

logging.basicConfig(level=logging.INFO)

def get_db_conn():
    return psycopg2.connect(DATABASE_URL)

def create_table():
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Tạo bảng nếu chưa có
            cur.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id SERIAL PRIMARY KEY,
                    input TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            # Thêm cột 'prediction' nếu chưa có
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='history' AND column_name='prediction'
                    ) THEN
                        ALTER TABLE history ADD COLUMN prediction TEXT;
                    END IF;
                END$$;
            """)
            # Thêm cột 'actual' nếu chưa có
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='history' AND column_name='actual'
                    ) THEN
                        ALTER TABLE history ADD COLUMN actual TEXT;
                    END IF;
                END$$;
            """)
            conn.commit()

def insert_to_db(numbers, prediction, actual=None):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO history (input, prediction, actual) VALUES (%s, %s, %s)",
                (numbers, prediction, actual)
            )
            conn.commit()

def fetch_history(limit=500, with_actual=True):
    with get_db_conn() as conn:
        query = "SELECT id, input, prediction, actual, created_at FROM history"
        if with_actual:
            query += " WHERE actual IS NOT NULL"
        query += " ORDER BY id DESC LIMIT %s"
        df = pd.read_sql(query, conn, params=(limit,))
    return df

def extract_features(nums):
    features = []
    features.extend(nums)  # 3 số gốc
    features.append(sum(nums))
    features.append(max(nums))
    features.append(min(nums))
    features.append(np.std(nums))
    features.append(np.mean(nums))
    features.append(1 if len(set(nums)) == 1 else 0)  # bão
    features.append(1 if sum(nums) % 2 == 0 else 0)   # chẵn lẻ
    features.append(1 if sum(nums) >= 11 else 0)      # tài/xỉu
    # Feature bổ sung:
    features.append(nums[0] + nums[1])  # Tổng 2 số đầu
    features.append(nums[1] + nums[2])  # Tổng 2 số cuối
    features.append(abs(nums[0] - nums[2]))  # Độ lệch đầu-cuối
    return features

def get_window_features(history_inputs):
    features = []
    for nums in history_inputs:
        features += extract_features(nums)
    # Fill 0 nếu thiếu data
    for _ in range(FEATURE_WINDOW - len(history_inputs)):
        features += [0] * len(extract_features([0,0,0]))
    return features

def label_func(nums):
    total = sum(nums)
    return "Tài" if total >= 11 else "Xỉu"

def is_bao(nums):
    return int(len(set(nums)) == 1)

def get_total_history_count():
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM history WHERE actual IS NOT NULL")
            return cur.fetchone()[0]

def train_and_save_model():
    total = get_total_history_count()
    df = fetch_history(5000)
    if df.empty:
        return None
    df = df[df["actual"].notnull()]
    if len(df) < 30:
        print("Not enough data to train model. Need at least 30 rows.")
        return None

    X, y = [], []
    for i in range(len(df)):
        history_inputs = []
        for j in range(FEATURE_WINDOW):
            if i-j < 0:
                continue
            nums = [int(x) for x in df.iloc[i-j]["input"].split()]
            history_inputs.insert(0, nums)
        X.append(get_window_features(history_inputs))
        y.append(df.iloc[i]["actual"])

    # Dưới 200 phiên chỉ dùng 3 model mạnh nhất, trên 200 dùng đủ 6 model
    if total < 200:
        models = [
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("xgb", XGBoostClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")),
            ("lr", LogisticRegression(max_iter=1000))
        ]
    else:
        models = [
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("xgb", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")),
            ("lgb", LGBMClassifier(n_estimators=100)),
            ("cat", CatBoostClassifier(verbose=0, iterations=100)),
            ("mlp", MLPClassifier(max_iter=2000)),
            ("lr", LogisticRegression(max_iter=1000))
        ]
    ensemble = VotingClassifier(estimators=models, voting='hard')
    ensemble.fit(X, y)
    dump(ensemble, MODEL_PATH)
    return ensemble

def load_model():
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    return train_and_save_model()

def predict_with_model(model, input_data, prev_inputs):
    history_inputs = prev_inputs[-(FEATURE_WINDOW-1):] + [input_data]
    features = get_window_features(history_inputs)
    X = np.array([features])
    return model.predict(X)[0], features

def detect_algo_change():
    df = fetch_history(WINDOW_SIZE)
    if len(df) < WINDOW_SIZE:
        return False
    acc = sum(df['prediction'] == df['actual']) / WINDOW_SIZE
    return acc < MIN_ACCURACY

def train_with_recent_data(n=100):
    df = fetch_history(n)
    df = df[df["actual"].notnull()]
    if len(df) < 30:
        print("Not enough data for LightGBM!")
        return None
    total = get_total_history_count()
    X, y = [], []
    for i in range(len(df)):
        history_inputs = []
        for j in range(FEATURE_WINDOW):
            if i-j < 0:
                continue
            nums = [int(x) for x in df.iloc[i-j]["input"].split()]
            history_inputs.insert(0, nums)
        X.append(get_window_features(history_inputs))
        y.append(df.iloc[i]["actual"])
    if total < 200:
        models = [
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("xgb", XGBoostClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")),
            ("lr", LogisticRegression(max_iter=1000))
        ]
    else:
        models = [
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("xgb", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")),
            ("lgb", LGBMClassifier(n_estimators=100)),
            ("cat", CatBoostClassifier(verbose=0, iterations=100)),
            ("mlp", MLPClassifier(max_iter=2000)),
            ("lr", LogisticRegression(max_iter=1000))
        ]
    ensemble = VotingClassifier(estimators=models, voting='hard')
    ensemble.fit(X, y)
    dump(ensemble, MODEL_PATH)
    return ensemble

def train_bao_model():
    df = fetch_history(2000)
    if df.empty or len(df) < 30:
        return None
    X, y = [], []
    for i in range(len(df)):
        history_inputs = []
        for j in range(FEATURE_WINDOW):
            if i-j < 0:
                continue
            nums = [int(x) for x in df.iloc[i-j]["input"].split()]
            history_inputs.insert(0, nums)
        X.append(get_window_features(history_inputs))
        label = is_bao([int(x) for x in df.iloc[i]["input"].split()])
        y.append(label)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    dump(model, BAO_MODEL_PATH)
    return model

def load_bao_model():
    if os.path.exists(BAO_MODEL_PATH):
        return load(BAO_MODEL_PATH)
    return train_bao_model()

def predict_bao_prob(model, input_data, prev_inputs):
    history_inputs = prev_inputs[-(FEATURE_WINDOW-1):] + [input_data]
    features = get_window_features(history_inputs)
    X = np.array([features])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    if len(classes) == 2:
        class_idx_1 = np.where(classes == 1)[0][0]
        return proba[class_idx_1]
    elif len(classes) == 1:
        return 1.0 if classes[0] == 1 else 0.0
    else:
        return 0.0

def train_total_model():
    df = fetch_history(2000)
    if df.empty or len(df) < 50:
        return None
    X, y = [], []
    for i in range(len(df)):
        history_inputs = []
        for j in range(FEATURE_WINDOW):
            if i-j < 0: continue
            nums = [int(x) for x in df.iloc[i-j]["input"].split()]
            history_inputs.insert(0, nums)
        X.append(get_window_features(history_inputs))
        label = sum([int(x) for x in df.iloc[i]["input"].split()])
        y.append(label)
    model = RandomForestClassifier(n_estimators=80)
    model.fit(X, y)
    dump(model, TOTAL_MODEL_PATH)
    return model

def load_total_model():
    if os.path.exists(TOTAL_MODEL_PATH):
        return load(TOTAL_MODEL_PATH)
    return train_total_model()

def predict_total_prob(model, input_data, prev_inputs):
    history_inputs = prev_inputs[-(FEATURE_WINDOW-1):] + [input_data]
    features = get_window_features(history_inputs)
    X = np.array([features])
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
    return prob_dict

def suggest_best_totals_any(prob_dict, top_n=3):
    ranked = sorted(prob_dict.keys(), key=lambda x: prob_dict[x], reverse=True)
    return ranked[:top_n]

def get_streak_stats(df, n=5):
    results = (df['prediction'] == df['actual']).tolist()
    if not results:
        return 0, 0, ''
    streak = 1
    last = results[0]
    for res in results[1:]:
        if res == last:
            streak += 1
        else:
            break
    return streak, last, "thắng" if last else "thua"

def get_trend_msg(stats, streak, last, trend, bao_warn):
    if stats['accuracy'] >= 75:
        msg = "🔥 Sóng đang rất đẹp, đừng bỏ lỡ cơ hội!"
    elif stats['accuracy'] >= 62:
        msg = "✅ Cầu đang ổn định, có thể tự tin vào tiền."
    elif stats['accuracy'] >= 55:
        msg = "⚠️ Sóng dao động, nên vào mức vừa phải, tránh all-in!"
    else:
        msg = "⚠️ Sóng nhiễu, hãy cân nhắc quan sát hoặc giảm điểm."
    if streak >= 3 and last:
        msg += f" (Chuỗi thắng {streak} phiên!)"
    if streak >= 3 and not last:
        msg += f" (Chuỗi thua {streak} phiên, nên giảm cược hoặc quan sát!)"
    if trend:
        msg += f" Xu hướng: {trend}."
    if bao_warn:
        msg += " Đặc biệt chú ý khả năng bão!"
    return msg

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Gửi 3 số kết quả gần nhất để nhận dự đoán (ví dụ: 456 hoặc 4 5 6). Gõ /backup để xuất lịch sử ra file."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global retrain_counter
    text = update.message.text.strip()

    # Check nhập nhầm, nhập lặp (so với phiên trước)
    df_hist_check = fetch_history(1, with_actual=False)
    if df_hist_check.shape[0] > 0:
        last_input_str = df_hist_check.iloc[0]["input"]
        if re.match(r"^\d{3}$", text):
            this_input = f"{text[0]} {text[1]} {text[2]}"
        elif re.match(r"^\d+ \d+ \d+$", text):
            this_input = text
        else:
            this_input = ""
        if last_input_str == this_input:
            await update.message.reply_text("⚠️ Bạn vừa nhập kết quả này ở phiên trước. Nếu nhập nhầm, gửi lại đúng kết quả mới!")
            return

    # Chuẩn hóa input: '123' hoặc '1 2 3'
    if re.match(r"^\d{3}$", text):
        numbers = [int(x) for x in text]
        input_str = f"{numbers[0]} {numbers[1]} {numbers[2]}"
    elif re.match(r"^\d+ \d+ \d+$", text):
        numbers = [int(x) for x in text.split()]
        input_str = text
    else:
        await update.message.reply_text("⚠️ Vui lòng nhập 3 số liền nhau (VD: 345) hoặc 3 số cách nhau bằng dấu cách (VD: 3 4 5).")
        return

    # Gán nhãn thực tế cho lượt chơi trước đó (nếu có)
    last_entry = None
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, input FROM history WHERE actual IS NULL ORDER BY id DESC LIMIT 1")
            last_entry = cur.fetchone()
    retrain_needed = False
    if last_entry:
        last_id, last_input = last_entry
        actual_label = label_func(numbers)
        with get_db_conn() as conn2:
            with conn2.cursor() as cur2:
                cur2.execute("UPDATE history SET actual = %s WHERE id = %s", (actual_label, last_id))
                conn2.commit()
        total = get_total_history_count()
        await update.message.reply_text(f"✅ Đã ghi nhận kết quả thực tế phiên mới. Tổng số phiên đã ghi nhận: {total}")
        retrain_counter += 1
        if retrain_counter >= RETRAIN_EVERY_N:
            retrain_needed = True
            retrain_counter = 0

    # Phát hiện đổi thuật toán
    algo_changed = detect_algo_change()
    if algo_changed:
        await update.message.reply_text(
            f"⚠️ BOT phát hiện tỉ lệ dự đoán đúng giảm mạnh! Game có thể đã đổi thuật toán. BOT sẽ tự động học lại sóng mới."
        )
        retrain_needed = True
        retrain_counter = 0

    # Lấy các phiên trước cho feature chuỗi
    df_hist = fetch_history(FEATURE_WINDOW-1, with_actual=False)
    prev_inputs = []
    if not df_hist.empty:
        prev_inputs = [[int(n) for n in s.split()] for s in reversed(df_hist["input"].tolist())]

    model = load_model()
    bao_model = load_bao_model()
    model_total = load_total_model()
    input_data = numbers

    # Lấy xác suất từng tổng từ model multi-class
    prob_dict = predict_total_prob(model_total, input_data, prev_inputs) if model_total else {}
    best_totals = suggest_best_totals_any(prob_dict, top_n=3) if prob_dict else []
    # Dự đoán tài/xỉu, chẵn/lẻ của tổng xác suất cao nhất
    if best_totals:
        top_total = best_totals[0]
        prediction = "Tài" if top_total >= 11 else "Xỉu"
        chan_le = "Chẵn" if top_total % 2 == 0 else "Lẻ"
    else:
        top_total = sum(numbers)
        prediction = "Tài" if top_total >= 11 else "Xỉu"
        chan_le = "Chẵn" if top_total % 2 == 0 else "Lẻ"
        best_totals = [top_total]

    insert_to_db(input_str, prediction, actual=None)
    stats = calculate_stats()
    # Chuỗi thắng/thua và trend ngắn
    df_stats = fetch_history(15)
    streak, last, trend_type = get_streak_stats(df_stats, n=5)
    trend = ""
    if stats['accuracy'] >= 75:
        trend = f"Sóng mạnh về {prediction}-{chan_le}."
    elif stats['accuracy'] >= 62:
        trend = f"Ưu tiên dải {prediction}-{chan_le}."
    elif stats['accuracy'] <= 55:
        trend = "Sóng nhiễu, nên cân nhắc quan sát thêm."

    # Dự báo bão
    bao_warn = ""
    if bao_model and len(set(input_data)) != 1:
        bao_prob = predict_bao_prob(bao_model, input_data, prev_inputs)
        if bao_prob > 0.08:
            bao_warn = "⚡️ Dự báo: Phiên tiếp theo có khả năng xuất hiện BÃO!"

    trend_msg = get_trend_msg(stats, streak, last, trend, bao_warn)

    response = (
        f"🎯 Dự đoán: {prediction} - {chan_le}\n"
        f"🎯 Dải tổng nên đánh: {', '.join(map(str, best_totals))}\n"
        f"✔️ Đúng: {stats['correct']} | ❌ Sai: {stats['wrong']} | 🎯 {stats['accuracy']}%\n"
        f"{trend_msg}"
    )
    if bao_warn and "BÃO" not in trend_msg:
        response += f"\n{bao_warn}"

    await update.message.reply_text(response.strip())

    # ==== RETRAIN SAU KHI ĐÃ TRẢ LỜI USER (nếu cần) ====
    if retrain_needed:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, train_and_save_model)
        await loop.run_in_executor(None, train_bao_model)
        await loop.run_in_executor(None, train_total_model)

async def backup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000, with_actual=False)
    if df.empty:
        await update.message.reply_text("⚠️ Không có dữ liệu để backup.")
        return
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = f"/tmp/sicbo_history_backup_{now_str}.csv"
    df.to_csv(path, index=False)
    await update.message.reply_document(document=open(path, "rb"), filename=f"sicbo_history_backup_{now_str}.csv")

def calculate_stats():
    df = fetch_history(50)
    correct = sum(df['prediction'] == df['actual'])
    total = len(df)
    wrong = total - correct
    acc = round(correct / total * 100, 2) if total > 0 else 0
    return {"correct": correct, "wrong": wrong, "accuracy": acc}

def main():
    create_table()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("backup", backup))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
