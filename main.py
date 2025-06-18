import os
import logging
import asyncio
from telegram import Update
from telegram.ext import (
    Application, ContextTypes,
    CommandHandler, MessageHandler, filters
)

# Cấu hình Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.environ.get("PORT", 10000))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Chào mừng bạn đến với bot dự đoán Sicbo! Gửi 3 số liền nhau (VD: 2 5 3).")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    try:
        nums = list(map(int, text.split()))
        if len(nums) != 3 or not all(1 <= n <= 6 for n in nums):
            raise ValueError("Phải là 3 số từ 1 đến 6.")
    except Exception:
        await update.message.reply_text("Vui lòng gửi đúng định dạng 3 số từ 1 đến 6, VD: 1 3 2")
        return

    # TODO: Gọi mô hình tính toán tại đây
    # Kết quả giả lập:
    result = f"""✅ Kết quả bạn gửi: {nums}
👉 Dự đoán phiên tới:
- Tài/Xỉu: Tài
- Chẵn/Lẻ: Lẻ
- Dải điểm nên đánh: 10-12
- 📊 Xác suất bão: 3.5%"""
    await update.message.reply_text(result)


async def main():
    application = Application.builder().token(BOT_TOKEN).build()

    # Thêm handler
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Khởi động server trên Render (Web Service)
    await application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        webhook_url=f"https://{os.environ['RENDER_EXTERNAL_URL'].strip('/')}/"
    )


if __name__ == "__main__":
    asyncio.run(main())
