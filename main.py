import os
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

BOT_TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.environ.get('PORT', 10000))
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")  # Render sẽ tự cấp URL này

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot sẵn sàng. Nhập 3 số liền nhau (ví dụ: 1 2 3)")

# Nhận input
async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        nums = list(map(int, update.message.text.strip().split()))
        if len(nums) != 3:
            raise ValueError
        await update.message.reply_text(f"Đã nhận: {nums}")
    except:
        await update.message.reply_text("Nhập sai định dạng. Vui lòng gửi 3 số (VD: 2 4 6)")

# Hàm main
async def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_input))

    # ✅ Bắt buộc phải dùng webhook để tránh lỗi Updater
    await app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        webhook_url=f"{RENDER_URL}/{BOT_TOKEN}",  # Bắt buộc
    )

if __name__ == "__main__":
    asyncio.run(main())
