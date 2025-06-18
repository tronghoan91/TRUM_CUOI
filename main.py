import os
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

BOT_TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.environ.get('PORT', 10000))

logging.basicConfig(level=logging.INFO)

# Xử lý khi gửi /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot đã sẵn sàng! Gửi 3 số liền nhau (ví dụ: 1 2 3).")

# Xử lý input 3 số liền nhau
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    try:
        nums = list(map(int, text.split()))
        if len(nums) != 3:
            raise ValueError
        await update.message.reply_text(f"Đã nhận kết quả: {nums}")
    except ValueError:
        await update.message.reply_text("Vui lòng nhập đúng định dạng: 3 số liền nhau, ví dụ: 1 3 2.")

# Hàm chính khởi tạo bot
async def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    # Chạy webhook trên Render
    await app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        webhook_url=os.getenv("RENDER_EXTERNAL_URL") + f"/{BOT_TOKEN}"
    )

if __name__ == "__main__":
    asyncio.run(main())
