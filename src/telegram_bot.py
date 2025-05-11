import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from main import generate_text

API_TOKEN = "7478237535:AAF07fia_NqD-VPAa5NotxbV50tpyGcX4ww"
AGENT = "Артем"

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(f"Бот активен. Общение ведется от имени: {AGENT}")

@dp.message(F.text)
async def handle_message(message: types.Message):
    prompt = message.text
    response = generate_text(prompt, AGENT)
    await message.answer(response)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
