import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, CallbackQuery
)

from main import generate_text

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

AGENTS = ["Артем", "Влад", "Максим", "Витя", "Ваня", "Костя", "Саня", "Егор"]
user_agents: dict[int, str] = {}

def agents_keyboard() -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(text=a, callback_data=f"agent:{a}") for a in AGENTS]
    keyboard = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

bottom_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Сменить бота")]],
    resize_keyboard=True
)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_agents.pop(message.from_user.id, None)
    await message.answer(
        "Привет! Выбери, от лица какого агента будешь общаться:",
        reply_markup=agents_keyboard()
    )

@dp.callback_query(lambda c: c.data and c.data.startswith("agent:"))
async def process_agent_selection(query: CallbackQuery):
    agent = query.data.split(":", 1)[1]
    user_agents[query.from_user.id] = agent
    await query.answer()
    await query.message.delete()
    await query.message.answer(
        f"Теперь общаемся от лица: {agent}.\n\nНапиши что-нибудь, я отвечу от его имени.",
        reply_markup=bottom_kb
    )

@dp.message()
async def handle_message(message: types.Message):
    text = message.text.strip()

    if text == "Сменить бота":
        return await cmd_start(message)

    if text.startswith("/"):
        return

    user_id = message.from_user.id
    agent = user_agents.get(user_id)
    if not agent:
        return await message.answer(
            "Сначала выберите агента командой /start.",
            reply_markup=agents_keyboard()
        )

    await message.chat.do("typing")
    try:
        response = generate_text(text, agent)
    except Exception as e:
        response = f"Ошибка при генерации: {e}"
    await message.answer(response, reply_markup=bottom_kb)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
