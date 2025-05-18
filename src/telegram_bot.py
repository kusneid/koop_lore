import os
import asyncio
import logging
from dotenv import load_dotenv
from pathlib import Path

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, CallbackQuery
)

from main import generate_text_qa, generate_text_default

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh = logging.StreamHandler()
sh.setFormatter(fmt)
fh = logging.FileHandler("bot.log", encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(sh)
logger.addHandler(fh)

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

AGENTS = ["Артем", "Влад", "Максим", "Витя", "Ваня", "Костя", "Саня", "Егор"]
user_agents: dict[int, str] = {}
user_methods: dict[int, str] = {}

def agents_keyboard() -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(text=a, callback_data=f"agent:{a}") for a in AGENTS]
    rows = [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    return InlineKeyboardMarkup(inline_keyboard=rows)

def methods_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text="вопрос–ответ(работает чуть более ебланско)", callback_data="method:qa"),
        InlineKeyboardButton(text="дефолт",callback_data="method:default"),
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons])

bottom_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="сменить бота")]],
    resize_keyboard=True
)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_agents.pop(message.from_user.id, None)
    user_methods.pop(message.from_user.id, None)
    await message.answer(
        "выбери с кем будешь общаться:",
        reply_markup=agents_keyboard()
    )

@dp.callback_query(lambda c: c.data and c.data.startswith("agent:"))
async def process_agent_selection(query: CallbackQuery):
    agent = query.data.split(":", 1)[1]
    user_agents[query.from_user.id] = agent
    await query.answer()
    await query.message.delete()
    await query.message.answer(
        f"«{agent}» выбран в качестве тульпы.\nтеперь выбери способ генерации:",
        reply_markup=methods_keyboard()
    )

@dp.callback_query(lambda c: c.data and c.data.startswith("method:"))
async def process_method_selection(query: CallbackQuery):
    method = query.data.split(":", 1)[1]
    user_methods[query.from_user.id] = method
    await query.answer()
    agent = user_agents.get(query.from_user.id)
    text = f"генерация методом «{method}» установлена для «{agent}».\nОтправь любое сообщение."
    await query.message.delete()
    await query.message.answer(text, reply_markup=bottom_kb)

@dp.message()
async def handle_message(message: types.Message):
    text = message.text.strip()
    if text.lower() == "сменить бота":
        return await cmd_start(message)

    if text.startswith("/"):
        return

    user_id = message.from_user.id
    agent = user_agents.get(user_id)
    method = user_methods.get(user_id)
    if not agent:
        return await message.answer(
            "Сначала выберите режим командой /start.",
            reply_markup=agents_keyboard()
        )
    if not method:
        return await message.answer(
            "теперь выбери способ генерации:",
            reply_markup=methods_keyboard()
        )

    u = message.from_user
    username = u.username or f"{u.first_name or ''} {u.last_name or ''}".strip()

    await message.chat.do("typing")
    try:
        if method == "qa":
            response = generate_text_qa(text, agent)
        else:
            response = generate_text_default(text, agent)
    except Exception as e:
        response = f"Ошибка при генерации: {e}"

    await message.answer(response, reply_markup=bottom_kb)

    logger.info(
        "user=%s agent=%s method=%s question=%r answer=%r",
        username, agent, method, text, response
    )

async def main():
    logger.info("Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
