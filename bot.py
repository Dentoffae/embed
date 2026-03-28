"""
Telegram-бот с долговременной векторной памятью.

При каждом сообщении бот:
  1. Генерирует эмбеддинг текста (OpenAI text-embedding-3-small).
  2. Ищет топ-10 релевантных воспоминаний в Pinecone.
  3. Формирует ответ через GPT-4o-mini с учётом найденного контекста.
  4. Сохраняет сообщение пользователя и ответ бота в Pinecone.

Запуск:
    python bot.py
"""

# ── Стандартная библиотека ───────────────────────────────────────────────────
import os
import uuid
import logging

# ── Сторонние пакеты ─────────────────────────────────────────────────────────
import telebot
from telebot import types
from openai import OpenAI
from dotenv import load_dotenv

# ── Локальные модули ─────────────────────────────────────────────────────────
from pine import PineconeClient

load_dotenv()

# ── Логирование ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Клиенты ──────────────────────────────────────────────────────────────────
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
    base_url="https://api.proxyapi.ru/openai/v1",
)

bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))

memory = PineconeClient(index_name="bot-memory", dimension=1536)


# ── Клавиатура ───────────────────────────────────────────────────────────────

def main_keyboard() -> types.ReplyKeyboardMarkup:
    """Возвращает постоянную клавиатуру с основными командами."""
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    kb.add(
        types.KeyboardButton("💾 Запомнить"),
        types.KeyboardButton("📋 Все воспоминания"),
        types.KeyboardButton("📊 Статистика"),
        types.KeyboardButton("🗑 Очистить память"),
    )
    return kb


# ── Вспомогательные функции ───────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    Возвращает эмбеддинг текста через OpenAI text-embedding-3-small.

    Параметры:
        text (str): Исходный текст.

    Возвращает:
        list[float]: Вектор размерностью 1536.
    """
    log.debug("get_embedding: %d chars", len(text))
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def save_memory(text: str, source: str = "user") -> None:
    """
    Сохраняет текст как воспоминание в Pinecone.

    Параметры:
        text   (str): Текст для сохранения.
        source (str): Источник записи — 'user', 'assistant' или 'manual'.
    """
    vector_id = str(uuid.uuid4())
    embedding = get_embedding(text)
    memory.upsert([{
        "id": vector_id,
        "values": embedding,
        "metadata": {"text": text, "source": source},
    }])
    log.info("Memory saved [%s] id=%s: %s", source, vector_id, text[:80])


def search_memories(query: str, top_k: int = 10) -> list[str]:
    """
    Ищет топ-N воспоминаний, релевантных запросу.

    Параметры:
        query  (str): Текст запроса.
        top_k  (int): Количество результатов (по умолчанию 10).

    Возвращает:
        list[str]: Список текстов найденных воспоминаний.
    """
    embedding = get_embedding(query)
    matches = memory.search(vector=embedding, top_k=top_k)
    results = [
        m["metadata"]["text"]
        for m in matches
        if "text" in m.get("metadata", {})
    ]
    log.info("Memory search for %r — found %d matches", query[:60], len(results))
    return results


def ask_gpt(user_message: str, memories: list[str]) -> str:
    """
    Формирует ответ GPT-4o-mini с учётом контекста из воспоминаний.

    Параметры:
        user_message (str):  Сообщение пользователя.
        memories     (list): Список релевантных воспоминаний из Pinecone.

    Возвращает:
        str: Текст ответа модели.
    """
    memory_block = ""
    if memories:
        memory_block = (
            "Ниже — информация из памяти, которая может быть полезна:\n"
            + "\n".join(f"- {m}" for m in memories)
            + "\n\n"
        )

    system_prompt = (
        "Ты умный ассистент с долговременной памятью. "
        "Ты запоминаешь всё, что сообщает пользователь, и используешь "
        "эти знания в своих ответах. Отвечай на русском языке.\n\n"
        + memory_block
    )

    log.info("GPT request: %r (memories in context: %d)", user_message[:60], len(memories))
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    reply = response.choices[0].message.content
    log.info("GPT reply: %r", reply[:80])
    return reply


# ── Обработчики команд и кнопок ───────────────────────────────────────────────

@bot.message_handler(commands=["start"])
def handle_start(message: types.Message) -> None:
    user = message.from_user
    log.info("/start from user_id=%s username=%s", user.id, user.username)
    bot.send_message(
        message.chat.id,
        "Привет! Я бот с долговременной памятью.\n\n"
        "Просто пиши мне — я запоминаю всё важное и учитываю это в ответах.\n\n"
        "Используй кнопки внизу или команды:\n"
        "/remember — сохранить что-то в память\n"
        "/all     — показать все воспоминания\n"
        "/stats   — статистика памяти\n"
        "/forget  — очистить всю память",
        reply_markup=main_keyboard(),
    )


@bot.message_handler(commands=["remember"])
@bot.message_handler(func=lambda m: m.text == "💾 Запомнить")
def handle_remember_prompt(message: types.Message) -> None:
    msg = bot.send_message(
        message.chat.id,
        "Что запомнить? Напиши текст:",
        reply_markup=types.ForceReply(),
    )
    bot.register_next_step_handler(msg, handle_remember_input)


def handle_remember_input(message: types.Message) -> None:
    text = message.text.strip()
    if not text:
        bot.send_message(
            message.chat.id,
            "Пустой текст — ничего не сохранено.",
            reply_markup=main_keyboard(),
        )
        return
    log.info("Manual remember from user_id=%s: %s", message.from_user.id, text[:80])
    save_memory(text, source="manual")
    bot.send_message(message.chat.id, f"✅ Запомнил:\n{text}", reply_markup=main_keyboard())


@bot.message_handler(commands=["all"])
@bot.message_handler(func=lambda m: m.text == "📋 Все воспоминания")
def handle_all(message: types.Message) -> None:
    log.info("/all from user_id=%s", message.from_user.id)
    bot.send_message(message.chat.id, "⏳ Загружаю все воспоминания...")

    all_memories = memory.list_all()

    if not all_memories:
        bot.send_message(message.chat.id, "Память пуста.", reply_markup=main_keyboard())
        return

    source_icon = {"user": "👤", "assistant": "🤖", "manual": "📌"}
    lines = []
    for i, item in enumerate(all_memories, 1):
        meta = item.get("metadata", {})
        text = meta.get("text", "—")
        icon = source_icon.get(meta.get("source", ""), "❓")
        lines.append(f"{i}. {icon} {text}")

    chunk = ""
    for line in lines:
        if len(chunk) + len(line) + 1 > 4000:
            bot.send_message(message.chat.id, chunk)
            chunk = ""
        chunk += line + "\n"

    if chunk:
        bot.send_message(message.chat.id, chunk, reply_markup=main_keyboard())


@bot.message_handler(commands=["stats"])
@bot.message_handler(func=lambda m: m.text == "📊 Статистика")
def handle_stats(message: types.Message) -> None:
    stats = memory.stats()
    count = stats.get("total_vector_count", "?")
    log.info("/stats — total_vector_count=%s", count)
    bot.send_message(
        message.chat.id,
        f"🧠 В памяти хранится воспоминаний: *{count}*",
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )


@bot.message_handler(commands=["forget"])
@bot.message_handler(func=lambda m: m.text == "🗑 Очистить память")
def handle_forget(message: types.Message) -> None:
    msg = bot.send_message(
        message.chat.id,
        "⚠️ Уверен? Это удалит ВСЕ воспоминания.\nНапиши *да* для подтверждения:",
        parse_mode="Markdown",
        reply_markup=types.ForceReply(),
    )
    bot.register_next_step_handler(msg, handle_forget_confirm)


def handle_forget_confirm(message: types.Message) -> None:
    if message.text.strip().lower() in ("да", "yes", "д"):
        log.warning("/forget confirmed by user_id=%s — clearing all memory", message.from_user.id)
        memory.delete_all()
        bot.send_message(
            message.chat.id,
            "🗑 Память полностью очищена.",
            reply_markup=main_keyboard(),
        )
    else:
        bot.send_message(
            message.chat.id,
            "Отменено. Память сохранена.",
            reply_markup=main_keyboard(),
        )


# ── Основной обработчик сообщений ─────────────────────────────────────────────

@bot.message_handler(func=lambda m: True)
def handle_message(message: types.Message) -> None:
    user = message.from_user
    user_text = message.text
    log.info("Message from user_id=%s username=%s: %r", user.id, user.username, user_text[:80])

    try:
        memories = search_memories(user_text, top_k=10)
        reply = ask_gpt(user_text, memories)
        save_memory(user_text, source="user")
        save_memory(reply, source="assistant")
        bot.reply_to(message, reply, reply_markup=main_keyboard())

    except Exception as e:
        log.exception("Error handling message from user_id=%s: %s", user.id, e)
        bot.reply_to(
            message,
            "Произошла ошибка. Попробуй ещё раз.",
            reply_markup=main_keyboard(),
        )


# ── Запуск ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Bot started. Polling...")
    bot.infinity_polling()
