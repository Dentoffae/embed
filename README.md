# Telegram-бот с долговременной векторной памятью

Умный Telegram-бот на базе GPT-4o-mini с долговременной памятью через векторную базу данных Pinecone. Бот запоминает всё, что ему говорит пользователь, и перед каждым ответом автоматически извлекает из базы 10 наиболее релевантных воспоминаний, чтобы отвечать с учётом контекста.

---

## Структура проекта

```
EMBED2/
├── bot.py          # Telegram-бот (основной файл)
├── pine.py         # Клиент Pinecone (обёртка над SDK)
├── phrases.py      # 100 фактов о Boeing 747 и Airbus A380 (тестовые данные)
├── test.py         # Скрипт загрузки тестовых данных в Pinecone
├── .env            # Секретные ключи (не коммитить!)
├── .env.example    # Шаблон переменных окружения
├── requirements.txt
└── bot.log         # Лог-файл (создаётся автоматически)
```

---

## Как это работает

```
Пользователь пишет сообщение
        │
        ▼
get_embedding(текст)  ──►  OpenAI text-embedding-3-small  ──►  вектор [1536 чисел]
        │
        ▼
search_memories(вектор, top_k=10)  ──►  Pinecone  ──►  10 похожих воспоминаний
        │
        ▼
ask_gpt(сообщение + воспоминания)  ──►  GPT-4o-mini  ──►  ответ
        │
        ▼
save_memory(сообщение пользователя)  ──►  Pinecone
save_memory(ответ бота)              ──►  Pinecone
        │
        ▼
Ответ пользователю
```

Каждый вектор в Pinecone хранит:
- `id` — случайный UUID
- `values` — эмбеддинг текста (1536 чисел float)
- `metadata.text` — исходный текст
- `metadata.source` — источник: `user`, `assistant` или `manual`

---

## Требования

- Python 3.11+
- Аккаунт [OpenAI](https://platform.openai.com) или [proxyapi.ru](https://proxyapi.ru)
- Аккаунт [Pinecone](https://app.pinecone.io) (бесплатный тариф достаточен)
- Telegram-бот (создать через [@BotFather](https://t.me/BotFather))

---

## Установка

### 1. Клонировать / распаковать проект

```bash
cd EMBED2
```

### 2. Создать виртуальное окружение

```bash
python -m venv venv
```

Активировать:

- **Windows:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Настроить переменные окружения

Скопируй `.env.example` в `.env` и заполни значения:

```bash
cp .env.example .env
```

`.env`:
```
OPENAI_KEY=sk-...           # Ключ OpenAI или proxyapi.ru
PINECONE_KEY=pcsk_...       # Ключ Pinecone
TELEGRAM_BOT_TOKEN=...      # Токен бота от @BotFather
```

> **Получить ключ Pinecone:** [app.pinecone.io](https://app.pinecone.io) → API Keys  
> **Получить токен бота:** написать [@BotFather](https://t.me/BotFather) → `/newbot`

---

## Запуск бота

```bash
.\venv\Scripts\python.exe bot.py   # Windows
python bot.py                       # macOS / Linux
```

При первом запуске бот автоматически создаст индекс `bot-memory` в Pinecone и дождётся его готовности.

Вывод в терминале:
```
[Pinecone] Creating index 'bot-memory'...
[Pinecone] Index 'bot-memory' created and ready.
[Pinecone] Connected to index 'bot-memory'.
2026-03-28 10:00:00 [INFO] Bot started. Polling...
```

---

## Команды бота

| Команда / Кнопка | Описание |
|---|---|
| `/start` | Приветствие и вывод клавиатуры |
| `/remember` или 💾 **Запомнить** | Бот просит написать текст и сохраняет его в память |
| `/all` или 📋 **Все воспоминания** | Выводит все записи из базы с иконками по источнику |
| `/stats` или 📊 **Статистика** | Показывает общее количество воспоминаний |
| `/forget` или 🗑 **Очистить память** | Запрашивает подтверждение, затем удаляет все векторы |
| Любое сообщение | Поиск релевантных воспоминаний → ответ GPT → сохранение |

Иконки источников в выводе `/all`:

| Иконка | Источник |
|---|---|
| 👤 | Сообщение пользователя |
| 🤖 | Ответ ассистента |
| 📌 | Сохранено вручную через «Запомнить» |

---

## Модули

### `pine.py` — PineconeClient

Обёртка над Pinecone SDK. Создаётся один раз при запуске бота.

```python
from pine import PineconeClient

client = PineconeClient(index_name="my-index", dimension=1536, metric="cosine")
```

| Метод | Описание |
|---|---|
| `upsert(vectors)` | Записать / обновить векторы |
| `fetch(ids)` | Прочитать векторы по ID |
| `search(vector, top_k, filter)` | Семантический поиск по сходству |
| `list_all(limit)` | Получить все записи (до `limit` штук) |
| `delete(ids)` | Удалить векторы по ID |
| `delete_all()` | Очистить весь индекс |
| `stats()` | Статистика индекса |

### `bot.py` — Telegram-бот

| Функция | Описание |
|---|---|
| `get_embedding(text)` | Получить вектор текста через OpenAI |
| `save_memory(text, source)` | Сохранить воспоминание в Pinecone |
| `search_memories(query, top_k)` | Найти топ-N релевантных воспоминаний |
| `ask_gpt(message, memories)` | Сформировать ответ GPT с контекстом |
| `main_keyboard()` | Вернуть объект клавиатуры Telegram |

### `phrases.py` — Тестовые данные

Содержит 100 фактов о самолётах:
- `boeing_747_facts` — 50 фактов о Boeing 747
- `airbus_a380_facts` — 50 фактов об Airbus A380
- `all_facts` — объединённый список

Используется в `test.py` для загрузки тестовых данных в Pinecone.

---

## Загрузка тестовых данных

Чтобы загрузить 100 фактов об авиации в индекс `nemo`:

```bash
python test.py
```

Скрипт создаст индекс, сгенерирует эмбеддинги для каждой фразы и загрузит их в Pinecone. После этого можно задавать боту вопросы об авиации — он будет находить релевантные факты из базы.

---

## Логирование

Бот пишет логи одновременно в терминал и в файл `bot.log`.

Формат строки:
```
2026-03-28 10:45:12 [INFO] Message from user_id=123 username=olgah: 'Привет!'
2026-03-28 10:45:12 [INFO] Memory search for 'Привет!' — found 5 matches
2026-03-28 10:45:13 [INFO] GPT request: 'Привет!' (memories in context: 5)
2026-03-28 10:45:14 [INFO] GPT reply: 'Привет! Чем могу помочь?'
2026-03-28 10:45:14 [INFO] Memory saved [user] id=uuid-...: Привет!
```

| Уровень | Когда используется |
|---|---|
| `DEBUG` | Внутренние вызовы эмбеддингов |
| `INFO` | Входящие сообщения, поиск, запросы к GPT, сохранение |
| `WARNING` | Очистка памяти (`/forget`) |
| `ERROR` | Исключения при обработке сообщений |

---

## Зависимости

| Пакет | Назначение |
|---|---|
| `openai` | Эмбеддинги (`text-embedding-3-small`) и генерация текста (`gpt-4o-mini`) |
| `pinecone` | Векторная база данных |
| `pytelegrambotapi` | Telegram Bot API |
| `python-dotenv` | Загрузка переменных из `.env` |

---

## Безопасность

- Файл `.env` содержит секретные ключи — **никогда не публикуй его в репозиторий**
- Добавь `.env` в `.gitignore`
- Команда `/forget` защищена подтверждением словом «да»
