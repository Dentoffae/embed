"""
Скрипт загрузки тестовых данных в Pinecone.

Генерирует эмбеддинги для 100 фраз об авиации (50 о Boeing 747
и 50 об Airbus A380) и записывает их в индекс 'nemo'.
После загрузки выполняет тестовый семантический поиск.

Запуск:
    python test.py
"""

import os
import logging

from dotenv import load_dotenv
from openai import OpenAI

from pine import PineconeClient
from phrases import all_facts

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Клиенты ─────────────────────────────────────────────────────────────────

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
    base_url="https://api.proxyapi.ru/openai/v1",
)

pine_client = PineconeClient(index_name="nemo")

# ── Вспомогательные функции ──────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Возвращает эмбеддинг текста через OpenAI text-embedding-3-small."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding

# ── Загрузка данных ──────────────────────────────────────────────────────────

log.info("Starting upload of %d phrases...", len(all_facts))

for idx, phrase in enumerate(all_facts, 1):
    vector = {
        "id": str(idx),
        "values": get_embedding(phrase),
        "metadata": {"text": phrase},
    }
    pine_client.upsert([vector])
    log.info("[%d/%d] Uploaded: %s", idx, len(all_facts), phrase[:70])

log.info("Upload complete. Total phrases: %d", len(all_facts))

# ── Тестовый поиск ───────────────────────────────────────────────────────────

query = "Какой максимальный взлётный вес у Boeing 747?"
log.info("Test search: %r", query)

results = pine_client.search(get_embedding(query), top_k=3)

print("\nSearch results:")
for r in results:
    print(f"  score={r['score']:.4f} | {r['metadata']['text']}")
