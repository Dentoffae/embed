import os
import time
import logging
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


class PineconeClient:
    """
    Клиент для работы с векторной базой данных Pinecone.

    Поддерживает операции: создание индекса, запись, чтение,
    поиск по сходству, перечисление всех записей и удаление векторов.

    Пример использования:
        client = PineconeClient(index_name="my-index", dimension=1536)
        client.upsert([{"id": "1", "values": [...], "metadata": {"text": "..."}}])
        results = client.search(vector=[...], top_k=5)
    """

    def __init__(self, index_name: str, dimension: int = 1536, metric: str = "cosine"):
        """
        Инициализирует клиент и подключается к индексу Pinecone.
        Если индекс не существует — создаёт его автоматически.

        Параметры:
            index_name (str): Название индекса в Pinecone.
            dimension  (int): Размерность векторов
                              (по умолчанию 1536 — для text-embedding-3-small).
            metric     (str): Метрика схожести: "cosine", "euclidean" или "dotproduct"
                              (по умолчанию "cosine").

        Исключения:
            ValueError: Если переменная окружения PINECONE_KEY не задана.
        """
        api_key = os.getenv("PINECONE_KEY")
        if not api_key:
            raise ValueError("Environment variable PINECONE_KEY is not set")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        existing = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing:
            log.info("[Pinecone] Creating index '%s'...", index_name)
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                log.info("[Pinecone] Waiting for index to be ready...")
                time.sleep(2)
            log.info("[Pinecone] Index '%s' created and ready.", index_name)
        else:
            log.info("[Pinecone] Index '%s' already exists, connecting...", index_name)

        self.index = self.pc.Index(index_name)
        log.info("[Pinecone] Connected to index '%s'.", index_name)

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------

    def upsert(self, vectors: list[dict]) -> dict:
        """
        Записывает или обновляет векторы в индексе.

        Параметры:
            vectors (list[dict]): Список векторов. Каждый элемент — словарь вида:
                {
                    "id":       str,          # уникальный идентификатор
                    "values":   list[float],  # вектор
                    "metadata": dict          # произвольные метаданные (опционально)
                }

        Возвращает:
            dict: Ответ Pinecone с полем "upserted_count".

        Пример:
            client.upsert([
                {"id": "doc-1", "values": [0.1, 0.2, ...], "metadata": {"text": "Привет"}},
            ])
        """
        return self.index.upsert(vectors=vectors)

    # ------------------------------------------------------------------
    # Чтение по ID
    # ------------------------------------------------------------------

    def fetch(self, ids: list[str]) -> dict:
        """
        Читает векторы по их идентификаторам.

        Параметры:
            ids (list[str]): Список идентификаторов векторов.

        Возвращает:
            dict: Словарь вида {"vectors": {"id": {"id", "values", "metadata"}, ...}}.

        Пример:
            result = client.fetch(["doc-1", "doc-2"])
            values = result["vectors"]["doc-1"]["values"]
        """
        return self.index.fetch(ids=ids)

    # ------------------------------------------------------------------
    # Поиск по схожести
    # ------------------------------------------------------------------

    def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
        include_metadata: bool = True,
    ) -> list[dict]:
        """
        Ищет ближайшие векторы по сходству.

        Параметры:
            vector           (list[float]): Вектор-запрос.
            top_k            (int):         Количество возвращаемых результатов
                                            (по умолчанию 10).
            filter           (dict | None): Фильтр по метаданным,
                                            например {"source": "manual"}.
                                            None — без фильтрации.
            include_metadata (bool):        Включать ли метаданные в ответ
                                            (по умолчанию True).

        Возвращает:
            list[dict]: Список совпадений, отсортированных по убыванию схожести.
                Каждый элемент содержит поля: "id", "score", "metadata".

        Пример:
            matches = client.search(vector=[0.1, 0.2, ...], top_k=3)
            for m in matches:
                print(m["score"], m["metadata"]["text"])
        """
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata,
        )
        return response["matches"]

    # ------------------------------------------------------------------
    # Перечисление всех записей
    # ------------------------------------------------------------------

    def list_all(self, limit: int = 500) -> list[dict]:
        """
        Возвращает все сохранённые записи из индекса.

        Параметры:
            limit (int): Максимальное количество возвращаемых записей
                         (по умолчанию 500).

        Возвращает:
            list[dict]: Список словарей с полями "id" и "metadata".

        Пример:
            items = client.list_all()
            for item in items:
                print(item["metadata"]["text"])
        """
        ids: list[str] = []
        for id_batch in self.index.list():
            ids.extend(id_batch)
            if len(ids) >= limit:
                ids = ids[:limit]
                break

        if not ids:
            return []

        result: list[dict] = []
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            fetched = self.index.fetch(ids=batch)
            for vid, data in fetched.get("vectors", {}).items():
                result.append({"id": vid, "metadata": data.get("metadata", {})})
        return result

    # ------------------------------------------------------------------
    # Удаление
    # ------------------------------------------------------------------

    def delete(self, ids: list[str]) -> dict:
        """
        Удаляет векторы по их идентификаторам.

        Параметры:
            ids (list[str]): Список идентификаторов векторов для удаления.

        Возвращает:
            dict: Пустой словарь {} при успешном удалении.

        Пример:
            client.delete(["doc-1", "doc-2"])
        """
        return self.index.delete(ids=ids)

    def delete_all(self) -> dict:
        """
        Удаляет все векторы из индекса (полная очистка).

        Возвращает:
            dict: Пустой словарь {} при успешном удалении.
        """
        return self.index.delete(delete_all=True)

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Возвращает статистику индекса.

        Возвращает:
            dict: Информация об индексе — количество векторов, размерность,
                  заполненность по пространствам имён и пр.

        Пример:
            info = client.stats()
            print(info["total_vector_count"])
        """
        return self.index.describe_index_stats()
