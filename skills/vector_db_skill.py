"""
Vector DB Skill — 低层向量数据库操作

职责：
- 封装 Milvus Lite 的 CRUD 操作（插入 / 语义搜索 / 标量过滤 / 删除）
- 对接 backend/RAG/rag_struct.py 中的嵌入模型和 MilvusStore
- 供 RAGSkill（高层语义封装）和 SummaryAgent 调用

依赖：
- backend.RAG.rag_struct._get_embeddings() — 获取嵌入模型（bge-small-zh-v1.5）
- backend.RAG.rag_struct.MilvusStore — 轻量 Milvus 包装
- pymilvus.MilvusClient — 底层 CRUD
"""

import os
import logging
from typing import Any

from pymilvus import MilvusClient, DataType

from backend.RAG.rag_struct import (
    MilvusStore,
    _get_embeddings,
    MILVUS_DB_PATH,
)

logger = logging.getLogger(__name__)

# 默认 collection 名称
DEFAULT_COLLECTION = "rag_struct"

# 元数据字段列表（与 rag_struct.py 中 MilvusStore.from_texts 的 schema 一致）
_META_FIELDS = {
    "source_type": DataType.VARCHAR,
    "script_path": DataType.VARCHAR,
    "script_name": DataType.VARCHAR,
    "entry_py": DataType.VARCHAR,
    "model": DataType.VARCHAR,
    "dataset": DataType.VARCHAR,
    "features": DataType.VARCHAR,
    "seq_len": DataType.VARCHAR,
    "label_len": DataType.VARCHAR,
    "pred_len": DataType.VARCHAR,
    "task_id": DataType.VARCHAR,
}


class VectorDBSkill:
    """
    低层向量数据库操作技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责底层向量数据库操作，对接 Milvus Lite。
    1. insert: 将文本向量化后写入指定 collection
    2. similarity_search: 语义相似度搜索
    3. query_by_metadata: 按元数据字段过滤查询
    4. delete_by_task_id: 按任务 ID 删除记录
    ---
    """

    PROMPT = """底层向量数据库操作，对接 Milvus Lite：
1. insert: 文本向量化后写入 collection
2. similarity_search: 语义相似度搜索
3. query_by_metadata: 按元数据字段过滤查询
4. delete_by_task_id: 按任务 ID 删除记录"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._embeddings = _get_embeddings(model_name)
        self._milvus_path = MILVUS_DB_PATH
        self._store: MilvusStore | None = None
        self._client: MilvusClient | None = None

    # -----------------------------------------------------------
    # 内部懒加载属性
    # -----------------------------------------------------------

    @property
    def store(self) -> MilvusStore:
        """MilvusStore 实例（供 similarity_search 使用）"""
        if self._store is None:
            self._store = MilvusStore(
                embedding_function=self._embeddings,
                db_path=self._milvus_path,
            )
        return self._store

    @property
    def client(self) -> MilvusClient:
        """底层 MilvusClient（供 insert / delete 使用）"""
        if self._client is None:
            os.makedirs(os.path.dirname(self._milvus_path), exist_ok=True)
            self._client = MilvusClient(self._milvus_path)
        return self._client

    # -----------------------------------------------------------
    # Collection 管理
    # -----------------------------------------------------------

    def _ensure_collection(self, collection: str) -> None:
        """
        确保 collection 存在，不存在则按标准 schema 创建
        """
        if self.client.has_collection(collection):
            return

        # 计算向量维度（通过一次 embed_query 测试）
        dim = len(self._embeddings.embed_query("test"))

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        for field_name, field_type in _META_FIELDS.items():
            schema.add_field(field_name=field_name, datatype=field_type, max_length=256)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")

        self.client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"[VectorDBSkill] 创建 collection: {collection}")

    # -----------------------------------------------------------
    # 公共 API
    # -----------------------------------------------------------

    def insert(
        self,
        collection: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        写入一条记录到向量数据库

        流程：提取 text/experience → 向量化 → 合并 metadata → 写入 Milvus

        Args:
            collection: collection 名称
            data: 要存储的数据（其中 experience 或 text 字段会被向量化）
            metadata: 附加元数据（model, dataset, task_id 等过滤字段）

        Returns:
            insert_id: 插入记录的主键 ID
        """
        logger.info(f"[VectorDBSkill] 写入 {collection}: {data.get('task_id', 'unknown')}")

        # 1. 提取待向量化的文本
        text = data.get("experience") or data.get("text", "")

        # 2. 向量化
        vector = self._embeddings.embed_query(text)

        # 3. 构建行数据（需匹配 schema 字段）
        row: dict[str, Any] = {
            "vector": vector,
            "text": text,
        }
        # 合并 metadata（只取 _META_FIELDS 中定义的字段）
        if metadata:
            for k in _META_FIELDS:
                if k in metadata:
                    row[k] = str(metadata[k])
        # 合并 data 中的可识别字段
        for k in _META_FIELDS:
            if k not in row and k in data:
                row[k] = str(data[k])

        # 4. 确保 collection 存在
        self._ensure_collection(collection)

        # 5. 插入
        result = self.client.insert(collection_name=collection, data=[row])
        insert_id = str(result.get("ids", [None])[0]) if result else "unknown"
        logger.info(f"[VectorDBSkill] 插入成功: id={insert_id}")
        return insert_id

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, str] | None = None,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        语义相似度搜索

        内部委托给 MilvusStore.similarity_search()，将 Document 列表转为 dict 列表。

        Args:
            query: 自然语言查询
            top_k: 返回最相似的记录数
            filter: 元数据过滤条件，如 {"model": "DLinear", "dataset": "ETTh1"}
            collection: 搜索的 collection（默认 rag_struct，一般无需指定）

        Returns:
            [
                {
                    "text": str,
                    "metadata": dict,
                    "score": float,
                },
                ...
            ]
        """
        logger.info(f"[VectorDBSkill] 语义搜索: {query[:50]}... (top_k={top_k})")

        docs = self.store.similarity_search(query, k=top_k, filter=filter)
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.pop("score", 0.0),
            })
        return results

    def query_by_metadata(
        self,
        collection: str,
        filter_expr: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        按元数据标量字段过滤查询（非语义搜索）

        Args:
            collection: collection 名称
            filter_expr: Milvus 过滤表达式，如 'model == "DLinear" and dataset == "ETTh1"'
            limit: 最大返回条数

        Returns:
            list[dict] — 匹配的记录
        """
        logger.info(f"[VectorDBSkill] 标量查询 {collection}: {filter_expr}")

        if not self.client.has_collection(collection):
            logger.warning(f"[VectorDBSkill] collection 不存在: {collection}")
            return []

        self.client.load_collection(collection)
        coll_info = self.client.describe_collection(collection)
        all_fields = [f["name"] for f in coll_info.get("fields", [])]

        results = self.client.query(
            collection_name=collection,
            filter=filter_expr,
            limit=limit,
            output_fields=all_fields,
        )
        # 清理输出中的向量和 id
        cleaned = []
        for row in results:
            row.pop("vector", None)
            row.pop("id", None)
            cleaned.append(row)
        return cleaned

    def delete_by_task_id(self, collection: str, task_id: str) -> bool:
        """
        按任务 ID 删除记录

        Args:
            collection: collection 名称
            task_id: 任务 ID

        Returns:
            是否删除成功
        """
        logger.info(f"[VectorDBSkill] 删除 {collection} 中 task_id={task_id}")

        if not self.client.has_collection(collection):
            return False

        self.client.load_collection(collection)
        result = self.client.delete(
            collection_name=collection,
            filter=f'task_id == "{task_id}"',
        )
        return len(result.get("ids", [])) > 0
