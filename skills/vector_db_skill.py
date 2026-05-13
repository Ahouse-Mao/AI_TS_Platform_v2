"""
Vector DB Skill — 向量数据库读写

职责：
- 将训练经验写入 Milvus 向量数据库
- 支持按 collection 管理不同类别的知识

输入：collection (str), data (dict), metadata (dict)
输出：insert_id (str)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VectorDBSkill:
    """
    向量数据库读写技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责管理 Milvus 向量数据库中的训练经验。
    1. 将 summary agent 生成的经验总结向量化并写入指定 collection
    2. 支持按任务 ID、模型、数据集等元数据过滤查询
    3. 维护 collection 的索引以保证检索效率

    Collection 列表：
    - training_experience: 训练任务经验
    - model_checkpoints: 模型检查点元数据
    ---
    """

    PROMPT = """管理 Milvus 向量数据库中的训练经验：
1. 将经验总结向量化并写入指定 collection
2. 支持按任务 ID、模型、数据集等元数据过滤查询
3. 维护 collection 索引以保证检索效率

Collection：training_experience, model_checkpoints"""

    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port

    def insert(
        self,
        collection: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        写入一条记录到向量数据库

        Args:
            collection: collection 名称
            data: 要存储的数据（会被向量化）
            metadata: 附加元数据（用于过滤）

        Returns:
            insert_id: 插入记录的 ID
        """
        logger.info(f"[VectorDBSkill] 写入 {collection}: {data.get('task_id', 'unknown')}")
        # TODO: Milvus insert
        # 1. embedding = self.embedder.encode(data["experience"])
        # 2. self.milvus.insert(collection, vector=embedding, metadata=metadata)
        return "insert_id_001"

    def query(
        self,
        collection: str,
        filter_expr: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """按条件查询"""
        logger.info(f"[VectorDBSkill] 查询 {collection}: {filter_expr}")
        return []

    def delete_by_task_id(self, collection: str, task_id: str) -> bool:
        """按任务 ID 删除"""
        logger.info(f"[VectorDBSkill] 删除 {collection} 中 task_id={task_id}")
        return True
