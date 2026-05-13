"""
RAG Skill — 向量数据库检索

职责：
- 将自然语言查询转为向量，在 Milvus 中检索最相似的 K 条历史经验
- 返回检索到的历史任务参数、指标、经验总结

输入：query (str), top_k (int)
输出：list[dict] — 相似历史记录
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RAGSkill:
    """
    检索增强生成技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责从历史训练经验库（Milvus 向量数据库）中检索与当前任务最相关的历史记录。
    检索维度包括：数据集名称、模型名称、预测长度、任务类型等。
    返回最相似的 top_k 条记录，每条包含：参数配置、指标结果、经验总结。
    ---
    """

    PROMPT = """从历史训练经验库中检索与当前任务最相关的历史记录。
检索维度：数据集名称、模型名称、预测长度、任务类型。
返回最相似的 top_k 条记录，每条包含参数配置、指标结果、经验总结。"""

    def __init__(self, collection_name: str = "training_experience"):
        self.collection_name = collection_name

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        检索相似历史任务

        Args:
            query: 自然语言查询（用户意图描述）
            top_k: 返回的最相似记录数

        Returns:
            [
                {
                    "task_id": str,
                    "model_name": str,
                    "dataset": str,
                    "params": dict,
                    "metrics": dict,
                    "experience": str,
                    "similarity": float,
                },
                ...
            ]
        """
        logger.info(f"[RAGSkill] 检索: {query[:50]}... (top_k={top_k})")
        # TODO: 调用 Milvus 向量检索
        # 1. embedding = self.embedder.encode(query)
        # 2. results = self.milvus.search(collection=self.collection_name, vector=embedding, top_k=top_k)
        return []

    def insert(self, doc: dict[str, Any]) -> bool:
        """写入一条经验到向量数据库"""
        logger.info(f"[RAGSkill] 写入经验: {doc.get('task_id', 'unknown')}")
        # TODO: 调用 Milvus 写入
        return True
