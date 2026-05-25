"""
RAG Skill — 高层语义检索（检索增强生成）

职责：
- 作为语义检索的高层封装，内部依赖 VectorDBSkill 操作向量数据库
- 将自然语言查询转为语义搜索，返回结构化的历史经验
- 供 PlanAgent 调用，用于检索相似历史任务以辅助参数规划

输入：query (str), top_k (int), model/dataset 过滤条件
输出：list[dict] — 格式化的相似历史记录

依赖：
- VectorDBSkill — 低层向量数据库操作
"""

import logging
from typing import Any

from skills.vector_db_skill import VectorDBSkill

logger = logging.getLogger(__name__)


class RAGSkill:
    """
    高层语义检索技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责从历史训练经验库中检索与当前任务最相关的历史记录。
    检索维度包括：数据集名称、模型名称、预测长度、任务类型等。
    返回最相似的 top_k 条记录，每条包含：参数配置、指标结果、经验总结。
    ---
    """

    PROMPT = """从历史训练经验库中检索与当前任务最相关的历史记录。
检索维度：数据集名称、模型名称、预测长度、任务类型。
返回最相似的 top_k 条记录，每条包含参数配置、指标结果、经验总结。"""

    def __init__(
        self,
        vector_db_skill: VectorDBSkill | None = None,
    ):
        """
        Args:
            vector_db_skill: VectorDBSkill 实例（不传则自动创建）
        """
        self._vector_db = vector_db_skill or VectorDBSkill()

    def search(
        self,
        query: str,
        top_k: int = 5,
        model: str | None = None,
        dataset: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        检索相似历史任务

        流程：
        1. 构造可选的元数据过滤条件
        2. 委托 VectorDBSkill.similarity_search() 执行语义搜索
        3. 将原始结果映射为结构化输出（task_id, model_name, params, metrics, experience）

        Args:
            query: 自然语言查询（用户意图描述，如"用 DLinear 预测 ETTh1 的气温"）
            top_k: 返回的最相似记录数
            model: 按模型名称过滤（可选）
            dataset: 按数据集名称过滤（可选）

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

        # 1. 构造过滤条件
        where: dict[str, str] = {}
        if model:
            where["model"] = model
        if dataset:
            where["dataset"] = dataset

        # 2. 执行语义搜索
        raw_results = self._vector_db.similarity_search(
            query=query,
            top_k=top_k,
            filter=where if where else None,
        )

        # 3. 映射为结构化输出
        results: list[dict[str, Any]] = []
        for raw in raw_results:
            meta = raw.get("metadata", {})
            results.append({
                "task_id": meta.get("task_id", ""),
                "model_name": meta.get("model", ""),
                "dataset": meta.get("dataset", ""),
                "params": {
                    "seq_len": meta.get("seq_len", ""),
                    "pred_len": meta.get("pred_len", ""),
                    "features": meta.get("features", ""),
                },
                "metrics": {},
                "experience": raw.get("text", ""),
                "similarity": raw.get("score", 0.0),
            })

        logger.info(f"[RAGSkill] 检索到 {len(results)} 条结果")
        return results
