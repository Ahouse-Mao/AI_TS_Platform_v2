"""
Summary Agent — 经验总结 & RAG 写入

职责：
1. 汇总所有迭代的历史信息
2. 提炼关键经验（最佳参数组合、避免的坑）
3. 将经验写入向量数据库（RAG），供后续任务参考

输入：AgentState（LangGraph 全局状态）
输出：{"summary": {...}, "next_action": "end"}
"""

import logging
from typing import Any

from skills.vector_db_skill import VectorDBSkill

logger = logging.getLogger(__name__)


class SummaryAgent:
    """
    总结智能体

    System Prompt（供 LLM 调用时使用）:
    ---
    你是时序预测训练经验总结专家。根据完整的训练历史，完成以下工作：
    1. 汇总所有迭代轮的指标变化（MSE、MAE 等）
    2. 提炼最佳参数组合和对应的最优指标
    3. 总结关键经验：
       - 哪些参数调整带来了显著改善
       - 哪些尝试无效甚至有负面影响
       - 模型在该数据集上的表现特点
    4. 将总结经验格式化并写入向量数据库，供后续 RAG 检索

    输入历史数据示例：
    {
      "history": [
        { "iteration": 1, "metrics": {"mse": 0.20, "mae": 0.30}, "param_adjustments": {} },
        { "iteration": 2, "metrics": {"mse": 0.15, "mae": 0.25}, "param_adjustments": {"learning_rate": 5e-5} },
      ],
      "plan": { "model_name": "DLinear", "dataset": "ETTh1", "seq_len": 96, "pred_len": 96 }
    }

    输出 JSON 格式：
    {
      "summary": {
        "task_id": "...",
        "best_metrics": { "mse": ..., "mae": ..., "iteration": ... },
        "best_params": { ... },
        "experience": "在 ETTh1 数据集上，DLinear 模型...",
        "recommendations": ["建议初始学习率设为 1e-4", "pred_len=96 时 seq_len 至少为 96"]
      },
      "next_action": "end"
    }
    ---
    """

    SYSTEM_PROMPT = """你是时序预测训练经验总结专家。根据完整训练历史：
1. 汇总所有迭代轮的指标变化
2. 提炼最佳参数组合和最优指标
3. 总结关键经验（有效调整 / 无效尝试 / 数据特点）
4. 将经验写入向量数据库供 RAG 检索

输出合法 JSON，包含 summary（最优指标/参数/经验/建议）和 next_action="end"。"""

    def __init__(self):
        self.vector_db_skill = VectorDBSkill()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        总结训练经验并写入知识库，直接修改并返回 state

        Args:
            state: AgentState，包含:
                - agent_data.plan: 初始参数
                - agent_data.history: 所有迭代快照
                - agent_data.work: 最后一轮 work 结果
                - agent_data.eval: 最后一轮 eval 结果
                - task_id: 任务 ID

        Returns:
            更新后的完整 AgentState
        """
        history = state.get("agent_data", {}).get("history", [])
        plan = state.get("agent_data", {}).get("plan", {})
        task_id = state.get("task_id", "unknown")

        logger.info(f"[SummaryAgent] 总结任务 {task_id}，共 {len(history)} 轮迭代")

        # TODO: 实际逻辑
        # 1. LLM 总结所有历史
        # summary = llm.invoke(system_prompt=self.SYSTEM_PROMPT, context={"history": history, "plan": plan})
        # 2. 写入向量数据库
        # self.vector_db_skill.insert(...)

        best_mse = min(
            (h.get("metrics", {}).get("mse", float("inf")) for h in history),
            default=0.15,
        )

        # ---- 占位：直接修改 state ----
        state["status"] = "success"
        state["agent"] = "summary"
        state["agent_data"]["summary"] = {
            "task_id": task_id,
            "best_metrics": {
                "mse": best_mse,
                "mae": 0.25,
                "iteration": len(history),
            },
            "best_params": plan,
            "experience": (
                f"在 {plan.get('dataset', 'unknown')} 数据集上，"
                f"{plan.get('model_name', 'unknown')} 模型经过 {len(history)} 轮迭代优化，"
                f"最终 MSE={best_mse}。"
            ),
            "recommendations": [
                "建议初始学习率设为 1e-4",
                "pred_len 较大时适当增加 seq_len",
                "定期使用 eval agent 检查过拟合",
            ],
        }
        state["next_action"] = "end"

        return state
