"""
Summary Agent — 经验总结 & RAG 写入

职责：
1. 汇总所有迭代的历史信息
2. 提炼关键经验（最佳参数组合、避免的坑）
3. 将经验写入向量数据库（RAG），供后续任务参考

输入：AgentState（LangGraph 全局状态）
输出：{"summary": {...}, "next_action": "end"}
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from skills.vector_db_skill import VectorDBSkill
from conf.llm import get_llm

logger = logging.getLogger(__name__)


class SummaryAgent:
    """
    总结智能体 — 汇总历史训练经验并写入向量库

    工作流程：
      1. 从 history 中找出最佳指标和参数
      2. LLM 生成经验总结文本
      3. 写入 Milvus 向量数据库供后续 RAG 检索
    """

    SYSTEM_PROMPT = """你是时序预测训练经验总结专家。根据完整训练历史：
1. 汇总所有迭代轮的指标变化（MSE, MAE 等）
2. 提炼最佳参数组合和最优指标
3. 总结关键经验（有效调整 / 无效尝试 / 数据特点）
4. 给出可操作的后续建议

输出合法 JSON，包含 summary（最优指标/参数/经验/建议）和 next_action="end"。"""

    def __init__(self, llm_model: str = "normal"):
        self.vector_db_skill = VectorDBSkill()
        self._llm = get_llm(advanced=(llm_model == "advanced"))

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        总结训练经验并写入知识库，直接修改并返回 state

        Args:
            state: AgentState，包含:
                - agent_data.plan: 初始参数
                - agent_data.history: 所有迭代快照
                - task_id: 任务 ID

        Returns:
            更新后的完整 AgentState
        """
        history = state.get("agent_data", {}).get("history", [])
        plan = state.get("agent_data", {}).get("plan", {})
        task_id = state.get("task_id", "unknown")

        logger.info(f"[SummaryAgent] 总结任务 {task_id}，共 {len(history)} 轮迭代")

        # ---- 1. 从 history 中提取最佳指标 ----
        best_entry = self._find_best(history)
        best_metrics = best_entry.get("metrics", {})
        best_iteration = best_entry.get("iteration", len(history))

        # ---- 2. LLM 生成总结（不可用时走规则逻辑） ----
        summary_data = self._generate_summary(task_id, plan, history, best_metrics, best_iteration)

        # ---- 3. 写入向量数据库 ----
        self._write_to_vector_db(task_id, plan, summary_data)

        # ---- 4. 写回 state ----
        state["status"] = "success"
        state["agent"] = "summary"
        state["agent_data"]["summary"] = summary_data
        state["next_action"] = "end"

        return state

    # -----------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------

    @staticmethod
    def _find_best(history: list[dict]) -> dict:
        """从 history 中按 MSE 升序找出最优条目"""
        if not history:
            return {"iteration": 0, "metrics": {}}
        return min(
            history,
            key=lambda h: h.get("metrics", {}).get("mse", float("inf")),
        )

    def _generate_summary(
        self,
        task_id: str,
        plan: dict,
        history: list[dict],
        best_metrics: dict,
        best_iteration: int,
    ) -> dict:
        """生成经验总结，优先用 LLM，不可用时走规则逻辑"""
        if self._llm and history:
            llm_result = self._call_llm(plan, history)
            if llm_result:
                return llm_result

        # 规则兜底
        return self._rule_based_summary(task_id, plan, history, best_metrics, best_iteration)

    def _call_llm(self, plan: dict, history: list[dict]) -> dict | None:
        """调用 LLM 生成总结，解析失败返回 None"""
        try:
            response = self._llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=json.dumps({"plan": plan, "history": history}, ensure_ascii=False)),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            return data.get("summary", data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[SummaryAgent] LLM 输出解析失败: {e}")
            return None

    @staticmethod
    def _rule_based_summary(
        task_id: str,
        plan: dict,
        history: list[dict],
        best_metrics: dict,
        best_iteration: int,
    ) -> dict:
        """无 LLM 时的规则总结"""
        model_name = plan.get("model_name", "unknown")
        dataset = plan.get("dataset", "unknown")
        best_mse = best_metrics.get("mse", "N/A")
        best_mae = best_metrics.get("mae", "N/A")

        # 从 history 中提取参数变化趋势
        param_changes = []
        for h in history:
            adj = h.get("param_adjustments", {})
            if adj:
                param_changes.append(f"第 {h['iteration']} 轮: {adj}")

        return {
            "task_id": task_id,
            "best_metrics": {
                "mse": best_mse,
                "mae": best_mae,
                "iteration": best_iteration,
            },
            "best_params": plan,
            "experience": (
                f"在 {dataset} 数据集上，{model_name} 模型经过 {len(history)} 轮迭代优化，"
                f"最佳 MSE={best_mse}，MAE={best_mae}。"
                + (f" 参数调整: {'; '.join(param_changes)}" if param_changes else "")
            ),
            "recommendations": [
                "建议初始学习率设为 1e-4",
                "pred_len 较大时适当增加 seq_len",
            ],
        }

    def _write_to_vector_db(self, task_id: str, plan: dict, summary_data: dict) -> None:
        """将经验写入向量数据库"""
        try:
            self.vector_db_skill.insert(
                collection="rag_struct",
                data={
                    "experience": summary_data.get("experience", ""),
                    "model": plan.get("model_name", ""),
                    "dataset": plan.get("dataset", ""),
                    "task_id": task_id,
                },
                metadata={
                    "model": plan.get("model_name", ""),
                    "dataset": plan.get("dataset", ""),
                    "task_id": task_id,
                },
            )
            logger.info(f"[SummaryAgent] 经验已写入向量数据库: task_id={task_id}")
        except Exception as e:
            logger.warning(f"[SummaryAgent] 向量数据库写入失败: {e}")
