"""
Eval Agent — 评估 & 优化建议

职责：
1. 接收 work 结果，计算/验证指标（MSE、MAE 等）
2. 分析训练效果，给出参数优化建议
3. 控制迭代深度：判断是否需要继续迭代

输入：AgentState（LangGraph 全局状态）
输出：{"eval": {...}, "next_action": "..."}
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from skills.metric_skill import MetricSkill
from conf.llm import get_llm

logger = logging.getLogger(__name__)


class EvalAgent:
    """
    评估智能体 — 分析训练结果，控制迭代

    工作流程：
      1. 从 work 结果中提取指标
      2. 与历史记录对比，判断趋势（改善/停滞/恶化）
      3. LLM 生成分析和参数调整建议
      4. 决定 next_action（继续迭代 / 结束）
    """

    SYSTEM_PROMPT = """你是时序预测模型评估专家。根据训练/推理结果：
1. 计算/复核关键指标（MSE, MAE, RMSE, MAPE）
2. 对比历史迭代指标，判断改善趋势
3. 给出参数优化建议（learning_rate, batch_size, epochs 等）
4. 判断是否继续迭代（next_action = "work" | "summary"）

输出合法 JSON，包含 eval（指标/分析/建议）和 next_action 字段。"""

    def __init__(self, llm_model: str = "normal"):
        self.metric_skill = MetricSkill()
        self._llm = get_llm(advanced=(llm_model == "advanced"))

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        评估训练结果并给出优化建议，直接修改并返回 state

        Args:
            state: AgentState，包含:
                - agent_data.work: work agent 的训练结果
                - agent_data.history: 历史迭代快照
                - agent_data.agent_params: 编排参数（max_iteration）
                - agent_data.agent_state: 当前状态（iteration）

        Returns:
            更新后的完整 AgentState
        """
        work_result = state.get("agent_data", {}).get("work", {})
        history = state.get("agent_data", {}).get("history", [])
        iteration = state["agent_data"]["agent_state"]["iteration"]
        max_iter = state["agent_data"]["agent_params"].get("max_iteration", 1)

        logger.info(f"[EvalAgent] 第 {iteration} 轮评估，最大迭代 {max_iter}")

        # ---- 1. 提取当前指标 ----
        curr_metrics = self._extract_metrics(work_result)

        # ---- 2. 判断趋势 ----
        trend = self._assess_trend(curr_metrics, history)
        should_continue = (
            iteration < max_iter
            and trend != "worsening"  # 持续恶化时提前结束
        )

        # ---- 3. LLM 生成分析与建议（不可用时走规则） ----
        eval_result = self._generate_eval(
            curr_metrics, history, iteration, trend, should_continue,
        )

        # ---- 4. 写回 state ----
        state["status"] = "success"
        state["agent"] = "eval"
        state["agent_data"]["eval"] = eval_result
        state["agent_data"]["history"].append({
            "iteration": iteration,
            "eval_summary": eval_result.get("summary", ""),
            "metrics": eval_result.get("metrics", {}),
        })
        state["next_action"] = "work" if should_continue else "summary"

        return state

    # -----------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------

    @staticmethod
    def _extract_metrics(work: dict) -> dict:
        """从 work 结果中提取关键指标"""
        metrics = dict(work.get("metrics", {}))

        # 从 loss 序列中提取首尾值反映变化趋势
        for key in ("train_loss", "val_loss"):
            series = metrics.get(key, [])
            if isinstance(series, list) and len(series) > 1:
                metrics[f"{key}_start"] = series[0]
                metrics[f"{key}_end"] = series[-1]
                metrics[f"{key}_delta"] = series[-1] - series[0]

        return metrics

    @staticmethod
    def _assess_trend(curr: dict, history: list[dict]) -> str:
        """
        判断趋势: improving / plateau / worsening
        比较当前 MSE 与历史最优 MSE
        """
        curr_mse = curr.get("mse")
        if curr_mse is None or not history:
            return "improving"  # 首轮默认改善

        best_hist_mse = min(
            h.get("metrics", {}).get("mse", float("inf"))
            for h in history
        )

        if curr_mse < best_hist_mse * 0.95:
            return "improving"
        elif curr_mse > best_hist_mse * 1.05:
            return "worsening"
        return "plateau"

    def _generate_eval(
        self,
        curr_metrics: dict,
        history: list[dict],
        iteration: int,
        trend: str,
        should_continue: bool,
    ) -> dict:
        """生成评估结果，优先用 LLM"""
        if self._llm and history:
            llm_result = self._call_llm(curr_metrics, history, iteration)
            if llm_result:
                return llm_result

        return self._rule_based_eval(curr_metrics, iteration, trend, should_continue)

    def _call_llm(self, curr_metrics: dict, history: list[dict], iteration: int) -> dict | None:
        """调用 LLM 生成评估"""
        try:
            response = self._llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=json.dumps({
                    "current_iteration": iteration,
                    "current_metrics": curr_metrics,
                    "history": history,
                }, ensure_ascii=False)),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            return data.get("eval", data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[EvalAgent] LLM 输出解析失败: {e}")
            return None

    @staticmethod
    def _rule_based_eval(
        curr_metrics: dict,
        iteration: int,
        trend: str,
        should_continue: bool,
    ) -> dict:
        """规则兜底的评估"""
        mse = curr_metrics.get("mse", 0.0)
        mae = curr_metrics.get("mae", 0.0)

        # 根据趋势生成分析和建议
        if trend == "improving":
            analysis = f"第 {iteration} 轮 MSE={mse:.4f}，指标持续改善。"
            adj = {"learning_rate": curr_metrics.get("learning_rate", 0.001) * 0.5}
        elif trend == "plateau":
            analysis = f"第 {iteration} 轮 MSE={mse:.4f}，指标趋于平稳，建议加大调整幅度。"
            adj = {"learning_rate": curr_metrics.get("learning_rate", 0.001) * 0.3}
        else:  # worsening
            analysis = f"第 {iteration} 轮 MSE={mse:.4f}，指标出现恶化，建议检查参数设置。"
            adj = {}

        return {
            "metrics": {"mse": mse, "mae": mae},
            "analysis": analysis,
            "param_adjustments": adj if should_continue else {},
            "summary": f"第 {iteration} 轮评估：MSE={mse:.4f}，MAE={mae:.4f}，{trend}",
        }
