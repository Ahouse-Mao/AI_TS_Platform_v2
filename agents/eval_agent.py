"""
Eval Agent — 评估 & 优化建议

职责：
1. 接收 work 结果，计算/验证指标（MSE、MAE 等）
2. 分析训练效果，给出参数优化建议
3. 控制迭代深度：判断是否需要继续迭代

输入：AgentState（LangGraph 全局状态）
输出：{"eval": {...}, "next_action": "..."}
"""

import logging
from typing import Any

from skills.metric_skill import MetricSkill
from skills.visualization_skill import VisualizationSkill

logger = logging.getLogger(__name__)


class EvalAgent:
    """
    评估智能体

    System Prompt（供 LLM 调用时使用）:
    ---
    你是时序预测模型评估专家。根据 work agent 的训练/推理结果，完成以下工作：
    1. 计算/复核关键指标：MSE、MAE、RMSE、MAPE 等
    2. 分析 loss 曲线：是否存在过拟合/欠拟合/收敛问题
    3. 与历史迭代中的指标对比，判断是否在持续改善
    4. 给出参数优化建议：
       - 调整 learning_rate
       - 调整 batch_size
       - 调整 seq_len / pred_len
       - 建议增加/减少 epochs
       - 建议切换模型架构
    5. 判断是否继续迭代：
       - 如果指标在持续改善且未达最大迭代 → next_action = "work"
       - 如果指标停滞或已达最大迭代 → next_action = "summary"

    输入 work 数据示例：
    {
      "status": "completed",
      "metrics": { "mse": 0.15, "mae": 0.25, "train_loss": [...], "val_loss": [...] }
    }

    输出 JSON 格式：
    {
      "eval": {
        "metrics": { "mse": ..., "mae": ..., "rmse": ..., "mape": ... },
        "analysis": "模型收敛良好，但 val_loss 在第 5 轮后开始上升，存在轻微过拟合",
        "param_adjustments": {
          "learning_rate": 5e-5,
          "epochs": 8
        },
        "summary": "第 N 轮评估：MSE=0.15, MAE=0.25，建议降低学习率继续训练"
      },
      "next_action": "work" | "summary"
    }
    ---
    """

    SYSTEM_PROMPT = """你是时序预测模型评估专家。根据训练/推理结果：
1. 计算/复核关键指标（MSE, MAE, RMSE, MAPE）
2. 分析 loss 曲线，诊断过拟合/欠拟合/收敛问题
3. 对比历史迭代指标，判断改善趋势
4. 给出参数优化建议（learning_rate, batch_size, epochs 等）
5. 判断是否继续迭代（next_action = "work" | "summary"）

输出合法 JSON，包含 eval（指标/分析/建议）和 next_action 字段。"""

    def __init__(self):
        self.metric_skill = MetricSkill()
        self.viz_skill = VisualizationSkill()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        评估训练结果并给出优化建议

        Args:
            state: AgentState，包含:
                - agent_data.work: work agent 的训练结果
                - agent_data.history: 历史迭代快照
                - agent_data.agent_params: 编排参数（max_iteration）
                - agent_data.agent_state: 当前状态（iteration）

        Returns:
            {
                "eval": {
                    "metrics": dict,
                    "analysis": str,
                    "param_adjustments": dict,
                    "summary": str,
                },
                "next_action": str,  # "work" | "summary"
            }
        """
        work_result = state.get("agent_data", {}).get("work", {})
        iteration = state["agent_data"]["agent_state"]["iteration"]
        max_iter = state["agent_data"]["agent_params"].get("max_iteration", 1)
        history = state.get("agent_data", {}).get("history", [])

        logger.info(f"[EvalAgent] 第 {iteration} 轮评估，最大迭代 {max_iter}")

        # TODO: 实际逻辑
        # 1. 计算指标
        # metrics = self.metric_skill.compute(work_result)

        # 2. LLM 分析 + 对比历史
        # analysis = llm.invoke(system_prompt=self.SYSTEM_PROMPT, context=work_result, history=history)

        # 3. 判断是否继续迭代
        # if iteration < max_iter and metrics improving:
        #     next_action = "work"
        # else:
        #     next_action = "summary"

        # ---- 占位返回 ----
        should_continue = iteration < max_iter

        return {
            "eval": {
                "metrics": {
                    "mse": work_result.get("metrics", {}).get("mse", 0.15),
                    "mae": work_result.get("metrics", {}).get("mae", 0.25),
                    "rmse": 0.39,
                    "mape": 8.5,
                },
                "analysis": "模型在第 {} 轮训练中收敛良好，val_loss 持续下降。".format(iteration),
                "param_adjustments": {
                    "learning_rate": 5e-5,
                } if should_continue else {},
                "summary": "第 {} 轮评估：MSE=0.15, MAE=0.25".format(iteration),
            },
            "next_action": "work" if should_continue else "summary",
        }
