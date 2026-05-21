"""
Work Agent — 训练 / 推理执行器

职责：
1. 接收 plan 或 eval 回传的参数
2. 找到或加载模型检查点
3. 调用后端训练/推理 API
4. 解析并回传训练日志 / 推理结果

输入：AgentState（LangGraph 全局状态）
输出：{"work": {...}, "next_action": "..."}
"""

import logging
from typing import Any

from skills.api_skill import APISkill
from skills.checkpoint_skill import CheckpointSkill
from skills.log_parser_skill import LogParserSkill

logger = logging.getLogger(__name__)


class WorkAgent:
    """
    工作智能体

    System Prompt（供 LLM 调用时使用）:
    ---
    你是时序预测任务执行专家。根据上游（plan/eval）传入的参数，完成以下工作：
    1. 查找模型检查点：如果是推理任务，从向量数据库中查找最近的模型权重路径
    2. 构造训练/推理配置：将参数填充到后端 API 所需的配置 JSON 中
    3. 调用后端 API 启动训练/推理
    4. 解析返回的日志/CSV 结果，提取关键指标（loss 曲线、MSE、MAE 等）

    输入参数示例（来自 plan 或 eval）：
    {
      "model_name": "DLinear",
      "dataset": "ETTh1",
      "seq_len": 96,
      "pred_len": 96,
      "batch_size": 32,
      "learning_rate": 0.0001,
      "epochs": 10,
      ...
    }

    输出 JSON 格式：
    {
      "work": {
        "status": "completed" | "failed",
        "checkpoint_path": "...",
        "log_path": "...",
        "metrics": {
          "train_loss": [...],
          "val_loss": [...],
          "mse": ...,
          "mae": ...
        },
        "raw_log": "..."
      },
      "next_action": "eval" | "summary" | "end"
    }
    ---
    """

    SYSTEM_PROMPT = """你是时序预测任务执行专家。根据上游传入的参数：
1. 查找模型检查点（推理任务时）
2. 构造训练/推理配置 JSON
3. 调用后端 API 启动训练/推理
4. 解析返回的日志，提取关键指标（loss, MSE, MAE 等）

输出合法 JSON，包含 work（状态/路径/指标/日志）和 next_action 字段。"""

    def __init__(self):
        self.api_skill = APISkill()
        self.checkpoint_skill = CheckpointSkill()
        self.log_parser_skill = LogParserSkill()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行训练或推理，直接修改并返回 state

        Args:
            state: AgentState，包含:
                - agent_data.plan: plan agent 的初始化参数
                - agent_data.eval: eval agent 的优化建议（迭代场景）
                - agent_data.agent_params: 编排参数

        Returns:
            更新后的完整 AgentState
        """
        plan_params = state.get("agent_data", {}).get("plan", {})
        eval_suggestions = state.get("agent_data", {}).get("eval", {})
        intent = state.get("agent_data", {}).get("intent", "train")

        logger.info(f"[WorkAgent] 任务类型: {intent}, plan 参数: {plan_params}")

        # TODO: 实际逻辑
        # 1. 如果是推理任务，先查找检查点
        # if intent == "inference":
        #     checkpoint = self.checkpoint_skill.find_best(...)
        # 2. 合并 eval 建议参数
        # merged_params = {**plan_params, **eval_suggestions.get("param_adjustments", {})}
        # 3. 调用后端 API
        # result = self.api_skill.run_training(merged_params)
        # 4. 解析日志
        # metrics = self.log_parser_skill.parse(result["log_path"])

        # ---- 占位：直接修改 state ----
        state["status"] = "success"
        state["agent"] = "work"
        state["agent_data"]["work"] = {
            "status": "completed",
            "checkpoint_path": "/path/to/checkpoint.pth",
            "log_path": "/path/to/training.log",
            "metrics": {
                "train_loss": [0.5, 0.3, 0.2],
                "val_loss": [0.6, 0.4, 0.35],
                "mse": 0.15,
                "mae": 0.25,
            },
            "raw_log": "Epoch 1: loss=0.5\nEpoch 2: loss=0.3\n...",
        }
        state["agent_data"]["agent_state"]["iteration"] += 1
        state["next_action"] = "eval"

        return state
