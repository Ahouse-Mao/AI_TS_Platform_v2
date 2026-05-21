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
    1. 查找模型检查点：如果是推理任务，从 checkpoints 目录中查找最近的模型权重路径
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

        try:
            # ---- 1. 合并参数 ----
            # plan 提供基础参数，eval 的 param_adjustments 覆盖/补充
            merged_params = {**plan_params}
            if eval_suggestions:
                adjustments = eval_suggestions.get("param_adjustments", {})
                if adjustments:
                    logger.info(f"[WorkAgent] 合并 eval 调整参数: {adjustments}")
                    merged_params.update(adjustments)

            # ---- 2. 根据任务类型执行 ----
            if intent == "inference":
                result = self._run_inference(merged_params)
            else:
                result = self._run_training(merged_params)

            # ---- 3. 更新 state ----
            state["status"] = result.get("status", "error")
            state["agent"] = "work"
            state["agent_data"]["work"] = result

            # ---- 4. 决定 next_action ----
            if result.get("status") == "failed":
                state["next_action"] = "summary" if state.get("agent_data", {}).get("history") else "end"
            elif intent == "inference":
                # 推理任务不需要 eval，直接结束
                state["next_action"] = "summary"
            else:
                # 训练任务 → 交给 eval 评估
                state["next_action"] = "eval"

            state["agent_data"]["agent_state"]["iteration"] = (
                state["agent_data"]["agent_state"].get("iteration", 0) + 1
            )

        except Exception as e:
            logger.error(f"[WorkAgent] 执行失败: {e}", exc_info=True)
            state["status"] = "error"
            state["agent"] = "work"
            state["agent_data"]["work"] = {
                "status": "failed",
                "error": str(e),
            }
            state["next_action"] = "end"

        return state

    # -----------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------

    def _run_training(self, params: dict[str, Any]) -> dict[str, Any]:
        """执行训练任务"""
        logger.info(f"[WorkAgent] 开始训练: {params.get('model_name')} on {params.get('dataset')}")

        # 调用后端 API（同步阻塞，内部轮询直到完成）
        api_result = self.api_skill.run_training(params)

        if api_result.get("status") == "failed":
            logger.error(f"[WorkAgent] 训练失败: {api_result.get('error')}")
            return {
                "status": "failed",
                "error": api_result.get("error", "训练失败"),
            }

        # 从 API 结果中提取指标
        checkpoint_path = api_result.get("checkpoint_path", "")
        log_path = api_result.get("log_path", "")

        # 解析日志获取详细 loss 曲线
        metrics = api_result.get("metrics", {})
        if log_path:
            parsed = self.log_parser_skill.parse(log_path)
            # API 返回的 metrics 是最终值，日志解析可以补全 loss 曲线
            if parsed.get("train_loss"):
                metrics["train_loss"] = parsed["train_loss"]
            if parsed.get("val_loss"):
                metrics["val_loss"] = parsed["val_loss"]
            if parsed.get("test_loss"):
                metrics["test_loss"] = parsed["test_loss"]
            if parsed.get("total_time"):
                metrics["total_time"] = parsed["total_time"]

        return {
            "status": "completed",
            "checkpoint_path": checkpoint_path,
            "log_path": log_path,
            "metrics": metrics,
            "raw_log": metrics.get("raw_summary", ""),
        }

    def _run_inference(self, params: dict[str, Any]) -> dict[str, Any]:
        """执行推理任务"""
        logger.info(f"[WorkAgent] 开始推理: {params.get('model_name')} on {params.get('dataset')}")

        # 如果没有指定 checkpoint_path，自动查找最优检查点
        if not params.get("checkpoint_path"):
            cp_result = self.checkpoint_skill.find_best(
                model_name=params.get("model_name", ""),
                dataset=params.get("dataset", ""),
                pred_len=params.get("pred_len"),
            )
            if cp_result.get("checkpoint_path"):
                params["checkpoint_path"] = cp_result["checkpoint_path"]
                logger.info(f"[WorkAgent] 自动找到检查点: {cp_result['checkpoint_path']}")
            else:
                logger.warning(f"[WorkAgent] 未找到匹配的检查点，使用默认路径")

        # 调用后端推理 API
        api_result = self.api_skill.run_inference(params)

        if api_result.get("status") == "failed":
            logger.error(f"[WorkAgent] 推理失败: {api_result.get('error')}")
            return {
                "status": "failed",
                "error": api_result.get("error", "推理失败"),
            }

        return {
            "status": "completed",
            "checkpoint_path": api_result.get("checkpoint_path", ""),
            "log_path": api_result.get("log_path", ""),
            "predictions_path": api_result.get("predictions_path"),
            "metrics": api_result.get("metrics", {}),
            "raw_log": "",
        }
