"""
Plan Agent — 意图解析 & 初始化参数生成

职责：
1. 解析用户自然语言输入，识别任务意图（普通训练 / 自动迭代训练 / 推理）
2. 结合 RAG 检索相似历史任务经验
3. 输出模型初始化参数 + 任务编排参数

输入：AgentState（LangGraph 全局状态）
输出：{"plan": {...}, "intent": "...", "agent_params": {...}, "next_action": "..."}
"""

import logging
from typing import Any

from skills.rag_skill import RAGSkill
from skills.param_validate_skill import ParamValidateSkill

logger = logging.getLogger(__name__)


class PlanAgent:
    """
    计划智能体

    System Prompt（供 LLM 调用时使用）:
    ---
    你是时序预测任务规划专家。根据用户的自然语言描述，完成以下工作：
    1. 识别用户意图（train / train_with_iteration / inference）
    2. 从知识库（RAG）中检索相似历史任务，参考其参数配置
    3. 生成模型初始化参数，包括但不限于：
       - model_name: 模型名称（DLinear / PatchTST / Autoformer / Transformer 等）
       - dataset: 数据集名称（ETTh1 / ETTh2 / ETTm1 / ETTm2 / weather / electricity 等）
       - seq_len: 输入序列长度
       - pred_len: 预测长度
       - batch_size: 批次大小
       - learning_rate: 学习率
       - epochs: 训练轮数
       - 其他模型特定参数
    4. 设定 agent_params（max_iteration / visualize）

    输出 JSON 格式：
    {
      "intent": "train" | "train_with_iteration" | "inference",
      "plan": {
        "model_name": "...",
        "dataset": "...",
        "seq_len": 96,
        "pred_len": 96,
        ...
      },
      "agent_params": {
        "max_iteration": 1,
        "visualize": false
      },
      "next_action": "work"
    }
    ---
    """

    SYSTEM_PROMPT = """你是时序预测任务规划专家。根据用户的自然语言描述：
1. 识别用户意图（train / train_with_iteration / inference）
2. 从知识库中检索相似历史任务，参考其参数配置
3. 生成模型初始化参数（model_name, dataset, seq_len, pred_len, batch_size, learning_rate, epochs 等）
4. 设定 agent_params（max_iteration / visualize）

输出合法 JSON，包含 intent, plan, agent_params, next_action 字段。"""

    def __init__(self):
        self.rag_skill = RAGSkill()
        self.param_validate_skill = ParamValidateSkill()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行计划推理

        Args:
            state: AgentState，包含:
                - agent_data.intent: 用户原始输入
                - agent_data.history: 历史迭代快照（可能为空）

        Returns:
            {
                "intent": str,          # 任务意图
                "plan": dict,           # 模型初始化参数
                "agent_params": dict,   # 编排参数
                "next_action": str,     # 下一步: "work"
            }
        """
        user_intent = state.get("agent_data", {}).get("intent", "")
        logger.info(f"[PlanAgent] 收到用户意图: {user_intent}")

        # TODO: 调用 LLM 解析意图 + RAG 检索 + 生成参数
        # 1. RAG 检索相似历史任务
        # rag_results = self.rag_skill.search(user_intent)

        # 2. LLM 推理生成 plan
        # plan = llm.invoke(system_prompt=self.SYSTEM_PROMPT, user_prompt=user_intent)

        # 3. 参数校验
        # self.param_validate_skill.validate(plan)

        # ---- 占位返回 ----
        plan = {
            "model_name": "DLinear",
            "dataset": "ETTh1",
            "seq_len": 96,
            "pred_len": 96,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "epochs": 10,
        }
        agent_params = state.get("agent_data", {}).get("agent_params", {
            "max_iteration": 1,
            "visualize": False,
        })

        return {
            "intent": "train",
            "plan": plan,
            "agent_params": agent_params,
            "next_action": "work",
        }
