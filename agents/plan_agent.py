"""
Plan Agent — 意图解析 & 初始化参数生成

职责：
1. 解析用户自然语言输入，识别任务意图（普通训练 / 自动迭代训练 / 推理）
2. 结合 RAG 检索相似历史任务经验
3. 输出模型初始化参数 + 任务编排参数

输入：AgentState（LangGraph 全局状态）
输出：{"plan": {...}, "intent": "...", "agent_params": {...}, "next_action": "..."}
"""

import json
import os
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from skills.rag_skill import RAGSkill
from skills.param_validate_skill import ParamValidateSkill

logger = logging.getLogger(__name__)


class PlanAgent:
    """
    计划智能体 — 意图解析 & 初始化参数

    工作流程：
      1. RAG 检索相似历史任务 → 提供上下文参考
      2. LLM 推理 → 生成 plan JSON（response_format=json_object）
      3. 参数校验 → 校验不通过时回退到默认参数
    """

    SYSTEM_PROMPT = """你是时序预测任务规划专家。根据用户的自然语言描述：
1. 识别用户意图（单次训练 / 迭代训练 / 推理）
2. 参考相似历史任务的参数配置（如已提供）
3. 生成模型初始化参数（model_name, dataset, seq_len, pred_len, batch_size, learning_rate, epochs 等）
4. 设定 agent_params（max_iteration / visualize）

输出合法 JSON，包含 intent, plan, agent_params, next_action 字段。"""

    # 默认参数（LLM 不可用或校验不通过时的兜底）
    _DEFAULT_PLAN = {
        "model_name": "DLinear",
        "dataset": "ETTh1",
        "seq_len": 96,
        "pred_len": 96,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
    }

    def __init__(self):
        self.rag_skill = RAGSkill()
        self.param_validate_skill = ParamValidateSkill()
        self._llm = self._init_llm()

    @staticmethod
    def _init_llm() -> ChatOpenAI | None:
        """初始化 LLM（无 API key 时不报错，后续走默认参数兜底）"""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            logger.warning("[PlanAgent] OPENAI_API_KEY 未设置，将使用默认参数")
            return None
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        执行计划推理，直接修改并返回 state

        Args:
            state: AgentState，包含:
                - agent_data.intent: 用户原始输入

        Returns:
            更新后的完整 AgentState
        """
        user_intent = state.get("agent_data", {}).get("intent", "")
        logger.info(f"[PlanAgent] 收到用户意图: {user_intent}")

        try:
            # ---- 1. RAG 检索 ----
            rag_context = ""
            if user_intent:
                rag_results = self.rag_skill.search(user_intent, top_k=3)
                if rag_results:
                    rag_context = "参考历史经验：\n" + "\n".join(
                        f"- [{r['model_name']} / {r['dataset']}] {r['experience'][:120]}"
                        for r in rag_results
                    )
                    logger.info(f"[PlanAgent] RAG 检索到 {len(rag_results)} 条历史经验")

            # ---- 2. LLM 推理生成 plan ----
            plan, intent, agent_params = self._call_llm(user_intent, rag_context)

            # ---- 3. 参数校验 ----
            validation = self.param_validate_skill.validate(plan)
            if not validation["valid"]:
                logger.warning(f"[PlanAgent] 参数校验失败: {validation['errors']}")
                # 校验不通过 → 回退到匹配的默认参数
                plan = self._fallback_plan(user_intent, plan)
                # 回退后再校验一次，确保兜底参数合法
                validation = self.param_validate_skill.validate(plan)
                if not validation["valid"]:
                    plan = dict(self._DEFAULT_PLAN)
                    intent = "train"

            # ---- 4. 写回 state ----
            state["status"] = "success"
            state["agent"] = "plan"
            state["agent_data"]["plan"] = plan
            state["agent_data"]["intent"] = intent
            state["agent_data"]["agent_params"] = {
                "max_iteration": agent_params.get("max_iteration", 1),
                "visualize": bool(agent_params.get("visualize", False)),
            }
            state["next_action"] = "work"

            if validation["warnings"]:
                logger.warning(f"[PlanAgent] 参数警告: {validation['warnings']}")

        except Exception as e:
            logger.error(f"[PlanAgent] 执行失败: {e}")
            state["status"] = "error"
            state["agent"] = "plan"
            state["agent_data"]["plan"] = dict(self._DEFAULT_PLAN)
            state["agent_data"]["intent"] = "train"
            state["agent_data"]["agent_params"] = {"max_iteration": 1, "visualize": False}
            state["errors"].append(str(e))
            state["next_action"] = "end"

        return state

    # -----------------------------------------------------------
    # LLM 调用
    # -----------------------------------------------------------

    def _call_llm(self, user_intent: str, rag_context: str) -> tuple[dict, str, dict]:
        """调用 LLM 生成 plan，不可用时从意图关键词提取参数"""
        if not self._llm or not user_intent:
            return self._fallback_plan(user_intent, {}), "train", {"max_iteration": 1, "visualize": False}

        # 构造提示
        system = self.SYSTEM_PROMPT
        if rag_context:
            system += f"\n\n{rag_context}"

        user = f"用户需求：{user_intent}\n\n请输出 JSON，包含 intent, plan, agent_params, next_action 字段。"

        try:
            response = self._llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            raw = response.content.strip()
            # 清理可能的 markdown 代码块包裹
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(raw)
            plan = data.get("plan", {})
            intent = data.get("intent", "train")
            agent_params = data.get("agent_params", {})
            return plan, intent, agent_params

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[PlanAgent] LLM 输出解析失败: {e}")
            return dict(self._DEFAULT_PLAN), "train", {"max_iteration": 1, "visualize": False}

    # -----------------------------------------------------------
    # 智能回退
    # -----------------------------------------------------------

    @staticmethod
    def _fallback_plan(user_intent: str, failed_plan: dict) -> dict:
        """LLM 参数校验失败后，尝试用用户意图中的信息修正默认参数"""
        plan = dict(PlanAgent._DEFAULT_PLAN)
        intent_lower = user_intent.lower()

        # 从意图中提取模型名
        known_models = {"dlinear": "DLinear", "patchtst": "PatchTST", "autoformer": "Autoformer",
                        "informer": "Informer", "transformer": "Transformer",
                        "linear": "Linear", "nlinear": "NLinear", "moderntcn": "ModernTCN"}
        for key, val in known_models.items():
            if key in intent_lower:
                plan["model_name"] = val
                break

        # 从意图中提取数据集名
        known_datasets = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2",
                          "weather": "weather", "electricity": "electricity", "traffic": "traffic",
                          "exchange": "exchange_rate", "illness": "national_illness"}
        for key, val in known_datasets.items():
            if key in intent_lower:
                plan["dataset"] = val
                break

        # 保留 LLM 输出中合理的数值参数
        for k in ("seq_len", "pred_len", "batch_size", "epochs"):
            v = failed_plan.get(k)
            if isinstance(v, (int, float)) and v > 0:
                plan[k] = int(v)

        # 模型特有参数补全
        if plan["model_name"] == "PatchTST":
            plan.setdefault("patch_len", 16)
            plan.setdefault("stride", 8)

        return plan
