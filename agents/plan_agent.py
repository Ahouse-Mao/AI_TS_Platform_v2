"""
Plan Agent — 意图解析 & 初始化参数生成 (Pydantic 重构版)

职责：
1. 解析用户自然语言输入，识别任务意图
2. 结合 RAG 检索相似历史任务经验
3. 输出模型初始化参数 + 任务编排参数（受 Pydantic 约束）
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, ConfigDict

from skills.rag_skill import RAGSkill
from skills.param_validate_skill import ParamValidateSkill
from conf.llm import get_llm

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Pydantic 输出模型
# ------------------------------------------------------------------

class PlanConfig(BaseModel):
    """模型训练 / 推理配置"""
    model_config = ConfigDict(strict=True, extra="allow")

    model_name: str = Field(default="DLinear", description="模型名称")
    dataset: str = Field(default="ETTh1", description="数据集名称")
    seq_len: int = Field(default=96, ge=1, description="输入序列长度")
    pred_len: int = Field(default=96, ge=1, description="预测长度")
    batch_size: int = Field(default=32, ge=1, description="批量大小")
    learning_rate: float = Field(default=1e-4, gt=0, description="学习率")
    epochs: int = Field(default=10, ge=1, description="训练轮数")
    # 允许携带额外参数，如 PatchTST 的 patch_len / stride
    extra_params: dict = Field(default_factory=dict, description="模型特有参数")

    @field_validator('learning_rate', mode='before')
    @classmethod
    def cast_lr(cls, v):
        """确保学习率是浮点数"""
        return float(v)


class AgentParams(BaseModel):
    """任务编排参数"""
    model_config = ConfigDict(strict=True)

    max_iteration: int = Field(default=1, ge=1, description="最大迭代次数")
    visualize: bool = Field(default=False, description="是否可视化")


class PlanOutput(BaseModel):
    """PlanAgent 完整输出"""
    model_config = ConfigDict(strict=True)

    intent: str = Field(..., description="用户意图：train / iterative_train / inference")
    plan: PlanConfig
    agent_params: AgentParams
    next_action: str = Field(default="work", description="下一步动作")

    @field_validator('intent')
    @classmethod
    def valid_intent(cls, v):
        allowed = {"train", "iterative_train", "inference"}
        if v not in allowed:
            raise ValueError(f"intent 必须为 {allowed} 之一，收到: {v}")
        return v


# ------------------------------------------------------------------
# PlanAgent
# ------------------------------------------------------------------

class PlanAgent:
    """计划智能体 — 意图解析 & 初始化参数"""

    SYSTEM_PROMPT = """你是时序预测任务规划专家。根据用户的自然语言描述：
1. 识别用户意图（单次训练 / 迭代训练 / 推理）
2. 参考相似历史任务的参数配置（如已提供）
3. 生成模型初始化参数（model_name, dataset, seq_len, pred_len, batch_size, learning_rate, epochs 等）
4. 设定 agent_params（max_iteration / visualize）

输出必须符合给定的函数调用定义，不要输出额外文本。"""

    # 默认配置（LLM 完全不可用时的硬兜底）
    _DEFAULT_PLAN = {
        "model_name": "DLinear",
        "dataset": "ETTh1",
        "seq_len": 96,
        "pred_len": 96,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
    }

    def __init__(self, llm_model: str = "normal"):
        self.rag_skill = RAGSkill()
        self.param_validate_skill = ParamValidateSkill()
        self._llm = get_llm(advanced=(llm_model == "advanced"))

    # -----------------------------------------------------------
    # 主流程
    # -----------------------------------------------------------

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        user_intent = state.get("agent_data", {}).get("intent", "")
        logger.info(f"[PlanAgent] 收到用户意图: {user_intent}")

        # 确保 errors 字段存在
        state.setdefault("errors", [])

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

            # ---- 2. LLM 推理（结构化输出） ----
            plan_output = self._call_llm(user_intent, rag_context)

            # ---- 3. 参数校验 ----
            plan_dict = plan_output.plan.model_dump(exclude={"extra_params"})
            # 合并额外参数
            extra = plan_output.plan.extra_params
            if extra:
                plan_dict.update(extra)

            validation = self.param_validate_skill.validate(plan_dict)
            if not validation["valid"]:
                logger.warning(f"[PlanAgent] 参数校验失败: {validation['errors']}")
                # 回退并重新校验
                plan_dict = self._fallback_plan(user_intent, plan_dict)
                validation = self.param_validate_skill.validate(plan_dict)
                if not validation["valid"]:
                    plan_dict = dict(self._DEFAULT_PLAN)
                    plan_output.intent = "train"

            # ---- 4. 写回 state ----
            state["status"] = "success"
            state["agent"] = "plan"
            state["agent_data"]["plan"] = plan_dict
            state["agent_data"]["intent"] = plan_output.intent
            # 保留已有 agent_params（用户传入的值优先），LLM 输出作为兜底
            existing = state.get("agent_data", {}).get("agent_params", {})
            state["agent_data"]["agent_params"] = {
                "max_iteration": existing.get("max_iteration", plan_output.agent_params.max_iteration),
                "visualize": existing.get("visualize", plan_output.agent_params.visualize),
            }
            state["next_action"] = plan_output.next_action

            if validation["warnings"]:
                logger.warning(f"[PlanAgent] 参数警告: {validation['warnings']}")

        except Exception as e:
            logger.error(f"[PlanAgent] 执行失败: {e}")
            state["status"] = "error"
            state["agent"] = "plan"
            state["agent_data"]["plan"] = dict(self._DEFAULT_PLAN)
            state["agent_data"]["intent"] = "train"
            # 异常时也保留用户传入的 agent_params
            existing = state.get("agent_data", {}).get("agent_params", {})
            state["agent_data"]["agent_params"] = {
                "max_iteration": existing.get("max_iteration", 1),
                "visualize": existing.get("visualize", False),
            }
            state["errors"].append(str(e))
            state["next_action"] = "end"

        return state

    # -----------------------------------------------------------
    # LLM 调用（Pydantic 结构化输出）
    # -----------------------------------------------------------

    def _call_llm(self, user_intent: str, rag_context: str) -> PlanOutput:
        """调用 LLM (JSON mode)，解析后用 Pydantic 校验，失败抛异常"""
        if not self._llm or not user_intent:
            raise RuntimeError("LLM 未初始化或用户输入为空")

        system = self.SYSTEM_PROMPT
        if rag_context:
            system += f"\n\n{rag_context}"

        user = (
            f"用户需求：{user_intent}\n\n"
            "请严格按以下 JSON 结构输出，字段名不可更改：\n"
            '{"intent":"train","plan":{"model_name":"DLinear","dataset":"ETTh1",'
            '"seq_len":96,"pred_len":96,"batch_size":32,"learning_rate":0.001,"epochs":10,'
            '"extra_params":{}},"agent_params":{"max_iteration":1,"visualize":false},'
            '"next_action":"work"}'
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]

        response = self._llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(raw)
        # 修正 LLM 可能返回的非法值，避免 Pydantic 校验阻塞
        self._sanitize_plan(data)
        result = PlanOutput(**data)
        logger.info(f"[PlanAgent] 解析成功: intent={result.intent}, model={result.plan.model_name}")
        return result

    @staticmethod
    def _sanitize_plan(data: dict) -> None:
        """修正 LLM 返回的非法数值，直接修改 data"""
        plan = data.get("plan", {})
        fixes = {
            "epochs": (10, lambda v: isinstance(v, (int, float)) and v < 1),
            "seq_len": (96, lambda v: not isinstance(v, (int, float)) or v < 1),
            "pred_len": (96, lambda v: not isinstance(v, (int, float)) or v < 1),
            "batch_size": (32, lambda v: not isinstance(v, (int, float)) or v < 1),
        }
        for key, (default, bad) in fixes.items():
            if key in plan and bad(plan.get(key)):
                logger.warning(f"[PlanAgent] LLM 返回非法 {key}={plan[key]}，修正为 {default}")
                plan[key] = default

    # -----------------------------------------------------------
    # 智能回退（保留原逻辑，但返回的是 dict）
    # -----------------------------------------------------------

    @staticmethod
    def _fallback_plan(user_intent: str, failed_plan: dict) -> dict:
        """LLM 参数校验失败后，尝试用意图中的信息修正默认参数"""
        plan = dict(PlanAgent._DEFAULT_PLAN)
        intent_lower = user_intent.lower()

        # 从意图中提取模型名
        known_models = {
            "dlinear": "DLinear", "patchtst": "PatchTST", "autoformer": "Autoformer",
            "informer": "Informer", "transformer": "Transformer",
            "linear": "Linear", "nlinear": "NLinear", "moderntcn": "ModernTCN"
        }
        for key, val in known_models.items():
            if key in intent_lower:
                plan["model_name"] = val
                break

        # 从意图中提取数据集名
        known_datasets = {
            "etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2",
            "weather": "weather", "electricity": "electricity", "traffic": "traffic",
            "exchange": "exchange_rate", "illness": "national_illness"
        }
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