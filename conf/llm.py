"""
LLM 配置 — 统一管理大模型连接参数

所有 Agent 通过 get_llm() 获取实例，无需各自独立配置。

模型分级：
  get_llm()          → 普通模型（gpt-4o-mini，日常推理）
  get_llm(advanced=True) → 高级模型（gpt-4o，复杂分析）
"""

import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ============================================================
# 配置区（按需修改）
# ============================================================
_API_KEY = "sk-df0f185a0cdf43be8c9c3bc447c8f523"
_BASE_URL = "https://api.deepseek.com"

_MODEL_NORMAL = "deepseek-v4-flash"  # 普通模型：快速、便宜
_MODEL_ADVANCED = "deepseek-v4-pro"     # 高级模型：更强、更贵
# ============================================================


def get_llm(advanced: bool = False) -> ChatOpenAI | None:
    """
    获取 ChatOpenAI 实例

    Args:
        advanced: 是否使用高级模型（默认 False）

    Returns:
        ChatOpenAI 实例，或 None（API key 为空时）
    """
    if not _API_KEY:
        logger.warning("[LLM] API key 未配置，LLM 不可用")
        return None

    model = _MODEL_ADVANCED if advanced else _MODEL_NORMAL
    logger.info("[LLM] model=%s  advanced=%s", model, advanced)
    return ChatOpenAI(
        model=model,
        base_url=_BASE_URL,
        api_key=_API_KEY,
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
