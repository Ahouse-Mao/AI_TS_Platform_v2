"""集成测试辅助工具：打印 LLM 的完整输入和输出"""

from unittest import mock
import conf.llm as conf_llm


class LoggingLLMWrapper:
    """包装 ChatOpenAI，在 invoke 前后打印完整内容"""

    def __init__(self, llm):
        self._llm = llm

    def invoke(self, messages, **kwargs):
        print()
        for msg in messages:
            role = type(msg).__name__.replace("Message", "")
            print(f"  ╔══ LLM [{role}] ══╗")
            print(msg.content)
            print(f"  ╚═══╝")
        resp = self._llm.invoke(messages, **kwargs)
        print(f"  ═══ LLM 返回 ═══")
        print(resp.content)
        print()
        return resp

    def __getattr__(self, name):
        return getattr(self._llm, name)


def patch_logging_llm():
    """返回一个 mock.patch，在真实 LLM 的 invoke 前后打印完整内容"""
    _real_get_llm = conf_llm.get_llm

    def logging_get_llm(advanced=False):
        llm = _real_get_llm(advanced)
        if llm is None:
            return None
        return LoggingLLMWrapper(llm)

    return mock.patch("conf.llm.get_llm", side_effect=logging_get_llm)
