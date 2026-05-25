"""
PlanAgent 流程测试 — 模拟 LLM 返回值，披露内部运转过程

测试覆盖：
  1. 空意图 → 默认参数回退
  2. 带关键词意图 → RAG 检索 → LLM 模拟 → 参数校验
  3. LLM 输出非法 JSON 的异常处理
  4. LLM 输出校验失败后的回退逻辑
  5. RAG 有/无结果的对比
  6. 迭代意图识别

场景	            触发条件	                            测试路径
1. 无 LLM 回退	    _llm=None	                            意图关键词提取 "PatchTST" "weather" → 自动补全 patch_len/stride
2. LLM 正常	        LLM 返回合法 JSON	                    RAG 检索 → LLM 生成 → 校验通过 → 写回 state
3. 非法 JSON	    LLM 返回 "我不是 JSON"	                json.JSONDecodeError 捕获 → 回退到意图提取
4. 校验失败	        LLM 返回 model_name="BadModel"	        ParamValidateSkill 报错 → _fallback_plan → 二次校验通过
5. LLM 异常	        MockLLM 抛出 RuntimeError	            try/except 捕获 → status="error", next_action="end"
6. RAG 无结果	    rag_results=[]	                       空库场景，LLM 无参考上下文仍正常输出
7. 迭代意图	LLM     返回 intent="train_with_iteration"	    max_iteration=5 正确传递到 agent_params


用法：
  uv run python agents/test_plan_agent.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)

from agents.plan_agent import PlanAgent
from skills.rag_skill import RAGSkill
from skills.param_validate_skill import ParamValidateSkill


# ============================================================
# 格式化辅助
# ============================================================

def section(title: str):
    print(f"\n╔{'═' * 70}╗")
    print(f"║  {title:^66s}║")
    print(f"╚{'═' * 70}╝")

def step(label: str, detail: str = ""):
    print(f"\n  ▶ {label}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"    {line}")

def show_json(label: str, data, indent: int = 2):
    print(f"    📄 {label}:")
    printed = json.dumps(data, ensure_ascii=False, indent=indent)
    for line in printed.split("\n"):
        print(f"      {line}")


# ============================================================
# 模拟工厂：接管 PlanAgent._call_llm 和 RAGSkill.search
# ============================================================

class MockLLM:
    """模拟 LLM 返回预定义的 plan JSON"""
    def __init__(self, response: dict, raise_error: bool = False):
        self.response = response
        self.raise_error = raise_error

    def invoke(self, messages):
        if self.raise_error:
            raise RuntimeError("模拟 LLM 网络超时")

        # 打印收到的 prompt
        print(f"    ╔══ LLM 收到消息 ══╗")
        for msg in messages:
            role = type(msg).__name__.replace("Message", "")
            content = msg.content[:200]
            print(f"    ║  [{role}]: {content}")
        print(f"    ╚══ LLM 返回 JSON ══╝")
        show_json("模拟 LLM 响应", self.response)

        # 模拟 ChatOpenAI 的返回格式
        class MockResponse:
            class MockContent:
                content = ""
            content = json.dumps(self.response, ensure_ascii=False)

        return MockResponse()


class MockRAGSkill:
    """模拟 RAG 检索结果"""
    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        print(f"    ╔══ RAG 检索 ══╗")
        print(f"    ║  query: {query}")
        print(f"    ║  top_k: {top_k}")
        print(f"    ║  命中数: {len(self._results)}")
        if self._results:
            for r in self._results:
                print(f"    ║    [{r['model_name']} / {r['dataset']}] sim={r['similarity']:.3f}")
        print(f"    ╚══ RAG 完毕 ══╝")
        return self._results


# 构建一个模拟的 PlanAgent 实例
def make_agent(llm_response: dict | None = None,
               rag_results: list[dict] | None = None,
               llm_error: bool = False) -> PlanAgent:
    """创建一个 PlanAgent，用模拟对象替换其内部依赖"""
    agent = PlanAgent.__new__(PlanAgent)
    agent.param_validate_skill = ParamValidateSkill()
    agent.rag_skill = MockRAGSkill(rag_results)
    agent._llm = MockLLM(llm_response or _DEFAULT_LLM_RESPONSE, raise_error=llm_error) if llm_response is not None else None
    return agent


# 默认的模拟 LLM 响应
_DEFAULT_LLM_RESPONSE = {
    "intent": "train",
    "plan": {
        "model_name": "PatchTST",
        "dataset": "ETTh1",
        "seq_len": 96,
        "pred_len": 96,
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 30,
        "patch_len": 16,
        "stride": 8,
        "d_model": 512,
        "n_heads": 8,
        "dropout": 0.1,
    },
    "agent_params": {"max_iteration": 3, "visualize": True},
    "next_action": "work",
}

# 默认的模拟 RAG 结果
_DEFAULT_RAG_RESULTS = [
    {"model_name": "DLinear", "dataset": "ETTh1",
     "experience": "在 ETTh1 上 DLinear 训练 50 轮，seq_len=96, pred_len=96，MSE=0.152",
     "similarity": 0.85},
    {"model_name": "PatchTST", "dataset": "ETTh1",
     "experience": "在 ETTh1 上 PatchTST 训练 50 轮，seq_len=96, pred_len=96，MSE=0.168",
     "similarity": 0.72},
]


# ============================================================
# 辅助：跟踪 PlanAgent.run 内部流程
# ============================================================

def traced_run(agent: PlanAgent, state: dict, label: str):
    """执行 agent.run() 并披露详细的内部变化"""
    print(f"\n  {'─' * 60}")
    step("初始 state")
    show_json("agent_data", state.get("agent_data", {}))

    # 记录执行前快照
    before = json.loads(json.dumps(state))

    # 执行
    print(f"\n  🔄 执行 PlanAgent.run(state)...")
    print(f"     内部流程:")
    print(f"       1. 读取 user_intent")
    print(f"       2. RAGSkill.search() 检索历史")
    print(f"       3. _call_llm() 生成 plan")
    print(f"       4. ParamValidateSkill.validate() 校验参数")
    print(f"       5. 写回 AgentState")
    print()

    result = agent.run(state)

    # 披露 state 变化
    print(f"\n  📊 State 变化:")

    def diff(before_val, after_val, path=""):
        changes = []
        if isinstance(before_val, dict) and isinstance(after_val, dict):
            all_keys = set(before_val.keys()) | set(after_val.keys())
            for k in sorted(all_keys):
                sub = diff(before_val.get(k), after_val.get(k), f"{path}.{k}")
                changes.extend(sub)
        elif before_val != after_val:
            changes.append((path, before_val, after_val))
        return changes

    changes = diff(before, result)
    for path, old, new in changes:
        if path.startswith(".agent_data.history"):
            continue
        old_s = json.dumps(old, ensure_ascii=False)[:60] if not isinstance(old, str) else str(old)[:60]
        new_s = json.dumps(new, ensure_ascii=False)[:60] if not isinstance(new, str) else str(new)[:60]
        print(f"      {path}: {old_s} → {new_s}")

    # 打印关键结果
    print(f"\n  📦 关键输出:")
    ad = result.get("agent_data", {})
    print(f"      status      = {result.get('status')}")
    print(f"      agent       = {result.get('agent')}")
    print(f"      next_action = {result.get('next_action')}")
    print(f"      intent      = {ad.get('intent')}")
    if ad.get("errors"):
        print(f"      errors      = {ad['errors']}")

    plan = ad.get("plan", {})
    if plan:
        print(f"      plan:")
        for k, v in plan.items():
            print(f"        {k:>20} = {v}")

    ap = ad.get("agent_params", {})
    if ap:
        print(f"      agent_params: max_iteration={ap.get('max_iteration')}, visualize={ap.get('visualize')}")

    return result


# ============================================================
# 测试场景
# ============================================================

def test_default_fallback():
    """场景 1：LLM 不可用 → _fallback_plan 从意图提取参数"""
    section("场景 1：LLM 不可用，从意图关键词回退")

    step("条件",
          "LLM=None（无 API key）\n"
          "RAG 检索正常\n"
          "意图包含 'PatchTST' 和 'weather'")

    agent = make_agent(llm_response=None, rag_results=_DEFAULT_RAG_RESULTS)
    state = {
        "task_id": "test_001",
        "agent_data": {
            "intent": "用 PatchTST 预测 weather 气温",
            "plan": {}, "eval": {}, "work": {}, "summary": {},
            "agent_params": {}, "agent_state": {"iteration": 0},
            "history": [],
        },
        "errors": [],
    }

    traced_run(agent, state, "无 LLM 回退")

    # 验证
    plan = state["agent_data"]["plan"]
    assert plan["model_name"] == "PatchTST", f"期望 PatchTST，实际 {plan['model_name']}"
    assert plan["dataset"] == "weather", f"期望 weather，实际 {plan['dataset']}"
    assert plan.get("patch_len") == 16, "PatchTST 应自动补全 patch_len"
    assert plan.get("stride") == 8, "PatchTST 应自动补全 stride"
    print(f"\n  ✅ 验证: model=PatchTST, dataset=weather, patch_len=16, stride=8")


def test_llm_success():
    """场景 2：LLM 正常返回 → 校验通过 → 写回 state"""
    section("场景 2：LLM 正常返回，校验通过")

    step("条件",
          "LLM 返回合法参数\n"
          "RAG 返回 2 条历史记录\n"
          "参数校验全部通过")

    agent = make_agent(llm_response=_DEFAULT_LLM_RESPONSE, rag_results=_DEFAULT_RAG_RESULTS)
    state = {
        "task_id": "test_002",
        "agent_data": {
            "intent": "用 PatchTST 在 ETTh1 上迭代优化预测",
            "plan": {}, "eval": {}, "work": {}, "summary": {},
            "agent_params": {}, "agent_state": {"iteration": 0},
            "history": [],
        },
        "errors": [],
    }

    result = traced_run(agent, state, "LLM 正常返回")

    plan = result["agent_data"]["plan"]
    assert plan["model_name"] == "PatchTST"
    assert plan["dataset"] == "ETTh1"
    assert plan["batch_size"] == 64
    assert result["agent_data"]["intent"] == "train"
    print(f"\n  ✅ 验证: 参数全部通过校验，next_action=work")


def test_llm_invalid_json():
    """场景 3：LLM 返回非法 JSON → 异常处理"""
    section("场景 3：LLM 返回非法 JSON")

    step("条件",
          "LLM 返回内容不是合法 JSON\n"
          "捕获 json.JSONDecodeError\n"
          "回退到 _fallback_plan")

    # 构造一个返回非法字符串的 mock
    class BrokenLLM:
        def invoke(self, messages):
            print(f"    ╔══ LLM 收到消息 ══╗")
            print(f"    ║  (打印省略)")
            print(f"    ╚══ LLM 返回非法内容 ══╝")
            print(f"    LLM 输出: 我不是 JSON```")
            class R:
                content = "我不是 JSON```"
            return R()

    agent = PlanAgent.__new__(PlanAgent)
    agent.param_validate_skill = ParamValidateSkill()
    agent.rag_skill = MockRAGSkill(_DEFAULT_RAG_RESULTS)
    agent._llm = BrokenLLM()

    state = {
        "task_id": "test_003",
        "agent_data": {
            "intent": "用 DLinear 在 ETTh1 上训练",
            "plan": {}, "eval": {}, "work": {}, "summary": {},
            "agent_params": {}, "agent_state": {"iteration": 0},
            "history": [],
        },
        "errors": [],
    }

    result = traced_run(agent, state, "非法 JSON")

    plan = result["agent_data"]["plan"]
    # 应回退到从意图提取的 DLinear
    print(f"\n  ✅ 验证: 非法 JSON 被捕获，回退到意图提取参数")
    assert plan["model_name"] == "DLinear", f"期望 DLinear，实际 {plan['model_name']}"
    assert plan["dataset"] == "ETTh1"


def test_llm_validation_fail():
    """场景 4：LLM 返回参数校验失败 → 回退"""
    section("场景 4：LLM 返回的参数校验失败")

    step("条件",
          "LLM 返回的参数中 model_name='BadModel'\n"
          "ParamValidateSkill 报错\n"
          "触发 _fallback_plan 从意图回退")

    bad_response = dict(_DEFAULT_LLM_RESPONSE)
    bad_response["plan"] = dict(_DEFAULT_LLM_RESPONSE["plan"])
    bad_response["plan"]["model_name"] = "BadModel"
    bad_response["plan"]["dataset"] = "BadDataset"

    agent = make_agent(llm_response=bad_response, rag_results=_DEFAULT_RAG_RESULTS)
    state = {
        "task_id": "test_004",
        "agent_data": {
            "intent": "用 PatchTST 在 ETTh1 上训练",
            "plan": {}, "eval": {}, "work": {}, "summary": {},
            "agent_params": {}, "agent_state": {"iteration": 0},
            "history": [],
        },
        "errors": [],
    }

    result = traced_run(agent, state, "校验失败回退")

    plan = result["agent_data"]["plan"]
    print(f"\n  ✅ 验证: LLM 的 BadModel 被校验拦截，回退到 PatchTST")
    assert plan["model_name"] == "PatchTST"
    assert plan["dataset"] == "ETTh1"
    assert plan.get("patch_len") == 16


def test_llm_exception():
    """场景 5：LLM 调用抛出异常 → 走 except 兜底"""
    section("场景 5：LLM 调用异常")

    step("条件",
          "LLM.invoke() 抛出 RuntimeError\n"
          "被 run() 的 try/except 捕获\n"
          "state.status='error', next_action='end'")

    agent = make_agent(llm_response=_DEFAULT_LLM_RESPONSE, rag_results=_DEFAULT_RAG_RESULTS, llm_error=True)
    state = {
        "task_id": "test_005",
        "agent_data": {
            "intent": "用 DLinear 预测",
            "plan": {}, "eval": {}, "work": {}, "summary": {},
            "agent_params": {}, "agent_state": {"iteration": 0},
            "history": [],
        },
        "errors": [],
    }

    result = traced_run(agent, state, "LLM 异常")

    print(f"\n  ✅ 验证: status=error, next_action=end")
    assert result["status"] == "error"
    assert result["next_action"] == "end"
    assert "模拟 LLM 网络超时" in str(result.get("errors", []))


def test_no_rag_results():
    """场景 6：RAG 无结果（空库首次运行）"""
    section("场景 6：RAG 无检索结果")

    step("条件",
          "RAG 返回空列表（新数据库）\n"
          "LLM 仍可正常生成 plan\n"
          "无历史经验参考，仅靠 LLM 知识推理")

    agent = make_agent(llm_response=_DEFAULT_LLM_RESPONSE, rag_results=[])
    state = {
        "task_id": "test_006",
        "agent_data": {
            "intent": "预测 weather 未来 7 天气温",
            "plan": {}, "eval": {}, "work": {}, "summary": {},
            "agent_params": {}, "agent_state": {"iteration": 0},
            "history": [],
        },
        "errors": [],
    }

    result = traced_run(agent, state, "RAG 无结果")

    plan = result["agent_data"]["plan"]
    print(f"\n  ✅ 验证: RAG 无结果不影响主流程，LLM 正常输出")
    assert result["status"] == "success"


def test_iteration_intent():
    """场景 7：意图识别为迭代训练"""
    section("场景 7：迭代训练意图")

    step("条件",
          "LLM 返回 intent='train_with_iteration'\n"
          "max_iteration=5\n"
          "agent_params 正确传递")

    response = dict(_DEFAULT_LLM_RESPONSE)
    response["intent"] = "train_with_iteration"
    response["agent_params"] = {"max_iteration": 5, "visualize": True}

    agent = make_agent(llm_response=response, rag_results=_DEFAULT_RAG_RESULTS)
    state = {
        "task_id": "test_007",
        "agent_data": {
            "intent": "用 DLinear 在 ETTh1 上迭代优化 5 轮",
            "plan": {}, "eval": {}, "work": {}, "summary": {},
            "agent_params": {}, "agent_state": {"iteration": 0},
            "history": [],
        },
        "errors": [],
    }

    result = traced_run(agent, state, "迭代意图")

    ad = result["agent_data"]
    print(f"\n  ✅ 验证: intent=train_with_iteration, max_iteration=5")
    assert ad["intent"] == "train_with_iteration"
    assert ad["agent_params"]["max_iteration"] == 5


# ============================================================
# 主入口
# ============================================================

def main():
    print()
    print(f"╔{'═' * 70}╗")
    print(f"║{'PlanAgent 流程测试 — 披露内部运转过程'.center(66)}║")
    print(f"║{'模拟 LLM + RAG，不依赖外部服务'.center(66)}║")
    print(f"╚{'═' * 70}╝")
    print()
    print(f"  覆盖场景:")
    print(f"    ├─ 场景 1: 无 LLM，从意图关键词回退")
    print(f"    ├─ 场景 2: LLM 正常返回 + 校验通过")
    print(f"    ├─ 场景 3: LLM 返回非法 JSON")
    print(f"    ├─ 场景 4: LLM 参数校验失败 → 回退")
    print(f"    ├─ 场景 5: LLM 调用异常 → error 状态")
    print(f"    ├─ 场景 6: RAG 无检索结果")
    print(f"    └─ 场景 7: 迭代训练意图识别")
    print()

    test_default_fallback()
    test_llm_success()
    test_llm_invalid_json()
    test_llm_validation_fail()
    test_llm_exception()
    test_no_rag_results()
    test_iteration_intent()

    print(f"\n{'═' * 72}")
    print(f"  ✅ 全部 7 个场景测试通过！")
    print(f"{'═' * 72}")
    print()


if __name__ == "__main__":
    main()
