"""
EvalAgent 流程测试 — 模拟 LLM 返回值，披露内部运转过程

测试覆盖：
  1. 首轮评估 → improving 趋势 → 继续迭代
  2. 连续改善 → 学习率递减
  3. 指标停滞 → plateau → 加大调整
  4. 指标恶化 → worsening → 提前结束
  5. 达到最大迭代 → 结束
  6. LLM 正常返回分析
  7. LLM 非法 JSON → 规则回退

用法：
  uv run python agents/test_eval_agent.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)

from agents.eval_agent import EvalAgent


# ============================================================
# 格式化辅助
# ============================================================

def section(title: str):
    print(f"\n╔{'═' * 70}╗")
    print(f"║  {title:^66s}║")
    print(f"╚{'═' * 70}╝")

def show_json(label: str, data, indent: int = 2):
    print(f"    📄 {label}:")
    printed = json.dumps(data, ensure_ascii=False, indent=indent)
    for line in printed.split("\n"):
        print(f"      {line}")


# ============================================================
# 模拟工厂
# ============================================================

class MockLLM:
    def __init__(self, response: dict | None, raise_error: bool = False):
        self.response = response
        self.raise_error = raise_error

    def invoke(self, messages):
        if self.raise_error:
            raise RuntimeError("模拟 LLM 超时")

        print(f"    ╔══ LLM 收到 ══╗")
        for msg in messages:
            role = type(msg).__name__.replace("Message", "")
            print(f"    ║  [{role}]: {msg.content[:100]}...")
        print(f"    ╚══ 返回 ══╝")
        show_json("响应", self.response)

        class R:
            content = json.dumps(self.response, ensure_ascii=False)
        return R()


def make_agent(llm_response: dict | None = None,
               llm_error: bool = False,
               enable_llm: bool = False) -> EvalAgent:
    agent = EvalAgent.__new__(EvalAgent)
    agent.metric_skill = None  # 测试中不实际计算指标

    if enable_llm and llm_response is not None:
        agent._llm = MockLLM(llm_response, raise_error=llm_error)
    elif enable_llm and llm_response is None:
        agent._llm = MockLLM(None)
    else:
        agent._llm = None

    return agent


def build_state(iteration: int, max_iter: int, work_metrics: dict,
                history: list | None = None) -> dict:
    return {
        "agent_data": {
            "work": {"status": "completed", "metrics": work_metrics},
            "history": history or [],
            "plan": {"learning_rate": 0.001},
            "eval": {}, "summary": {},
            "agent_params": {"max_iteration": max_iter, "visualize": False},
            "agent_state": {"iteration": iteration},
        },
        "errors": [],
    }


def print_result(result: dict, scenario: str):
    """打印评估结果"""
    ev = result["agent_data"]["eval"]
    print(f"\n  📊 {scenario} 评估结果:")
    print(f"      metrics       = {ev.get('metrics')}")
    print(f"      trend         = {'improving' if '持续改善' in ev.get('analysis','') else 'plateau' if '趋于平稳' in ev.get('analysis','') else 'worsening' if '恶化' in ev.get('analysis','') else '?'}")
    print(f"      next_action   = {result.get('next_action')}")
    print(f"      adjustments   = {ev.get('param_adjustments')}")
    print(f"      summary       = {ev.get('summary')}")
    print(f"      history 条数   = {len(result['agent_data']['history'])}")


# ============================================================
# 测试场景
# ============================================================

def test_first_round_improving():
    """场景 1：首轮评估，无历史 → improving → 继续迭代"""
    section("场景 1：首轮评估 → improving → work")

    print(f"  ▶ 条件: iteration=1, max_iter=3, history=[]")
    print(f"         work_metrics = {{'mse': 0.25, 'mae': 0.35}}")
    print()

    agent = make_agent()
    state = build_state(iteration=1, max_iter=3,
                        work_metrics={"mse": 0.25, "mae": 0.35},
                        history=[])

    print(f"  🔄 EvalAgent.run():")
    print(f"     1. _extract_metrics() → 提取 MSE=0.25, MAE=0.35")
    print(f"     2. _assess_trend() → 无历史，默认 'improving'")
    print(f"     3. _generate_eval() → LLM 不可用，走规则")
    print(f"     4. 决定 next_action → iteration(1) < max(3) → 'work'")
    print()

    result = agent.run(state)
    print_result(result, "首轮")

    ev = result["agent_data"]["eval"]
    assert ev["metrics"]["mse"] == 0.25
    assert result["next_action"] == "work"
    assert "持续改善" in ev["analysis"]
    print(f"\n  ✅ 首轮 improving → next_action=work")


def test_continuous_improvement():
    """场景 2：持续改善 → 学习率递减"""
    section("场景 2：持续改善 → 学习率递减")

    history = [
        {"iteration": 1, "metrics": {"mse": 0.30, "mae": 0.40},
         "eval_summary": "第 1 轮 MSE=0.30"},
        {"iteration": 2, "metrics": {"mse": 0.22, "mae": 0.32},
         "eval_summary": "第 2 轮 MSE=0.22"},
    ]

    print(f"  ▶ 条件: iteration=3, max_iter=5")
    print(f"         历史 MSE: [0.30, 0.22], 当前 MSE=0.15 (持续下降)")
    print()

    agent = make_agent()
    state = build_state(iteration=3, max_iter=5,
                        work_metrics={"mse": 0.15, "mae": 0.25},
                        history=history)

    print(f"  🔄 趋势判断:")
    print(f"     历史最优 MSE = min(0.30, 0.22) = 0.22")
    print(f"     当前 MSE     = 0.15")
    print(f"     0.15 < 0.22×0.95=0.209 → improving")
    print()

    result = agent.run(state)
    print_result(result, "持续改善")

    ev = result["agent_data"]["eval"]
    assert ev["metrics"]["mse"] == 0.15
    assert result["next_action"] == "work"
    assert "持续改善" in ev["analysis"]
    assert "learning_rate" in ev["param_adjustments"]  # 有参数调整
    print(f"\n  ✅ 持续改善 → lr 调整建议, next_action=work")


def test_plateau():
    """场景 3：指标停滞 → plateau → 加大调整"""
    section("场景 3：指标停滞 → plateau")

    history = [
        {"iteration": 1, "metrics": {"mse": 0.20, "mae": 0.30}},
        {"iteration": 2, "metrics": {"mse": 0.19, "mae": 0.29}},
    ]

    print(f"  ▶ 条件: iteration=3, max_iter=5")
    print(f"         历史 MSE: [0.20, 0.19], 当前 MSE=0.19 (停滞)")
    print()

    agent = make_agent()
    state = build_state(iteration=3, max_iter=5,
                        work_metrics={"mse": 0.19, "mae": 0.29},
                        history=history)

    print(f"  🔄 趋势判断:")
    print(f"     历史最优 MSE = min(0.20, 0.19) = 0.19")
    print(f"     当前 MSE     = 0.19")
    print(f"     0.19 在 [0.19×0.95, 0.19×1.05] = [0.181, 0.200] 内 → plateau")
    print()

    result = agent.run(state)
    print_result(result, "停滞")

    ev = result["agent_data"]["eval"]
    assert "趋于平稳" in ev["analysis"]
    assert result["next_action"] == "work"
    print(f"\n  ✅ 停滞 → 加大调整幅度, next_action=work")


def test_worsening():
    """场景 4：指标恶化 → worsening → 提前结束"""
    section("场景 4：指标恶化 → 提前结束")

    history = [
        {"iteration": 1, "metrics": {"mse": 0.15, "mae": 0.25}},
        {"iteration": 2, "metrics": {"mse": 0.14, "mae": 0.24}},
    ]

    print(f"  ▶ 条件: iteration=3, max_iter=5（还有 2 轮空间）")
    print(f"         历史最优 MSE=0.14, 当前 MSE=0.28 (回升)")
    print()

    agent = make_agent()
    state = build_state(iteration=3, max_iter=5,
                        work_metrics={"mse": 0.28, "mae": 0.38},
                        history=history)

    print(f"  🔄 趋势判断:")
    print(f"     历史最优 MSE = 0.14")
    print(f"     当前 MSE     = 0.28")
    print(f"     0.28 > 0.14×1.05=0.147 → worsening → 提前结束")
    print()

    result = agent.run(state)
    print_result(result, "恶化")

    ev = result["agent_data"]["eval"]
    assert "恶化" in ev["analysis"]
    assert result["next_action"] == "summary"  # 提前结束
    assert ev["param_adjustments"] == {}  # 无调整必要
    print(f"\n  ✅ 恶化 → 提前结束（summary），无调整建议")


def test_max_iter_reached():
    """场景 5：达到最大迭代 → 结束"""
    section("场景 5：达到最大迭代 → summary")

    history = [
        {"iteration": 1, "metrics": {"mse": 0.30, "mae": 0.40}},
        {"iteration": 2, "metrics": {"mse": 0.20, "mae": 0.30}},
    ]

    print(f"  ▶ 条件: iteration=3, max_iter=3（已用完）")
    print(f"         当前 MSE=0.15（仍在改善，但轮数用完）")
    print()

    agent = make_agent()
    state = build_state(iteration=3, max_iter=3,
                        work_metrics={"mse": 0.15, "mae": 0.25},
                        history=history)

    print(f"  🔄 判断逻辑:")
    print(f"     趋势: improving（0.15 < 0.20×0.95）")
    print(f"     iteration(3) < max_iter(3) → False → summary")
    print()

    result = agent.run(state)
    print_result(result, "达上限")

    assert result["next_action"] == "summary"
    print(f"\n  ✅ 已达最大迭代 → next_action=summary")


def test_llm_success():
    """场景 6：LLM 正常返回分析"""
    section("场景 6：LLM 正常返回分析")

    llm_response = {
        "eval": {
            "metrics": {"mse": 0.12, "mae": 0.22},
            "analysis": "模型第 3 轮 MSE 降至 0.12，loss 曲线持续下降，"
                        "建议保持当前学习率继续训练 2 轮观察收敛。",
            "param_adjustments": {"learning_rate": 1e-4},
            "summary": "第 3 轮评估：MSE=0.12，建议继续训练",
        },
        "next_action": "work",
    }

    history = [
        {"iteration": 1, "metrics": {"mse": 0.30, "mae": 0.40}},
        {"iteration": 2, "metrics": {"mse": 0.20, "mae": 0.30}},
    ]

    print(f"  ▶ 条件: LLM 返回完整 eval JSON")
    print(f"         当前 MSE=0.12, history=2 条")
    print()

    agent = make_agent(llm_response=llm_response, enable_llm=True)
    state = build_state(iteration=3, max_iter=5,
                        work_metrics={"mse": 0.12, "mae": 0.22},
                        history=history)

    result = agent.run(state)
    ev = result["agent_data"]["eval"]

    print(f"  📦 输出:")
    print(f"      analysis = {ev['analysis']}")
    print(f"      summary  = {ev['summary']}")
    print(f"      result   = {result['next_action']}")

    assert "建议保持当前学习率" in ev["analysis"]
    assert result["next_action"] == "work"
    print(f"\n  ✅ LLM 分析文本完整写入 state")


def test_llm_invalid_json():
    """场景 7：LLM 非法 JSON → 规则回退"""
    section("场景 7：LLM 非法 JSON → 规则回退")

    class BrokenLLM:
        def invoke(self, messages):
            print(f"    ╔══ LLM 返回非法内容 ══╗")
            print(f"    ║  不是合法 JSON")
            print(f"    ╚══ ══╝")
            class R:
                content = "模型表现不错，继续训练```"
            return R()

    history = [
        {"iteration": 1, "metrics": {"mse": 0.30, "mae": 0.40}},
    ]

    print(f"  ▶ 条件: LLM 返回非 JSON 文本")
    print(f"         触发 json.JSONDecodeError → 规则兜底")
    print()

    agent = EvalAgent.__new__(EvalAgent)
    agent.metric_skill = None
    agent._llm = BrokenLLM()

    state = build_state(iteration=2, max_iter=5,
                        work_metrics={"mse": 0.20, "mae": 0.30},
                        history=history)

    result = agent.run(state)
    ev = result["agent_data"]["eval"]

    print(f"  📦 输出（规则回退）:")
    print(f"      analysis = {ev['analysis']}")
    print(f"      summary  = {ev['summary']}")

    assert "持续改善" in ev["analysis"]  # 规则文本特征
    assert result["next_action"] == "work"
    print(f"\n  ✅ LLM 非法 JSON → 规则回退，分析文本来自规则")


# ============================================================
# 主入口
# ============================================================

def main():
    print()
    print(f"╔{'═' * 70}╗")
    print(f"║{'EvalAgent 流程测试 — 披露内部运转过程'.center(66)}║")
    print(f"║{'模拟 LLM，不依赖外部服务'.center(66)}║")
    print(f"╚{'═' * 70}╝")
    print()
    print(f"  覆盖场景:")
    print(f"    ├─ 场景 1: 首轮评估 → improving → 继续迭代")
    print(f"    ├─ 场景 2: 持续改善 → 学习率递减")
    print(f"    ├─ 场景 3: 指标停滞 → plateau → 加大调整")
    print(f"    ├─ 场景 4: 指标恶化 → 提前结束")
    print(f"    ├─ 场景 5: 达到最大迭代 → summary")
    print(f"    ├─ 场景 6: LLM 正常返回分析")
    print(f"    └─ 场景 7: LLM 非法 JSON → 规则回退")
    print()

    test_first_round_improving()
    test_continuous_improvement()
    test_plateau()
    test_worsening()
    test_max_iter_reached()
    test_llm_success()
    test_llm_invalid_json()

    print(f"\n{'═' * 72}")
    print(f"  ✅ 全部 7 个场景测试通过！")
    print(f"{'═' * 72}")
    print()


if __name__ == "__main__":
    main()
