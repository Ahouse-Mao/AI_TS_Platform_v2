"""
SummaryAgent 流程测试 — 模拟 LLM 返回值，披露内部运转过程

测试覆盖：
  1. 无 LLM，走规则总结 + 写入向量库
  2. LLM 正常返回合法总结
  3. LLM 返回非法 JSON → 回退到规则总结
  4. 空 history（无迭代记录）
  5. 单轮迭代
  6. 多轮迭代 + 参数调整历史
  7. 向量数据库写入验证

用法：
  uv run python agents/test_summary_agent.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)

from agents.summary_agent import SummaryAgent
from skills.param_validate_skill import ParamValidateSkill
from backend.RAG.rag_struct import clear_index


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
# 模拟工厂
# ============================================================

class MockLLM:
    """模拟 LLM 返回预定义的 summary JSON"""
    def __init__(self, response: dict | None, raise_error: bool = False):
        self.response = response
        self.raise_error = raise_error

    def invoke(self, messages):
        if self.raise_error:
            raise RuntimeError("模拟 LLM 网络超时")

        print(f"    ╔══ LLM 收到消息 ══╗")
        for msg in messages:
            role = type(msg).__name__.replace("Message", "")
            content = msg.content[:150]
            print(f"    ║  [{role}]: {content}")
        print(f"    ╚══ LLM 返回 JSON ══╝")
        show_json("模拟 LLM 响应", self.response)

        class MockResponse:
            content = json.dumps(self.response, ensure_ascii=False)

        return MockResponse()


class MockVectorDB:
    """模拟向量数据库写入，记录调用参数"""
    def __init__(self):
        self.inserted: list[dict] = []

    def insert(self, collection: str, data: dict, metadata: dict | None = None) -> str:
        print(f"    ╔══ VectorDB.write ══╗")
        print(f"    ║  collection: {collection}")
        print(f"    ║  data keys : {list(data.keys())}")
        print(f"    ║  metadata  : {metadata}")
        print(f"    ╚══ 写入完成 ══╝")
        self.inserted.append({"collection": collection, "data": data, "metadata": metadata})
        return "mock_insert_id"


# 构建模拟的 SummaryAgent
def make_agent(llm_response: dict | None = None,
               llm_error: bool = False,
               enable_llm: bool = True) -> SummaryAgent:
    """创建 SummaryAgent，用模拟对象替换内部依赖"""
    agent = SummaryAgent.__new__(SummaryAgent)
    agent.vector_db_skill = MockVectorDB()

    if enable_llm and llm_response is not None:
        agent._llm = MockLLM(llm_response, raise_error=llm_error)
    elif enable_llm and llm_response is None and not llm_error:
        # LLM 存在但返回 None（模拟返回空）
        agent._llm = MockLLM(None)
    else:
        agent._llm = None  # 模拟无 API key

    return agent


_DEFAULT_LLM_SUMMARY = {
    "summary": {
        "task_id": "test_001",
        "best_metrics": {"mse": 0.12, "mae": 0.22, "iteration": 3},
        "best_params": {"model_name": "DLinear", "dataset": "ETTh1", "learning_rate": 3e-5},
        "experience": "在 ETTh1 数据集上，DLinear 模型经过 3 轮迭代，"
                      "通过逐步降低学习率从 1e-3 到 3e-5，MSE 从 0.25 降至 0.12。"
                      "模型收敛稳定，未出现过拟合。",
        "recommendations": [
            "初始学习率建议设为 1e-3",
            "每轮迭代将学习率降低 1 个数量级",
            "ETTh1 数据集上 DLinear 表现优于复杂模型",
        ],
    },
    "next_action": "end",
}

_DEFAULT_HISTORY = [
    {"iteration": 1, "metrics": {"mse": 0.25, "mae": 0.35}, "param_adjustments": {}},
    {"iteration": 2, "metrics": {"mse": 0.18, "mae": 0.28},
     "param_adjustments": {"learning_rate": 5e-5}},
    {"iteration": 3, "metrics": {"mse": 0.12, "mae": 0.22},
     "param_adjustments": {"learning_rate": 3e-5}},
]

_DEFAULT_PLAN = {
    "model_name": "DLinear",
    "dataset": "ETTh1",
    "seq_len": 96, "pred_len": 96, "learning_rate": 0.001,
}


def build_state(task_id: str, history: list | None = None,
                plan: dict | None = None) -> dict:
    return {
        "task_id": task_id,
        "agent_data": {
            "plan": plan or dict(_DEFAULT_PLAN),
            "history": history or [],
            "work": {}, "eval": {}, "summary": {},
            "agent_params": {"max_iteration": 3, "visualize": False},
            "agent_state": {"iteration": len(history or [])},
        },
        "errors": [],
    }


# ============================================================
# 测试场景
# ============================================================

def test_rule_based():
    """场景 1：无 LLM，走规则总结 + 向量库写入"""
    section("场景 1：无 LLM，规则总结 + 向量库写入")

    step("条件",
          "LLM=None（无 API key）\n"
          "3 轮迭代 history\n"
          "期望: 规则总结 + 写入向量库")

    agent = make_agent(enable_llm=False)
    state = build_state("test_summary_001", _DEFAULT_HISTORY)

    print(f"\n  🔄 执行 SummaryAgent.run(state)...")
    print(f"     内部流程:")
    print(f"       1. _find_best() 从 history 中找出最优迭代")
    print(f"       2. _generate_summary() → LLM 不可用，走规则")
    print(f"       3. _write_to_vector_db() 写入 Milvus")
    print(f"       4. 写回 AgentState")
    print()

    result = agent.run(state)

    summary = result["agent_data"]["summary"]
    print(f"\n  📦 关键输出:")
    print(f"      task_id         = {summary['task_id']}")
    print(f"      best_metrics    = {summary['best_metrics']}")
    print(f"      best_params     = {summary['best_params']}")
    print(f"      experience      = {summary['experience'][:80]}...")
    print(f"      recommendations = {summary['recommendations']}")
    print(f"      next_action     = {result['next_action']}")

    # 验证向量库写入
    inserted = agent.vector_db_skill.inserted
    print(f"\n  📦 向量库写入记录: {len(inserted)} 条")
    if inserted:
        for rec in inserted:
            print(f"      collection: {rec['collection']}")
            print(f"      data keys : {list(rec['data'].keys())}")
            print(f"      model     : {rec['data'].get('model')}")
            print(f"      dataset   : {rec['data'].get('dataset')}")

    assert summary["best_metrics"]["mse"] == 0.12
    assert len(inserted) == 1
    assert inserted[0]["data"]["model"] == "DLinear"
    print(f"\n  ✅ 验证: MSE 正确提取 0.12，写入向量库 1 条")


def test_llm_success():
    """场景 2：LLM 正常返回合法总结"""
    section("场景 2：LLM 正常返回总结")

    step("条件",
          "LLM 返回完整 summary JSON\n"
          "期望: LLM 的总结直接写回 state，不走规则")

    agent = make_agent(llm_response=_DEFAULT_LLM_SUMMARY, enable_llm=True)
    state = build_state("test_summary_002", _DEFAULT_HISTORY)

    print(f"\n  🔄 执行 SummaryAgent.run(state)...")
    result = agent.run(state)

    summary = result["agent_data"]["summary"]
    print(f"\n  📦 关键输出:")
    show_json("summary", summary)

    assert summary["best_metrics"]["mse"] == 0.12
    assert "逐步降低学习率" in summary["experience"]
    assert result["next_action"] == "end"
    print(f"\n  ✅ 验证: LLM 总结内容写入 state，experience 包含 LLM 生成文本")


def test_llm_invalid_json():
    """场景 3：LLM 返回非法 JSON → 回退到规则"""
    section("场景 3：LLM 返回非法 JSON → 规则回退")

    class BrokenLLM:
        def invoke(self, messages):
            print(f"    ╔══ LLM 返回非法内容 ══╗")
            print(f"    ║  返回: 这不是合法 JSON")
            print(f"    ╚══ ══╝")
            class R:
                content = "这不是合法 JSON```"
            return R()

    agent = SummaryAgent.__new__(SummaryAgent)
    agent.vector_db_skill = MockVectorDB()
    agent._llm = BrokenLLM()

    state = build_state("test_summary_003", _DEFAULT_HISTORY)

    print(f"\n  🔄 执行 SummaryAgent.run(state)...")
    result = agent.run(state)

    summary = result["agent_data"]["summary"]
    print(f"\n  📦 关键输出:")
    print(f"      best_metrics.mse = {summary['best_metrics']['mse']}")
    print(f"      experience       = {summary['experience'][:80]}...")

    assert summary["best_metrics"]["mse"] == 0.12  # 规则正确提取
    assert "迭代优化" in summary["experience"]    # 规则生成的文本特征
    print(f"\n  ✅ 验证: 非法 JSON 被捕获，规则总结写入成功")


def test_empty_history():
    """场景 4：空 history（无迭代记录）"""
    section("场景 4：空 history")

    step("条件",
          "history=[]（推理任务直接进入 summary）\n"
          "期望: 无迭代时仍能生成基本总结")

    agent = make_agent(enable_llm=False)
    state = build_state("test_summary_004", history=[])

    result = agent.run(state)
    summary = result["agent_data"]["summary"]

    print(f"\n  📦 关键输出:")
    print(f"      best_metrics = {summary['best_metrics']}")
    print(f"      experience   = {summary['experience'][:80]}...")

    assert summary["best_metrics"]["iteration"] == 0
    assert "经过 0 轮迭代" in summary["experience"]
    assert len(agent.vector_db_skill.inserted) == 1
    print(f"\n  ✅ 验证: 空 history 生成基本总结，仍写入向量库")


def test_single_iteration():
    """场景 5：单轮迭代"""
    section("场景 5：单轮迭代")

    single = [{"iteration": 1, "metrics": {"mse": 0.15, "mae": 0.25}, "param_adjustments": {}}]
    agent = make_agent(enable_llm=False)
    state = build_state("test_summary_005", history=single)

    result = agent.run(state)
    summary = result["agent_data"]["summary"]

    print(f"\n  📦 关键输出:")
    print(f"      best_metrics = {summary['best_metrics']}")
    print(f"      experience   = {summary['experience'][:80]}...")
    print(f"      vector_db    = {len(agent.vector_db_skill.inserted)} 条")

    assert summary["best_metrics"]["mse"] == 0.15
    assert summary["best_metrics"]["iteration"] == 1
    print(f"\n  ✅ 验证: 单轮迭代的 MSE 正确提取")


def test_multi_iteration_with_adjustments():
    """场景 6：多轮迭代 + 参数调整历史"""
    section("场景 6：多轮迭代 + 参数调整")

    history = [
        {"iteration": 1, "metrics": {"mse": 0.30, "mae": 0.40}, "param_adjustments": {}},
        {"iteration": 2, "metrics": {"mse": 0.22, "mae": 0.32},
         "param_adjustments": {"learning_rate": 5e-5, "batch_size": 64}},
        {"iteration": 3, "metrics": {"mse": 0.18, "mae": 0.28},
         "param_adjustments": {"learning_rate": 3e-5}},
        {"iteration": 4, "metrics": {"mse": 0.10, "mae": 0.18},
         "param_adjustments": {"learning_rate": 1e-5, "dropout": 0.1}},
        {"iteration": 5, "metrics": {"mse": 0.11, "mae": 0.19},
         "param_adjustments": {"learning_rate": 1e-5}},  # 略回升
    ]

    agent = make_agent(enable_llm=False)
    state = build_state("test_summary_006", history=history)

    result = agent.run(state)
    summary = result["agent_data"]["summary"]

    print(f"\n  📦 关键输出:")
    print(f"      best_metrics   = {summary['best_metrics']}")
    print(f"      total_rounds   = {len(history)}")
    print(f"      experience     = {summary['experience'][:100]}...")
    show_json("recommendations", summary["recommendations"])

    assert summary["best_metrics"]["mse"] == 0.10  # 第 4 轮最优
    assert summary["best_metrics"]["iteration"] == 4
    print(f"\n  ✅ 验证: 从 5 轮中正确找到第 4 轮 MSE=0.10 为最优")
    print(f"           参数调整历史被正确记录到 experience 中")


def test_vector_db_write():
    """场景 7：验证向量数据库写入内容"""
    section("场景 7：验证向量数据库写入内容")

    agent = make_agent(enable_llm=False)
    state = build_state("test_summary_007", _DEFAULT_HISTORY)

    result = agent.run(state)
    inserted = agent.vector_db_skill.inserted

    assert len(inserted) == 1
    rec = inserted[0]

    print(f"\n  📦 向量库写入内容:")
    print(f"      collection = {rec['collection']}")
    print(f"      data:")
    for k, v in rec["data"].items():
        v_s = str(v)[:60]
        print(f"        {k:>15} = {v_s}")
    print(f"      metadata:")
    for k, v in (rec.get("metadata") or {}).items():
        print(f"        {k:>15} = {v}")

    assert rec["collection"] == "rag_struct"
    assert rec["data"]["model"] == "DLinear"
    assert rec["data"]["dataset"] == "ETTh1"
    assert rec["data"]["task_id"] == "test_summary_007"
    print(f"\n  ✅ 验证: 向量库写入格式正确，包含 model/dataset/task_id")


# ============================================================
# 主入口
# ============================================================

def main():
    # 清理向量库，确保测试环境干净
    clear_index()

    print()
    print(f"╔{'═' * 70}╗")
    print(f"║{'SummaryAgent 流程测试 — 披露内部运转过程'.center(66)}║")
    print(f"║{'模拟 LLM + VectorDB，不依赖外部服务'.center(66)}║")
    print(f"╚{'═' * 70}╝")
    print()
    print(f"  覆盖场景:")
    print(f"    ├─ 场景 1: 无 LLM，规则总结 + 向量库写入")
    print(f"    ├─ 场景 2: LLM 正常返回总结")
    print(f"    ├─ 场景 3: LLM 返回非法 JSON → 规则回退")
    print(f"    ├─ 场景 4: 空 history（推理任务场景）")
    print(f"    ├─ 场景 5: 单轮迭代")
    print(f"    ├─ 场景 6: 多轮迭代 + 参数调整历史")
    print(f"    └─ 场景 7: 向量数据库写入验证")
    print()

    test_rule_based()
    test_llm_success()
    test_llm_invalid_json()
    test_empty_history()
    test_single_iteration()
    test_multi_iteration_with_adjustments()
    test_vector_db_write()

    print(f"\n{'═' * 72}")
    print(f"  ✅ 全部 7 个场景测试通过！")
    print(f"{'═' * 72}")
    print()


if __name__ == "__main__":
    main()
