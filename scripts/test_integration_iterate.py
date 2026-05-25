"""
集成测试 2：迭代训练 —— plan → work → eval ⇄ work → … → summary

模拟 LLM 和后端 API，LangGraph 完整迭代链路展示数据流转。

用法： uv run python scripts/test_integration_iterate.py
"""

import sys, os, json, logging
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

sep = lambda t: print(f"\n{'─' * 60}\n  {t}\n{'─' * 60}")


class MultiMockLLM:
    """多轮 LLM 模拟：按顺序返回不同 JSON"""
    def __init__(self, responses: list[dict]):
        self.resps = responses
        self.idx = 0

    def invoke(self, msgs):
        d = self.resps[min(self.idx, len(self.resps) - 1)]
        self.idx += 1
        print(f"  ╔ LLM #{self.idx}: {type(msgs[0]).__name__[:10]}")
        print(f"  ╚ 返回: {json.dumps(d, ensure_ascii=False)[:120]}")
        class R: content = json.dumps(d, ensure_ascii=False)
        return R()


# 模拟训练 API：每次 MSE 递减
_train_round = 0

def fake_train_improving(params):
    global _train_round
    _train_round += 1
    mse = [0.35, 0.20, 0.12][_train_round - 1] if _train_round <= 3 else 0.10
    m = params.get("model_name")
    print(f"  ╔ POST /api/train  model={m}  lr={params.get('learning_rate'):.0e}")
    print(f"  ║  … 第 {_train_round} 轮训练 …  MSE≈{mse:.2f}")
    print(f"  ╚ 完成")
    return {"status": "completed", "task_id": f"t{_train_round}",
            "checkpoint_path": f"/ckpts/{m}/r{_train_round}.pth", "log_path": "",
            "metrics": {"mse": mse, "mae": mse + 0.1}}


def main():
    global _train_round
    _train_round = 0

    print(f"\n╔{'═' * 60}╗")
    print(f"║  集成测试 2：迭代优化  plan→work→eval⇄work→…→summary ║")
    print(f"╚{'═' * 60}╝")

    # LLM 按调用顺序返回：
    #   #1 PlanAgent    → 初始参数 + max_iter=3
    #   #2 EvalAgent  第1轮 → 分析 + 降低 lr
    #   #3 EvalAgent  第2轮 → 继续分析
    #   #4 EvalAgent  第3轮 → 改善趋缓，建议结束
    #   #5 SummaryAgent → 最终总结
    llm = MultiMockLLM([
        {  # Plan
            "intent": "train", "plan": {"model_name": "DLinear", "dataset": "ETTh1",
                "seq_len": 96, "pred_len": 96, "batch_size": 64, "learning_rate": 0.01,
                "epochs": 3, "d_model": 512, "n_heads": 8, "e_layers": 2, "d_layers": 1,
                "d_ff": 2048, "dropout": 0.1, "features": "M", "use_gpu": True},
            "agent_params": {"max_iteration": 3, "visualize": False}, "next_action": "work"},
        {  # Eval round 1
            "eval": {"metrics": {"mse": 0.35, "mae": 0.45},
                "analysis": "首轮 MSE=0.35，偏高。建议降低学习率一个数量级。",
                "param_adjustments": {"learning_rate": 0.001},
                "summary": "第1轮：MSE=0.35 → 降低lr"}},
        {  # Eval round 2
            "eval": {"metrics": {"mse": 0.20, "mae": 0.30},
                "analysis": "第2轮 MSE=0.20，改善明显，建议继续降低lr。",
                "param_adjustments": {"learning_rate": 1e-4},
                "summary": "第2轮：MSE=0.20 → 降低lr"}},
        {  # Eval round 3
            "eval": {"metrics": {"mse": 0.12, "mae": 0.22},
                "analysis": "第3轮 MSE=0.12，改善趋缓，趋于收敛。建议结束迭代。",
                "param_adjustments": {},
                "summary": "第3轮：MSE=0.12 → 收敛"}},
        {  # Summary
            "summary": {"task_id": "t_sum", "best_metrics": {"mse": 0.12, "mae": 0.22, "iteration": 3},
                "best_params": {"learning_rate": 1e-4},
                "experience": "3轮迭代：MSE从0.35降至0.12，逐次降低lr效果显著。DLinear在ETTh1上表现稳定。",
                "recommendations": ["初始lr=1e-2，逐轮降低一个数量级"]}},
    ])

    from scripts._test_util import patch_logging_llm

    with patch_logging_llm(), \
         mock.patch("skills.api_skill.APISkill.run_training", side_effect=fake_train_improving), \
         mock.patch("skills.rag_skill.RAGSkill.search", return_value=[]), \
         mock.patch("skills.vector_db_skill.VectorDBSkill.insert", return_value="mock_id"):

        import conf.llm as conf_llm
        from main import TSPlatform
        p = TSPlatform()

        print(f"\n  用户: '用 DLinear 在 ETTh1 上迭代优化'")
        print(f"  路由: plan → work → eval ⇄ work → ... → summary → END")
        print(f"  LLM: {conf_llm._MODEL_NORMAL} @ {conf_llm._BASE_URL}")

        state = p.run("用 DLinear 在 ETTh1 上迭代优化", max_iteration=3)

        sep("最终 State")
        ad = state["agent_data"]
        h = ad.get("history", [])
        s = ad.get("summary", {})

        print(f"  status={state['status']}  agent={state['agent']}  next={state['next_action']}")
        print(f"  history 记录: {len(h)} 轮")
        for i, entry in enumerate(h):
            m = entry['metrics'].get('mse', '?')
            print(f"    第{entry['iteration']}轮: mse={m}"
                  f"  {entry.get('eval_summary', '')[:40]}")
        print(f"  summary: best_mse={s.get('best_metrics',{}).get('mse')}  "
              f"exp={s.get('experience','')[:60]}…")
        print(f"  errors={state.get('errors', [])}")

        assert state["status"] in ("success", "completed")
        assert len(h) >= 3  # 至少 3 轮评估
        assert state["agent"] == "summary"
        print(f"\n  ✅ 迭代训练测试通过！")


if __name__ == "__main__":
    main()
