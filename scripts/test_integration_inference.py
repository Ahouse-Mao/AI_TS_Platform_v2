"""
集成测试 3：直接推理 —— plan → work → summary → end

模拟 LLM 和后端 API，LangGraph 完整推理链路展示数据流转。

用法： uv run python scripts/test_integration_inference.py
"""

import sys, os, json, logging
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

sep = lambda t: print(f"\n{'─' * 60}\n  {t}\n{'─' * 60}")


class MultiMockLLM:
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


def fake_infer(params):
    m, d = params.get("model_name"), params.get("dataset")
    cp = params.get("checkpoint_path", "(auto)")
    print(f"  ╔ POST /api/infer  model={m}  dataset={d}")
    print(f"  ║  checkpoint={cp}")
    print(f"  ║  … 推理中 …")
    print(f"  ╚ 完成")
    return {"status": "completed", "task_id": "inf_001",
            "checkpoint_path": cp, "log_path": "",
            "predictions_path": "pred.npy",
            "metrics": {"mse": 0.145, "mae": 0.240}}


def main():
    print(f"\n╔{'═' * 60}╗")
    print(f"║  集成测试 3：直接推理  plan→work→summary→end  ║")
    print(f"╚{'═' * 60}╝")

    llm = MultiMockLLM([
        {  # Plan: 识别为推理意图
            "intent": "inference", "plan": {"model_name": "DLinear", "dataset": "ETTh1",
                "seq_len": 96, "pred_len": 96, "features": "M", "use_gpu": True,
                "checkpoint_path": "/ckpts/DLinear_ETTh1/ckpt.pth"},
            "agent_params": {"max_iteration": 1, "visualize": False},
            "next_action": "work"},
        {  # Summary
            "summary": {"task_id": "inf_sum", "best_metrics": {"mse": 0.145, "mae": 0.24, "iteration": 0},
                "best_params": {"model_name": "DLinear", "dataset": "ETTh1"},
                "experience": "推理任务：DLinear/ETTh1，MSE=0.145。模型表现良好，可直接用于预测。",
                "recommendations": ["该检查点可直接用于生产推理"]}},
    ])

    from scripts._test_util import patch_logging_llm

    with patch_logging_llm(), \
         mock.patch("skills.api_skill.APISkill.run_inference", side_effect=fake_infer), \
         mock.patch("skills.checkpoint_skill.CheckpointSkill.find_best",
                     return_value={"checkpoint_path": "/ckpts/DLinear_ETTh1/ckpt.pth"}), \
         mock.patch("skills.rag_skill.RAGSkill.search", return_value=[]), \
         mock.patch("skills.vector_db_skill.VectorDBSkill.insert", return_value="mock_id"):

        import conf.llm as conf_llm
        from main import TSPlatform
        p = TSPlatform()

        print(f"\n  用户: '用 DLinear 对 ETTh1 做推理预测'")
        print(f"  路由: plan → work → summary → END（推理跳过 eval）")
        print(f"  LLM: {conf_llm._MODEL_NORMAL} @ {conf_llm._BASE_URL}")

        state = p.run("用 DLinear 对 ETTh1 做推理预测", max_iteration=1)

        sep("最终 State")
        ad = state["agent_data"]

        print(f"  status={state['status']}  agent={state['agent']}")
        print(f"  intent={ad['intent']}")
        plan = ad["plan"]
        print(f"  [plan] {plan['model_name']}/{plan['dataset']}  ckpt={str(plan.get('checkpoint_path','?'))[:40]}…")
        work = ad["work"]
        print(f"  [work] status={work['status']}  mse={work['metrics']['mse']:.4f}")
        s = ad.get("summary", {})
        print(f"  [summary] exp={s.get('experience','')[:60]}…")
        print(f"  errors={state.get('errors', [])}")

        assert state["status"] == "success"
        assert ad["intent"] == "inference"
        assert work["status"] == "completed"
        assert work["metrics"]["mse"] == 0.145
        assert state["agent"] == "summary"
        print(f"\n  ✅ 推理测试通过！")


if __name__ == "__main__":
    main()
