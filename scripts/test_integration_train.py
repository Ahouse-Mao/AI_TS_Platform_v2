"""
集成测试 1：单次训练 —— plan → work → end

使用真实 LLM API，模拟后端训练 API，LangGraph 完整链路。

用法： uv run python scripts/test_integration_train.py
"""

import sys, os, json, logging
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

sep = lambda t: print(f"\n{'─' * 60}\n  {t}\n{'─' * 60}")


def fake_train(params):
    m, d = params.get("model_name"), params.get("dataset")
    print(f"  ╔ POST /api/train  model={m}  dataset={d}")
    print(f"  ║  lr={params.get('learning_rate')}  epochs={params.get('epochs')}")
    print(f"  ║  …轮询等待…")
    print(f"  ╚ 训练完成")
    return {"status": "completed", "task_id": "t1",
            "checkpoint_path": f"/ckpts/{m}_{d}/ckpt.pth", "log_path": "r.txt",
            "metrics": {"mse": 0.152, "mae": 0.253,
                        "train_loss": [0.5,0.35,0.25,0.18,0.16,0.155,0.153,0.152],
                        "val_loss": [0.6,0.45,0.35,0.28,0.25,0.24,0.235,0.23]}}


def main():
    print(f"\n╔{'═' * 60}╗")
    print(f"║  集成测试 1：单次训练  plan → work → end     ║")
    print(f"╚{'═' * 60}╝")

    from scripts._test_util import patch_logging_llm

    with patch_logging_llm(), \
         mock.patch("skills.api_skill.APISkill.run_training", side_effect=fake_train), \
         mock.patch("skills.rag_skill.RAGSkill.search", return_value=[]):

        import conf.llm as conf_llm
        from main import TSPlatform
        p = TSPlatform()

        print(f"\n  用户: '用 DLinear 在 ETTh1 上训练'")
        print(f"  路由: plan → work → END（max_iter=1，不迭代）")
        print(f"  LLM: {conf_llm._MODEL_NORMAL} @ {conf_llm._BASE_URL}")

        state = p.run("用 DLinear 在 ETTh1 上训练", max_iteration=1)

        sep("最终 State")
        ad = state["agent_data"]
        print(f"  status={state['status']}  agent={state['agent']}  next={state['next_action']}")

        plan, work = ad["plan"], ad["work"]
        print(f"  [plan] {plan['model_name']}/{plan['dataset']}"
              f"  seq={plan['seq_len']}  pred={plan['pred_len']}"
              f"  lr={plan['learning_rate']}")
        print(f"  [work] status={work['status']}  mse={work['metrics']['mse']:.4f}"
              f"  ckpt={work['checkpoint_path'][:35]}…")
        print(f"  errors={state.get('errors', [])}")

        assert state["status"] in ("success", "completed")
        assert work["metrics"]["mse"] == 0.152
        print(f"\n  ✅ 单次训练测试通过！")


if __name__ == "__main__":
    main()
