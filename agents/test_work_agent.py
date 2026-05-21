"""
WorkAgent 端到端流程测试脚本

演示 WorkAgent 的完整运转流程，包含详细的中间状态打印。
测试覆盖两种核心路径：
  1. 训练任务：plan → work → (eval)
  2. 推理任务：plan → work → (summary)

用法：
  # 先启动后端服务
  uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000

  # 运行测试
  uv run python agents/test_work_agent.py
"""

import sys
import os
import json
import logging

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.work_agent import WorkAgent

# 日志配置：让 WorkAgent 内部的所有日志打印到终端
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================
# 辅助函数
# ============================================================

def print_separator(title: str):
    """打印分隔标题"""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_json(label: str, data: dict, indent: int = 2):
    """格式化打印 JSON"""
    print(f"  {label}:")
    print(f"    {json.dumps(data, ensure_ascii=False, indent=indent).replace(chr(10), chr(10) + '    ')}")


def build_state(intent: str, plan_params: dict, iteration: int = 0) -> dict:
    """构造一个模拟的 AgentState"""
    return {
        "status": "initial",
        "agent": "",
        "task_id": f"test_{intent}_{iteration}",
        "agent_data": {
            "intent": intent,
            "plan": plan_params,
            "eval": {},                              # 无 eval 建议
            "work": {},
            "summary": {},
            "agent_params": {
                "max_iteration": 3,
                "visualize": False,
            },
            "agent_state": {
                "iteration": iteration,
            },
            "history": [],
        },
        "errors": [],
        "next_action": "",
    }


def print_state_diff(before: dict, after: dict):
    """对比 state 的前后变化，只打印有改动的关键字段"""
    print(f"\n  📊 State 变化:")

    changed_keys = []
    for key in ["status", "agent", "next_action"]:
        if before.get(key) != after.get(key):
            changed_keys.append(f"{key}: {before.get(key)} → {after.get(key)}")

    if changed_keys:
        print(f"    {'  |  '.join(changed_keys)}")

    # agent_data 变化
    bd = before.get("agent_data", {})
    ad = after.get("agent_data", {})
    if bd.get("agent_state", {}).get("iteration") != ad.get("agent_state", {}).get("iteration"):
        print(f"    iteration: {bd.get('agent_state', {}).get('iteration')} → {ad.get('agent_state', {}).get('iteration')}")


def print_work_result(work: dict):
    """格式化打印 work 结果"""
    print(f"\n  📦 Work 结果:")
    print(f"    status           : {work.get('status')}")
    print(f"    error            : {work.get('error', '无')}")
    print(f"    checkpoint_path  : {work.get('checkpoint_path', '无')}")
    print(f"    log_path         : {work.get('log_path', '无')}")
    if work.get("predictions_path"):
        print(f"    predictions_path : {work.get('predictions_path')}")

    metrics = work.get("metrics", {})
    if metrics:
        print(f"    指标:")
        for key in ["mse", "mae", "rmse", "rse", "mape"]:
            if key in metrics:
                print(f"      {key:>10} = {metrics[key]:.6f}")
        for key in ["total_time"]:
            if key in metrics:
                print(f"      {key:>10} = {metrics[key]:.2f}s")
        for key in ["train_loss", "val_loss", "test_loss"]:
            vals = metrics.get(key, [])
            if vals:
                print(f"      {key:>10} = [{', '.join(f'{v:.4f}' for v in vals[:5])}{'...' if len(vals) > 5 else ''}] ({len(vals)} epochs)")


# ============================================================
# 测试场景 1：训练任务（短周期）
# ============================================================

def test_training():
    print_separator("场景 1：训练任务 (intent=train)")

    # ---- 1. 构造初始 State ----
    plan_params = {
        "model_name": "DLinear",
        "dataset": "ETTh1",
        "seq_len": 96,
        "pred_len": 96,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 2,            # 2 轮，快速验证
        "patience": 3,
        "features": "M",
        "use_gpu": True,
    }
    state = build_state(intent="train", plan_params=plan_params, iteration=0)
    print("\n  📝 Plan 参数:")
    for k, v in plan_params.items():
        print(f"    {k:>20} = {v}")

    # ---- 2. 创建 WorkAgent 并执行 ----
    print(f"\n  🚀 创建 WorkAgent...")
    agent = WorkAgent()
    print(f"      api_skill.base_url = {agent.api_skill.base_url}")
    print(f"      检查点目录         = {agent.checkpoint_skill.checkpoints_root}")

    print(f"\n  🔄 执行 WorkAgent.run(state)...")
    print(f"     这一步会依次执行:")
    print(f"       1. 合并 plan 参数")
    print(f"       2. 调用后端 POST /api/train")
    print(f"       3. 轮询等待训练完成")
    print(f"       4. 解析日志提取指标")
    print(f"       5. 更新 AgentState")
    print(f"     请耐心等待训练完成...\n")

    state_before = {k: v for k, v in state.items()}
    new_state = agent.run(state)

    # ---- 3. 打印结果 ----
    print_separator("训练结果")
    print_state_diff(state_before, new_state)
    work = new_state.get("agent_data", {}).get("work", {})
    print_work_result(work)

    print(f"\n  ➡️  next_action = '{new_state.get('next_action')}'")
    if new_state.get("next_action") == "eval":
        print(f"     → 接下来将进入 EvalAgent 评估阶段")
    elif new_state.get("next_action") == "summary":
        print(f"     → 接下来将进入 SummaryAgent 总结阶段")

    return new_state


# ============================================================
# 测试场景 2：推理任务
# ============================================================

def test_inference():
    print_separator("场景 2：推理任务 (intent=inference)")

    # ---- 1. 构造初始 State ----
    plan_params = {
        "model_name": "DLinear",
        "dataset": "ETTh1",
        "seq_len": 96,
        "pred_len": 96,
        "features": "M",
        "use_gpu": True,
        # 不传 checkpoint_path，让 WorkAgent 自动查找
    }
    state = build_state(intent="inference", plan_params=plan_params, iteration=0)
    print("\n  📝 Plan 参数:")
    for k, v in plan_params.items():
        print(f"    {k:>20} = {v}")
    print(f"    {'checkpoint_path':>20} = (自动查找)")

    # ---- 2. 创建 WorkAgent 并执行 ----
    print(f"\n  🚀 创建 WorkAgent...")
    agent = WorkAgent()

    print(f"\n  🔄 执行 WorkAgent.run(state)...")
    print(f"     这一步会依次执行:")
    print(f"       1. 检查 intent=inference")
    print(f"       2. 调用 CheckpointSkill.find_best() 搜索检查点")
    print(f"       3. 调用后端 POST /api/infer")
    print(f"       4. 轮询等待推理完成")
    print(f"       5. 更新 AgentState\n")

    state_before = {k: v for k, v in state.items()}
    new_state = agent.run(state)

    # ---- 3. 打印结果 ----
    print_separator("推理结果")
    print_state_diff(state_before, new_state)
    work = new_state.get("agent_data", {}).get("work", {})
    print_work_result(work)

    print(f"\n  ➡️  next_action = '{new_state.get('next_action')}'")
    if new_state.get("next_action") == "summary":
        print(f"     → 推理任务无需评估，直接进入 SummaryAgent 总结阶段")
    elif new_state.get("next_action") == "eval":
        print(f"     → 将进入 EvalAgent 评估阶段")

    return new_state


# ============================================================
# 测试场景 3：训练任务（含 eval 迭代建议）
# ============================================================

def test_training_with_eval():
    print_separator("场景 3：训练任务 — 带 Eval 建议 (模拟第 2 轮迭代)")

    plan_params = {
        "model_name": "DLinear",
        "dataset": "ETTh1",
        "seq_len": 96,
        "pred_len": 96,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 2,
    }

    # 模拟 EvalAgent 给出的优化建议
    eval_suggestions = {
        "metrics": {"mse": 0.4, "mae": 0.42},
        "analysis": "模型收敛良好，建议降低学习率继续训练",
        "param_adjustments": {
            "learning_rate": 0.0005,       # 学习率减半
            "batch_size": 64,              # 增大 batch
        },
    }

    state = build_state(intent="train", plan_params=plan_params, iteration=1)
    state["agent_data"]["eval"] = eval_suggestions
    state["agent_data"]["history"] = [
        {
            "iteration": 0,
            "metrics": {"mse": 0.5, "mae": 0.5},
            "param_adjustments": {},
        }
    ]

    print("\n  📝 Plan 参数:")
    for k, v in plan_params.items():
        print(f"    {k:>20} = {v}")

    print(f"\n  📝 Eval 建议:")
    adj = eval_suggestions.get("param_adjustments", {})
    for k, v in adj.items():
        print(f"    {k:>20} = {v} (覆盖 plan 中的 {plan_params.get(k, '未设置')})")

    # ---- 执行 ----
    print(f"\n  🔄 执行 WorkAgent.run(state)...")
    print(f"     注意：合并后的 learning_rate = 0.0005 (eval 覆盖 plan 的 0.001)\n")

    state_before = {k: v for k, v in state.items()}
    agent = WorkAgent()
    new_state = agent.run(state)

    # ---- 结果 ----
    print_separator("第 2 轮训练结果")
    print_state_diff(state_before, new_state)
    work = new_state.get("agent_data", {}).get("work", {})
    print_work_result(work)

    print(f"\n  ➡️  next_action = '{new_state.get('next_action')}'")
    return new_state


# ============================================================
# 主入口
# ============================================================

def main():
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "      WorkAgent 端到端流程测试".center(68) + "║")
    print("║" + "      AI-TS-Platform".center(68) + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    print("  测试内容:")
    print("    ├─ 场景 1: 训练任务 (epochs=2, 快速)")
    print("    ├─ 场景 2: 推理任务 (自动找检查点)")
    print("    └─ 场景 3: 训练 + Eval 建议 (参数合并)")
    print()
    print("  ⚠️  请确保后端服务已启动: uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000")
    print()

    # 先检查后端是否可达
    import requests
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        assert r.json().get("status") == "ok"
        print("  ✅ 后端服务已连接: http://localhost:8000")
    except Exception as e:
        print(f"  ❌ 后端服务不可达: {e}")
        print(f"     请先启动: uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # ---- 执行测试 ----
    results = {}

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + "  场景 1：训练任务".center(68) + "║")
    print("╚" + "═" * 70 + "╝")
    results["train"] = test_training()

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + "  场景 2：推理任务".center(68) + "║")
    print("╚" + "═" * 70 + "╝")
    results["inference"] = test_inference()

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + "  场景 3：训练 + Eval 建议".center(68) + "║")
    print("╚" + "═" * 70 + "╝")
    results["train_with_eval"] = test_training_with_eval()

    # ---- 汇总 ----
    print_separator("最终汇总")
    for name, state in results.items():
        work = state.get("agent_data", {}).get("work", {})
        metrics = work.get("metrics", {})
        print(f"  [{name:>18}]")
        print(f"    status       = {state.get('status')}")
        print(f"    next_action  = {state.get('next_action')}")
        print(f"    iteration    = {state.get('agent_data', {}).get('agent_state', {}).get('iteration')}")
        print(f"    work.status  = {work.get('status')}")
        print(f"    work.mse     = {metrics.get('mse', 'N/A')}")
        print()

    print("✅ 所有场景测试完成！")


if __name__ == "__main__":
    main()
