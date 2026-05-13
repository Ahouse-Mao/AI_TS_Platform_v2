"""
时间序列预测多智能体系统
基于 LangGraph 的工作流编排
统一数据格式：所有智能体均通过 state["message_json"] 通信
"""

import os
import json
import logging
from typing import TypedDict, List, Optional, Any, Dict, Literal
from datetime import datetime
from enum import Enum

# LangGraph 核心
from langgraph.graph import StateGraph, END

# LangChain LLM（可根据需要替换为其他兼容的模型）
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. 状态定义 —— 与统一数据格式对应
# =============================================================================

class AgentState(TypedDict, total=False):
    """全局工作流状态，各节点读写"""
    # 统一信封
    message_json: Dict[str, Any]          # 顶层数据总线
    # 快捷字段（从 message_json 中解包出来便于操作，但最终同步回 message_json）
    intent: str
    action: str
    task_id: str
    errors: List[str]
    next_action: str
    # 内部控制字段
    agent_params: Dict[str, Any]
    agent_state: Dict[str, Any]
    history: List[Dict[str, Any]]
    # 各 Agent 的专属数据
    plan_json: Optional[Dict[str, Any]]
    work_json: Optional[Dict[str, Any]]
    eval_json: Optional[Dict[str, Any]]
    summary_json: Optional[Dict[str, Any]]

# =============================================================================
# 2. LLM 工厂函数（用于生成 JSON 输出的 LLM）
# =============================================================================

def get_json_llm():
    """返回一个配置为输出 JSON 的 LLM 实例"""
    # 请根据你的环境变量配置 OPENAI_COMPATIBLE_MODEL 等参数
    return ChatOpenAI(
        model=os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}},
        max_tokens=4096,
    )

# =============================================================================
# 3. Skills 层 —— 原子能力（占位实现）
# =============================================================================

def rag_search_skill(query: str, top_k: int = 3) -> List[Dict]:
    """RAG 检索：从向量数据库中检索历史经验"""
    # 集成你的 Milvus Lite 或 ChromaDB
    logger.info(f"RAG 检索: {query}")
    # 占位返回
    return [{"experience": "示例经验：对于周期性数据 PatchTST 效果较好"}]

def model_checkpoint_skill(model_name: str) -> str:
    """查找最近的模型检查点地址"""
    logger.info(f"查找模型检查点: {model_name}")
    # 占位返回
    return f"/models/{model_name}/checkpoint.pth"

def train_api_skill(params: Dict) -> Dict:
    """调用后端训练 API"""
    logger.info(f"启动训练，参数: {params}")
    # 模拟返回训练日志路径
    return {"log_path": "/tmp/train_log.csv", "metrics": {"mse": 0.0023, "mae": 0.034}}

def inference_api_skill(model_path: str, data_path: str) -> Dict:
    """调用后端推理 API"""
    logger.info(f"启动推理，模型: {model_path}")
    # 模拟返回推理结果
    return {"predictions": [1,2,3], "plot_path": "/tmp/pred_plot.png"}

def log_parsing_skill(log_path: str) -> Dict:
    """解析训练日志文件"""
    # 实际可读取 CSV 或 TensorBoard 日志
    return {"final_train_loss": 0.01, "final_val_loss": 0.02}

def plot_skill(data: Dict, output_path: str) -> str:
    """可视化绘图"""
    logger.info(f"生成图表: {output_path}")
    return output_path

def vector_db_write_skill(collection: str, data: Dict):
    """将经验入库"""
    logger.info(f"写入向量数据库: {collection}")

def parameter_validation_skill(params: Dict) -> bool:
    """参数校验"""
    # 检查必填项、范围等
    return True

def metrics_skill(y_true: List, y_pred: List) -> Dict:
    """计算 MSE, MAE 等指标"""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"mse": mse, "mae": mae}

# =============================================================================
# 4. 节点函数实现（各 Agent）
# =============================================================================

def intent_agent(state: AgentState) -> AgentState:
    """
    意图识别 Agent：使用 LLM 分析用户查询，输出统一格式的意图 JSON
    填充 state["message_json"] 中的 intent 和 agent_params 等
    """
    llm = get_json_llm()
    user_query = state["intent"]  # 假设用户输入已存入 state["intent"]

    system_prompt = """你是一个时间序列预测系统的意图识别机器人。请根据用户输入判断意图，并以 JSON 格式返回：
{
  "action": "train" | "inference" | "auto_iter",
  "task_id": "<时间戳ID>",
  "agent_data": {
    "intent": "<用户原始意图描述>",
    "agent_params": {
      "max_iteration": 5,
      "visualize": true
    }
  },
  "errors": [],
  "next_action": "plan"
}
action 说明：
- train: 用户要求训练模型但未要求自动调优
- auto_iter: 用户要求自动迭代优化
- inference: 用户只要求使用已有模型进行预测
如果没有明确说明则默认为 "train"。
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    response = llm.invoke(messages)
    intent_json = json.loads(response.content)

    # 构建统一数据格式
    message = {
        "status": "success",
        "agent": "intent",
        "task_id": intent_json["task_id"],
        "agent_data": {
            "intent": intent_json["agent_data"]["intent"],
            "agent_params": intent_json["agent_data"]["agent_params"],
            "agent_state": {"iteration": 0},   # 初始化迭代计数
            "history": []
        },
        "errors": intent_json.get("errors", []),
        "next_action": intent_json["next_action"]
    }
    state["message_json"] = message
    state["action"] = intent_json["action"]  # 供路由函数快速访问
    state["agent_params"] = message["agent_data"]["agent_params"]
    state["agent_state"] = message["agent_data"]["agent_state"]
    state["history"] = []
    state["errors"] = []
    state["next_action"] = message["next_action"]

    return state

def plan_agent(state: AgentState) -> AgentState:
    """
    计划 Agent：结合 RAG 检索和意图，生成模型初始化参数计划
    输出 plan_json（标准格式）
    """
    llm = get_json_llm()
    intent_info = state["agent_data"]["intent"]
    agent_params = state["agent_params"]
    # 检索历史经验
    rag_results = rag_search_skill(intent_info)

    system_prompt = f"""你是一个时间序列模型配置专家。根据用户意图和检索到的历史经验，输出模型初始化参数 JSON。
参考经验：{rag_results}
要求：必须输出合法 JSON，且必须包含 seq_len, pred_len 和主要模型超参数。
返回格式：
{{
  "plan": {{
    "model_name": "PatchTST",
    "model_initial_params": {{
      "seq_len": 96,
      "pred_len": 24,
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 50
    }}
  }},
  "rag_experience_used": true,
  "next_action": "work"
}}
"""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=intent_info)]
    response = llm.invoke(messages)
    plan_data = json.loads(response.content)

    # 校验参数
    if not parameter_validation_skill(plan_data["plan"]["model_initial_params"]):
        state["errors"].append("plan 参数校验失败")
        state["next_action"] = "end"
        return state

    # 更新 state
    state["plan_json"] = plan_data
    # 同步到统一数据格式中
    state["message_json"]["agent_data"]["plan"] = plan_data["plan"]
    state["message_json"]["agent_data"]["agent_params"] = agent_params  # 保留控制参数
    state["next_action"] = "work"
    return state

def work_node(state: AgentState) -> AgentState:
    """
    工作节点（非 LLM）：根据 plan_json 或 eval_json 调用后端服务，
    完成训练/推理，收集日志，可选可视化，生成 work_json
    """
    plan = state["plan_json"]["plan"] if "plan_json" in state else {}
    eval_json = state.get("eval_json")
    visualize = state["agent_params"].get("visualize", False)
    action = state["action"]

    # 确定使用的参数
    if eval_json:
        # 自动迭代：使用 eval 建议的新参数
        params = eval_json.get("suggested_params", {})
    else:
        params = plan.get("model_initial_params", {})

    # 调用相应 skill
    if action in ["train", "auto_iter"]:
        # 训练
        result = train_api_skill(params)
        # 解析日志
        log_metrics = log_parsing_skill(result["log_path"])
        work_data = {
            "model_initial_params": params,
            "result": {
                "log_path": result["log_path"],
                "metrics": log_metrics
            }
        }
    elif action == "inference":
        # 推理：需要先获得模型路径（通过 plan 传递或检查点查找）
        model_path = plan.get("model_checkpoint_path") or model_checkpoint_skill(plan.get("model_name", "default"))
        result = inference_api_skill(model_path, data_path="data/test.csv")
        work_data = {
            "model_initial_params": params,
            "result": {
                "predictions_path": result.get("predictions_path", ""),
                "plot_path": result.get("plot_path", "")
            }
        }
    else:
        state["errors"].append("未知 action")
        return state

    # 可视化（如果需要）
    if visualize:
        plot_path = plot_skill(work_data["result"], "/tmp/plot.png")
        work_data["result"]["plot_path"] = plot_path

    state["work_json"] = work_data
    # 更新统一数据格式
    state["message_json"]["agent_data"]["work"] = {
        "model_initial_params": work_data["model_initial_params"],
        "result_summary": work_data["result"]
    }
    # 如果是初次训练（非自动迭代），更新 history
    if state["agent_state"]["iteration"] == 0:
        snapshot = {
            "iteration": 0,
            "plan_snapshot": state["plan_json"]["plan"]["model_initial_params"] if state.get("plan_json") else {},
            "work_metrics": work_data["result"].get("metrics", {}),
            "eval_advice": None
        }
        state["history"].append(snapshot)

    if action != "auto_iter":
        state["next_action"] = "end"  # 普通训练或推理直接结束
    else:
        state["next_action"] = "eval"
        # 自动迭代时也记录本次迭代快照
        snapshot = {
            "iteration": state["agent_state"]["iteration"],
            "plan_snapshot": params,
            "work_metrics": work_data["result"].get("metrics", {}),
            "eval_advice": None  # 等 eval 填充
        }
        state["history"].append(snapshot)

    return state

def eval_agent(state: AgentState) -> AgentState:
    """
    评估 Agent：使用 LLM 分析 work 结果，生成优化建议，控制迭代深度
    更新 eval_json 和 history 中的 eval_advice
    """
    llm = get_json_llm()
    work_summary = state["message_json"]["agent_data"]["work"]
    plan_summary = state["message_json"]["agent_data"].get("plan", {})
    iteration = state["agent_state"]["iteration"]
    max_iter = state["agent_params"]["max_iteration"]

    # 可选：计算额外指标（Skill）
    # metrics = metrics_skill(y_true, y_pred)

    system_prompt = f"""你是一个时间序列模型评估专家。根据当前训练结果和原始计划，输出优化建议 JSON。
结果摘要：{work_summary}
计划参数：{plan_summary}
当前迭代轮次：{iteration} / 最大允许 {max_iter}
返回格式：
{{
  "eval": {{
    "score": 0.85,
    "main_issue": "模型存在轻微过拟合",
    "suggestion": "增加 dropout 至 0.2",
    "should_continue": true,
    "suggested_params": {{   // 修改后的参数
      "seq_len": 96,
      "pred_len": 24,
      "learning_rate": 0.0005,
      "dropout": 0.2
    }}
  }},
  "next_action": "plan"  // 继续则去 plan，否则 summary
}}
"""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content="请评估")]
    response = llm.invoke(messages)
    eval_data = json.loads(response.content)

    state["eval_json"] = eval_data["eval"]
    # 更新统一数据格式
    state["message_json"]["agent_data"]["eval"] = eval_data["eval"]

    # 更新最近一次 history 的 eval_advice
    if state["history"]:
        state["history"][-1]["eval_advice"] = {
            "score": eval_data["eval"]["score"],
            "main_issue": eval_data["eval"]["main_issue"],
            "suggestion": eval_data["eval"]["suggestion"]
        }

    # 决定下一步
    if eval_data["eval"]["should_continue"] and iteration < max_iter:
        state["next_action"] = "plan"
        state["agent_state"]["iteration"] = iteration + 1
    else:
        state["next_action"] = "summary"

    return state

def summary_agent(state: AgentState) -> AgentState:
    """
    总结 Agent：基于 history 生成训练经验总结，存入向量数据库
    输出 summary_json 并写入 RAG
    """
    llm = get_json_llm()
    history = state["history"]  # 结构化摘要列表

    system_prompt = f"""你是一个经验总结专家。根据以下迭代历史，提炼出本次训练的关键经验和最佳配置。
历史记录：{history}
返回 JSON：
{{
  "summary": {{
    "best_iteration": 2,
    "best_params": {{...}},
    "overall_experience": "对于该类周期性数据，PatchTST 在 seq_len=96 时表现最佳。",
    "final_advice": "建议未来类似任务直接采用该配置。"
  }},
  "next_action": "end"
}}
"""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content="请总结")]
    response = llm.invoke(messages)
    sum_data = json.loads(response.content)

    state["summary_json"] = sum_data["summary"]

    # 将经验写入向量数据库（RAG）
    experience_text = sum_data["summary"]["overall_experience"]
    vector_db_write_skill("time_series_experience", {"text": experience_text, "metadata": {"task_id": state["task_id"]}})

    state["message_json"]["agent_data"]["summary"] = sum_data["summary"]
    state["next_action"] = "end"
    return state

# =============================================================================
# 5. 路由函数
# =============================================================================

def route_after_intent(state: AgentState) -> str:
    action = state["action"]
    if action in ["train", "auto_iter"]:
        return "plan"
    elif action == "inference":
        return "work"
    else:
        # 默认走 plan
        return "plan"

def route_after_work(state: AgentState) -> str:
    if state["action"] == "auto_iter":
        return "eval"
    else:
        return END  # 直接结束，或改为 "summary" 记录经验

def should_continue(state: AgentState) -> str:
    if state["next_action"] == "plan":
        return "plan"
    else:
        return "summary"

# =============================================================================
# 6. 构建工作流图
# =============================================================================

def build_workflow():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("intent", intent_agent)
    workflow.add_node("plan", plan_agent)
    workflow.add_node("work", work_node)
    workflow.add_node("eval", eval_agent)
    workflow.add_node("summary", summary_agent)

    # 入口
    workflow.set_entry_point("intent")

    # 条件边
    workflow.add_conditional_edges("intent", route_after_intent, {
        "plan": "plan",
        "work": "work"
    })

    workflow.add_edge("plan", "work")

    workflow.add_conditional_edges("work", route_after_work, {
        "eval": "eval",
        END: END
    })

    workflow.add_conditional_edges("eval", should_continue, {
        "plan": "plan",
        "summary": "summary"
    })

    workflow.add_edge("summary", END)

    return workflow.compile()

# =============================================================================
# 7. 示例运行
# =============================================================================

async def main():
    app = build_workflow()

    # 模拟用户输入
    test_state = {
        "intent": "请帮我训练一个预测未来24小时温度的时间序列模型，并自动优化",
        "message_json": {},  # 空字典，由 intent_agent 填充
        "history": [],
        "errors": [],
        "agent_state": {},
        "agent_params": {}
    }

    final_state = await app.ainvoke(test_state)
    print("最终状态摘要：")
    print("任务ID:", final_state.get("task_id"))
    print("迭代次数:", final_state["agent_state"].get("iteration", 0))
    print("总结:", final_state.get("summary_json", {}))
    print("完整历史摘要:")
    for item in final_state.get("history", []):
        print(item)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())