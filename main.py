"""
AI-TS-Platform 主入口
基于 LangGraph 构建时序预测智能体集群的流程编排
"""

import uuid
import logging
from typing import TypedDict, Literal, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.plan_agent import PlanAgent
from agents.work_agent import WorkAgent
from agents.eval_agent import EvalAgent
from agents.summary_agent import SummaryAgent

logger = logging.getLogger(__name__)


# ============================================================
# 统一状态定义（与 README 约定的 JSON 格式对齐）
# ============================================================

class AgentState(TypedDict, total=False):
    """LangGraph 全局状态，贯穿所有 Agent 节点"""

    status: str                       # "success" | "error"
    agent: str                        # 当前所在 agent: "plan" | "work" | "eval" | "summary"
    task_id: str                      # 任务唯一 ID
    agent_data: dict[str, Any]        # 各 agent 传递数据的核心容器
    errors: list[str]                 # 错误信息列表
    next_action: str                  # 指导下一个 agent: "work" | "eval" | "summary" | "end"


def _init_agent_data() -> dict[str, Any]:
    """初始化 agent_data 结构"""
    return {
        "intent": "",                  # plan 解析出的用户意图
        "plan": {},                    # plan agent 输出的初始化参数
        "work": {},                    # work agent 输出的训练/推理结果
        "eval": {},                    # eval agent 输出的评估建议
        "summary": {},                 # summary agent 输出的经验总结
        "agent_params": {
            "max_iteration": 1,        # 最大迭代轮数（1 = 不自动迭代）
            "visualize": False,        # 是否可视化
        },
        "agent_state": {
            "iteration": 0,            # 当前迭代轮数
        },
        "history": [],                 # 历史迭代快照
    }


# ============================================================
# Agent 节点函数
# ============================================================

def plan_node(state: AgentState) -> AgentState:
    """
    Plan Agent 节点：
    分析用户意图 → 结合 RAG 检索 → 给出初始化模型参数
    """
    logger.info("[Plan] 开始分析用户意图...")
    agent = PlanAgent()
    try:
        result = agent.run(state)
        state["status"] = "success"
        state["agent"] = "plan"
        state["agent_data"]["plan"] = result.get("plan", {})
        state["agent_data"]["intent"] = result.get("intent", "")
        state["agent_data"]["agent_params"] = result.get("agent_params", state["agent_data"]["agent_params"])
        state["next_action"] = result.get("next_action", "work")
    except Exception as e:
        logger.error(f"[Plan] 执行失败: {e}")
        state["status"] = "error"
        state["errors"].append(str(e))
        state["next_action"] = "end"
    return state


def work_node(state: AgentState) -> AgentState:
    """
    Work Agent 节点：
    接收 plan 或 eval 的参数 → 调用后端 API → 启动训练/推理
    """
    logger.info("[Work] 开始执行训练/推理任务...")
    agent = WorkAgent()
    try:
        result = agent.run(state)
        state["status"] = "success"
        state["agent"] = "work"
        state["agent_data"]["work"] = result.get("work", {})
        state["next_action"] = result.get("next_action", "end")
    except Exception as e:
        logger.error(f"[Work] 执行失败: {e}")
        state["status"] = "error"
        state["errors"].append(str(e))
        state["next_action"] = "end"
    return state


def eval_node(state: AgentState) -> AgentState:
    """
    Eval Agent 节点：
    分析 work 结果 → 计算指标 → 给出优化建议 → 控制迭代深度
    """
    logger.info("[Eval] 开始评估训练结果...")
    agent = EvalAgent()
    try:
        result = agent.run(state)
        state["status"] = "success"
        state["agent"] = "eval"
        state["agent_data"]["eval"] = result.get("eval", {})

        # 记录迭代快照
        iteration = state["agent_data"]["agent_state"]["iteration"]
        state["agent_data"]["history"].append({
            "iteration": iteration,
            "eval_summary": result.get("eval", {}).get("summary", ""),
            "metrics": result.get("eval", {}).get("metrics", {}),
        })

        state["next_action"] = result.get("next_action", "summary")
    except Exception as e:
        logger.error(f"[Eval] 执行失败: {e}")
        state["status"] = "error"
        state["errors"].append(str(e))
        state["next_action"] = "summary"
    return state


def summary_node(state: AgentState) -> AgentState:
    """
    Summary Agent 节点：
    总结训练经验 → 存入 RAG 知识库
    """
    logger.info("[Summary] 开始总结训练经验...")
    agent = SummaryAgent()
    try:
        result = agent.run(state)
        state["status"] = "success"
        state["agent"] = "summary"
        state["agent_data"]["summary"] = result.get("summary", {})
        state["next_action"] = "end"
    except Exception as e:
        logger.error(f"[Summary] 执行失败: {e}")
        state["status"] = "error"
        state["errors"].append(str(e))
        state["next_action"] = "end"
    return state


# ============================================================
# 路由函数 — 控制 Agent 间的流转
# ============================================================

def router_after_plan(state: AgentState) -> Literal["work", "end"]:
    """plan 之后的路由：判断是否需要继续到 work"""
    if state.get("status") == "error":
        return "end"
    next_action = state.get("next_action", "end")
    if next_action == "work":
        return "work"
    return "end"


def router_after_work(state: AgentState) -> Literal["eval", "summary", "end"]:
    """
    work 之后的路由：
    - 需要自动迭代 → 进入 eval
    - 需要总结     → 进入 summary
    - 普通任务     → 结束
    """
    if state.get("status") == "error":
        return "end"

    max_iter = state["agent_data"]["agent_params"].get("max_iteration", 1)
    iteration = state["agent_data"]["agent_state"].get("iteration", 0)

    # 首次 work 后 iteration=1 表示已完成第 1 轮
    state["agent_data"]["agent_state"]["iteration"] = iteration + 1

    next_action = state.get("next_action", "end")
    if next_action == "eval":
        return "eval"
    elif next_action == "summary":
        return "summary"
    else:
        return "end"


def router_after_eval(state: AgentState) -> Literal["work", "summary"]:
    """
    eval 之后的路由：
    - 未达到最大迭代 → 回到 work 继续优化
    - 达到最大迭代   → 进入 summary 总结
    """
    if state.get("status") == "error":
        return "summary"

    iteration = state["agent_data"]["agent_state"].get("iteration", 0)
    max_iter = state["agent_data"]["agent_params"].get("max_iteration", 1)

    next_action = state.get("next_action", "summary")

    if next_action == "work" and iteration < max_iter:
        return "work"
    return "summary"


# ============================================================
# 构建 LangGraph 图
# ============================================================

def build_graph() -> StateGraph:
    """
    构建 Agent 工作流图：

    START
      │
      ▼
    plan ──(router)──► work ──(router)──► eval ──(router)──► work (loop)
      │                   │                  │
      │                   │                  └──► summary ──► END
      │                   └──► summary / END
      └──► END
    """
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("plan", plan_node)
    graph.add_node("work", work_node)
    graph.add_node("eval", eval_node)
    graph.add_node("summary", summary_node)

    # 添加边
    graph.add_edge(START, "plan")

    # plan → router → work / end
    graph.add_conditional_edges("plan", router_after_plan, {
        "work": "work",
        "end": END,
    })

    # work → router → eval / summary / end
    graph.add_conditional_edges("work", router_after_work, {
        "eval": "eval",
        "summary": "summary",
        "end": END,
    })

    # eval → router → work (loop) / summary
    graph.add_conditional_edges("eval", router_after_eval, {
        "work": "work",
        "summary": "summary",
    })

    # summary → END
    graph.add_edge("summary", END)

    return graph


# ============================================================
# 运行时入口
# ============================================================

class TSPlatform:
    """
    时序预测平台主类
    封装 LangGraph 工作流，提供简洁的调用接口
    """

    def __init__(self):
        self.graph = build_graph()
        self.checkpointer = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    def run(self, user_input: str, **kwargs) -> AgentState:
        """
        执行一次完整的 Agent 工作流

        Args:
            user_input: 用户的自然语言输入
            **kwargs: 额外参数（如 max_iteration, visualize 等）

        Returns:
            最终的 AgentState
        """
        task_id = str(uuid.uuid4())[:8]
        initial_state: AgentState = {
            "status": "running",
            "agent": "",
            "task_id": task_id,
            "agent_data": _init_agent_data(),
            "errors": [],
            "next_action": "plan",
        }

        # 将用户输入和额外参数注入初始状态
        initial_state["agent_data"]["intent"] = user_input
        initial_state["agent_data"]["agent_params"].update({
            "max_iteration": kwargs.get("max_iteration", 1),
            "visualize": kwargs.get("visualize", False),
        })

        logger.info(f"启动任务 {task_id}，用户输入: {user_input}")
        config = {"configurable": {"thread_id": task_id}}

        final_state = self.app.invoke(initial_state, config)
        return final_state

    async def run_async(self, user_input: str, **kwargs) -> AgentState:
        """异步执行"""
        task_id = str(uuid.uuid4())[:8]
        initial_state: AgentState = {
            "status": "running",
            "agent": "",
            "task_id": task_id,
            "agent_data": _init_agent_data(),
            "errors": [],
            "next_action": "plan",
        }
        initial_state["agent_data"]["intent"] = user_input
        initial_state["agent_data"]["agent_params"].update({
            "max_iteration": kwargs.get("max_iteration", 1),
            "visualize": kwargs.get("visualize", False),
        })

        logger.info(f"启动任务 {task_id}，用户输入: {user_input}")
        config = {"configurable": {"thread_id": task_id}}

        final_state = await self.app.ainvoke(initial_state, config)
        return final_state


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    platform = TSPlatform()

    # 示例 1：普通训练任务（不自动迭代）
    print("\n" + "=" * 60)
    print("示例 1：普通训练任务")
    print("=" * 60)
    result = platform.run("用 DLinear 模型在 ETTh1 数据集上训练，预测长度 96")
    print(f"状态: {result['status']}, 最终 agent: {result['agent']}")
    print(f"错误: {result['errors']}")

    # 示例 2：自动迭代训练任务
    print("\n" + "=" * 60)
    print("示例 2：自动迭代优化任务")
    print("=" * 60)
    result = platform.run(
        "用 PatchTST 在 ETTh1 上训练并自动调优",
        max_iteration=3,
        visualize=True,
    )
    print(f"状态: {result['status']}, 最终 agent: {result['agent']}")
    print(f"迭代轮数: {result['agent_data']['agent_state']['iteration']}")
    print(f"历史快照数: {len(result['agent_data']['history'])}")

    # 示例 3：推理任务
    print("\n" + "=" * 60)
    print("示例 3：推理任务")
    print("=" * 60)
    result = platform.run("加载最近的 ETTh1 模型进行推理")
    print(f"状态: {result['status']}, 最终 agent: {result['agent']}")
