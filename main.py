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
# 一、统一状态定义（与 README 约定的 JSON 格式对齐）
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
        "intent": "",
        "plan": {},
        "work": {},
        "eval": {},
        "summary": {},
        "agent_params": {
            "max_iteration": 1,
            "visualize": False,
        },
        "agent_state": {
            "iteration": 0,
        },
        "history": [],
    }


# ============================================================
# 二、各 Agent 节点函数
# ============================================================

def plan_node(state: AgentState) -> AgentState:
    """Plan Agent 节点 — state 修改下沉到 agent 内部"""
    logger.info("[Plan] 开始分析用户意图...")
    try:
        state = PlanAgent().run(state)
    except Exception as e:
        logger.error(f"[Plan] 执行失败: {e}")
        state["status"] = "error"
        state["errors"].append(str(e))
        state["next_action"] = "end"
    return state


def work_node(state: AgentState) -> AgentState:
    """Work Agent 节点 — state 修改下沉到 agent 内部"""
    logger.info("[Work] 开始执行训练/推理任务...")
    try:
        state = WorkAgent().run(state)
    except Exception as e:
        logger.error(f"[Work] 执行失败: {e}")
        state["status"] = "error"
        state["errors"].append(str(e))
        state["next_action"] = "end"
    return state


def eval_node(state: AgentState) -> AgentState:
    """Eval Agent 节点 — state 修改下沉到 agent 内部"""
    logger.info("[Eval] 开始评估训练结果...")
    try:
        state = EvalAgent().run(state)
    except Exception as e:
        logger.error(f"[Eval] 执行失败: {e}")
        state["status"] = "error"
        state["errors"].append(str(e))
        state["next_action"] = "summary"
    return state


def summary_node(state: AgentState) -> AgentState:
    """Summary Agent 节点 — state 修改下沉到 agent 内部"""
    logger.info("[Summary] 开始总结训练经验...")
    try:
        state = SummaryAgent().run(state)
    except Exception as e:
        logger.error(f"[Summary] 执行失败: {e}")
        state["status"] = "error"
        state["errors"].append(str(e))
        state["next_action"] = "end"
    return state


# ============================================================
# 三、统一路由 —— 根据 next_action + 迭代状态决定下一个节点
# ============================================================

# 图构建过程
def router(state: AgentState) -> Literal["plan", "work", "eval", "summary", "end"]:
    """
    唯一路由函数：读取 state.next_action 和迭代状态来决定下一步

    路由逻辑（文字描述等价于下方的 if-else 链）：
      id: plan
        后接 -> work

      id: work
        后接 -> eval | summary | end
        根据 next_action 决定

      id: eval
        后接 -> summary（默认）
        如果 next_action == "work" 且 iteration < max_iteration → work（继续循环）
        否则 → summary

      id: summary
        后接 -> end

    """
    if state.get("status") == "error":
        return "end"

    na = state.get("next_action", "end")
    current_agent = state.get("agent", "")

    # 错误兜底
    if na == "end":
        return "end"

    # work → eval 或 summary 或 end
    if current_agent == "work":
        if na == "eval":
            return "eval"
        if na == "summary":
            return "summary"
        return "end"

    # eval → 循环回 work 或 summary
    if current_agent == "eval":
        iteration = state["agent_data"]["agent_state"]["iteration"]
        max_iter = state["agent_data"]["agent_params"]["max_iteration"]
        if na == "work" and iteration < max_iter:
            return "work"
        return "summary"

    # plan → work（默认）
    return "work"


# ============================================================
# 四、构建 LangGraph 图
# ============================================================

def build_graph() -> StateGraph:
    """
    构建 Agent 工作流图

    START → plan → router ──→ work → router ──→ eval → router ──→ work (loop)
                              │                  │
                              ├──→ summary       └──→ summary → END
                              └──→ END
    """
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("work", work_node)
    graph.add_node("eval", eval_node)
    graph.add_node("summary", summary_node)

    graph.add_edge(START, "plan")

    # 所有节点运行完后统一走 router 决定下一步，router 内部根据 next_action 和迭代状态进行判断
    graph.add_conditional_edges("plan", router, { # plan之后必定进入work或者直接结束
        "work": "work", # 三个参数分别是源节点、路由函数、路由结果与目标节点的映射关系
        "end": END,
    })
    graph.add_conditional_edges("work", router, { # work之后可能进入eval、summary或者直接结束
        "eval": "eval",
        "summary": "summary",
        "end": END,
    })
    graph.add_conditional_edges("eval", router, { # eval之后可能进入work（继续迭代）或者summary（结束迭代）
        "work": "work",
        "summary": "summary",
        "end": END,
    })
    graph.add_edge("summary", END) # summary 之后可能结束

    return graph


# ============================================================
# 五、运行时入口
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
# 六、CLI 入口
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
