"""
API Skill — 训练/推理 API 调用

职责：
- 构造后端 API 请求（JSON 配置）
- 调用训练/推理接口
- 返回任务状态和结果路径

输入：params (dict), task_type ("train" | "inference")
输出：{"status": str, "checkpoint_path": str, "log_path": str}
"""

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)


class APISkill:
    """
    后端 API 调用技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责与后端训练/推理服务通信。
    1. 根据参数构造符合后端 API 规范的 JSON 配置
    2. POST 到训练或推理端点
    3. 轮询任务状态直到完成
    4. 返回 checkpoint 路径和日志路径

    API 端点：
    - POST /api/train  启动训练
    - POST /api/infer  启动推理
    - GET  /api/status/{task_id}  查询任务状态
    ---
    """

    PROMPT = """与后端训练/推理服务通信：
1. 构造符合后端 API 规范的 JSON 配置
2. POST 到训练或推理端点
3. 轮询任务状态直到完成
4. 返回 checkpoint 路径和日志路径

API 端点：POST /api/train, POST /api/infer, GET /api/status/{task_id}"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def run_training(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        启动训练任务

        Args:
            params: 模型参数配置，包含:
                - model_name, dataset, seq_len, pred_len
                - batch_size, learning_rate, epochs
                - 其他模型特定参数

        Returns:
            {
                "status": "completed" | "failed",
                "task_id": str,
                "checkpoint_path": str,
                "log_path": str,
            }
        """
        logger.info(f"[APISkill] 启动训练: {params.get('model_name')} on {params.get('dataset')}")
        # TODO: POST /api/train
        return {
            "status": "completed",
            "task_id": "train_001",
            "checkpoint_path": "/path/to/checkpoint.pth",
            "log_path": "/path/to/training.log",
        }

    def run_inference(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        启动推理任务

        Args:
            params: 包含 checkpoint_path, dataset, pred_len 等

        Returns:
            {
                "status": "completed" | "failed",
                "predictions": list[float],
                "log_path": str,
            }
        """
        logger.info(f"[APISkill] 启动推理: checkpoint={params.get('checkpoint_path')}")
        # TODO: POST /api/infer
        return {
            "status": "completed",
            "predictions": [],
            "log_path": "/path/to/inference.log",
        }

    def get_status(self, task_id: str) -> dict[str, Any]:
        """查询任务状态"""
        logger.info(f"[APISkill] 查询状态: {task_id}")
        return {"status": "completed", "progress": 100}
