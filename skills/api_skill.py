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
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# 默认后端 API 基础地址
_DEFAULT_BASE_URL = "http://localhost:8000"

# 轮询间隔（秒）
_POLL_INTERVAL = 5.0

# 最大轮询时间（秒），防止无限等待
_MAX_POLL_TIME = 7200  # 2 小时


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

    def __init__(self, base_url: str = _DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    # -----------------------------------------------------------
    # 公共方法
    # -----------------------------------------------------------

    def run_training(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        启动训练任务（同步阻塞直到完成）

        Args:
            params: 模型参数配置，包含：
                - model_name, dataset, seq_len, pred_len
                - batch_size, learning_rate, epochs 等

        Returns:
            {
                "status": "completed" | "failed",
                "task_id": str,
                "checkpoint_path": str | None,
                "log_path": str | None,
                "metrics": dict,
                "error": str | None,
            }
        """
        logger.info(f"[APISkill] 启动训练: {params.get('model_name')} on {params.get('dataset')}")

        try:
            resp = requests.post(
                f"{self.base_url}/api/train",
                json=params,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            task_id = result["task_id"]
            logger.info(f"[APISkill] 训练任务已提交: {task_id}")

            # 轮询直到完成
            return self._poll_task(task_id)

        except requests.RequestException as e:
            logger.error(f"[APISkill] 请求训练 API 失败: {e}")
            return {
                "status": "failed",
                "task_id": "",
                "checkpoint_path": None,
                "log_path": None,
                "metrics": {},
                "error": str(e),
            }

    def run_inference(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        启动推理任务（同步阻塞直到完成）

        Args:
            params: 包含 checkpoint_path, dataset, model_name, seq_len, pred_len 等

        Returns:
            {
                "status": "completed" | "failed",
                "task_id": str,
                "predictions_path": str | None,
                "log_path": str | None,
                "metrics": dict,
                "error": str | None,
            }
        """
        logger.info(f"[APISkill] 启动推理: model={params.get('model_name')}, dataset={params.get('dataset')}")

        try:
            resp = requests.post(
                f"{self.base_url}/api/infer",
                json=params,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            task_id = result["task_id"]
            logger.info(f"[APISkill] 推理任务已提交: {task_id}")

            # 轮询直到完成
            return self._poll_task(task_id)

        except requests.RequestException as e:
            logger.error(f"[APISkill] 请求推理 API 失败: {e}")
            return {
                "status": "failed",
                "task_id": "",
                "predictions_path": None,
                "log_path": None,
                "metrics": {},
                "error": str(e),
            }

    def get_status(self, task_id: str) -> dict[str, Any]:
        """查询任务状态"""
        try:
            resp = requests.get(
                f"{self.base_url}/api/status/{task_id}",
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"[APISkill] 查询状态失败: task_id={task_id}, error={e}")
            return {"status": "unknown", "progress": 0, "error": str(e)}

    # -----------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------

    def _poll_task(self, task_id: str) -> dict[str, Any]:
        """轮询任务直到完成或失败"""
        start_time = time.time()
        last_status = "pending"

        while True:
            elapsed = time.time() - start_time
            if elapsed > _MAX_POLL_TIME:
                logger.warning(f"[APISkill] 任务 {task_id} 轮询超时")
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": "轮询超时（超过 2 小时）",
                }

            status_data = self.get_status(task_id)

            cur_status = status_data.get("status", "unknown")
            if cur_status != last_status:
                logger.info(
                    f"[APISkill] 任务 {task_id} 状态变更: "
                    f"{last_status} → {cur_status} (进度: {status_data.get('progress', 0)}%)"
                )
                last_status = cur_status

            if cur_status == "completed":
                return {
                    "status": "completed",
                    "task_id": task_id,
                    "checkpoint_path": status_data.get("checkpoint_path"),
                    "log_path": status_data.get("log_path"),
                    "predictions_path": status_data.get("predictions_path"),
                    "metrics": status_data.get("metrics", {}),
                    "error": None,
                }

            if cur_status == "failed":
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": status_data.get("error", "未知错误"),
                }

            time.sleep(_POLL_INTERVAL)
