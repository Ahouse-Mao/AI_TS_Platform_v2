"""
Checkpoint Skill — 模型检查点查找

职责：
- 在本地 checkpoints 目录中查找最优/最新的模型权重文件
- 支持按 model_name, dataset, pred_len 等条件过滤

输入：model_name, dataset, pred_len（可选）
输出：checkpoint_path (str), metadata (dict)
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CheckpointSkill:
    """
    模型检查点查找技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责在本地 checkpoints 目录中搜索模型的权重文件。
    搜索条件：模型名称（DLinear/PatchTST/...）、数据集（ETTh1/...）、预测长度。
    返回最佳匹配的 .pth 检查点路径及其元数据。
    优先级：metrics 最优 > 训练时间最新。
    ---
    """

    PROMPT = """在本地 checkpoints 目录中搜索模型权重文件。
搜索条件：模型名称、数据集、预测长度。
返回最佳匹配的 .pth 检查点路径及其元数据。
优先级：metrics 最优 > 训练时间最新。"""

    def __init__(self, checkpoints_root: str = "backend/model_src/checkpoints"):
        self.checkpoints_root = Path(checkpoints_root)

    def find_best(
        self,
        model_name: str,
        dataset: str,
        pred_len: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        查找最优检查点

        Args:
            model_name: 模型名称（如 "DLinear", "PatchTST"）
            dataset: 数据集名称（如 "ETTh1", "ETTm1"）
            pred_len: 预测长度（可选）

        Returns:
            {
                "checkpoint_path": str,
                "model_name": str,
                "dataset": str,
                "pred_len": int,
                "metrics": dict,
                "timestamp": str,
            }
        """
        logger.info(f"[CheckpointSkill] 查找: model={model_name}, dataset={dataset}, pred_len={pred_len}")
        # TODO: 遍历 checkpoints_root 目录，匹配条件
        # 1. glob 匹配目录名包含 model_name + dataset + pred_len
        # 2. 读取 result.txt 或 checkpoint 元数据
        # 3. 按指标排序返回最优
        return {
            "checkpoint_path": "",
            "model_name": model_name,
            "dataset": dataset,
            "pred_len": pred_len or 96,
            "metrics": {},
            "timestamp": "",
        }

    def find_latest(
        self,
        model_name: str,
        dataset: str,
    ) -> dict[str, Any]:
        """查找最新检查点（按时间排序）"""
        logger.info(f"[CheckpointSkill] 查找最新: model={model_name}, dataset={dataset}")
        return self.find_best(model_name, dataset)
