"""
Log Parser Skill — 日志解析

职责：
- 解析训练日志文件（.log / .csv）
- 提取 loss 曲线、评估指标、运行时间等

输入：log_path (str)
输出：{"train_loss": [...], "val_loss": [...], "mse": float, "mae": float, ...}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LogParserSkill:
    """
    日志解析技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责解析模型训练/推理产生的日志文件。
    1. 从 .log 文件中正则提取每轮的 train_loss / val_loss
    2. 从 results CSV 中提取最终指标（MSE, MAE, RMSE, MAPE）
    3. 计算训练总耗时

    日志格式示例：
    Epoch 1/10 - train_loss: 0.5234 - val_loss: 0.6123 - time: 12.3s
    Epoch 2/10 - train_loss: 0.3456 - val_loss: 0.4567 - time: 11.8s
    ---
    """

    PROMPT = """解析模型训练/推理产生的日志文件：
1. 从 .log 文件中正则提取每轮的 train_loss / val_loss
2. 从 results CSV 中提取最终指标（MSE, MAE, RMSE, MAPE）
3. 计算训练总耗时"""

    def parse(self, log_path: str) -> dict[str, Any]:
        """
        解析日志文件

        Args:
            log_path: 日志文件路径（.log 或 .csv）

        Returns:
            {
                "train_loss": list[float],
                "val_loss": list[float],
                "mse": float,
                "mae": float,
                "rmse": float,
                "mape": float,
                "total_time": float,       # 秒
                "best_epoch": int,
                "raw_summary": str,
            }
        """
        logger.info(f"[LogParserSkill] 解析日志: {log_path}")
        # TODO: 用正则提取日志中的指标
        return {
            "train_loss": [],
            "val_loss": [],
            "mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "total_time": 0.0,
            "best_epoch": 0,
            "raw_summary": "",
        }
