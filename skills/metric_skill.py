"""
Metric Skill — 指标计算

职责：
- 计算时序预测常用指标：MSE、MAE、RMSE、MAPE
- 对比多组预测结果

输入：ground_truth (list), predictions (list)
输出：{"mse": float, "mae": float, "rmse": float, "mape": float}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MetricSkill:
    """
    指标计算技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责计算时序预测的常用评估指标：
    1. MSE  (Mean Squared Error)         = mean((y_true - y_pred)^2)
    2. MAE  (Mean Absolute Error)        = mean(|y_true - y_pred|)
    3. RMSE (Root Mean Squared Error)    = sqrt(MSE)
    4. MAPE (Mean Absolute Percentage Error) = mean(|(y_true - y_pred) / y_true|) * 100

    输入为真实值和预测值的等长列表。
    ---
    """

    PROMPT = """计算时序预测常用评估指标：
1. MSE  = mean((y_true - y_pred)^2)
2. MAE  = mean(|y_true - y_pred|)
3. RMSE = sqrt(MSE)
4. MAPE = mean(|(y_true - y_pred) / y_true|) * 100

输入为真实值和预测值的等长列表。"""

    def compute(
        self,
        ground_truth: list[float],
        predictions: list[float],
    ) -> dict[str, float]:
        """
        计算所有指标

        Args:
            ground_truth: 真实值列表
            predictions: 预测值列表

        Returns:
            {"mse": float, "mae": float, "rmse": float, "mape": float}
        """
        logger.info(f"[MetricSkill] 计算指标, 数据点数: {len(ground_truth)}")
        # TODO: numpy 计算
        import numpy as np

        y_true = np.array(ground_truth, dtype=np.float64)
        y_pred = np.array(predictions, dtype=np.float64)

        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))

        # MAPE 处理分母为 0 的情况
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float("inf")

        return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}
