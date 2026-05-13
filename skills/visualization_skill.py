"""
Visualization Skill — 可视化绘图

职责：
- 绘制预测 vs 真实值对比图
- 绘制 loss 曲线
- 绘制多轮迭代指标对比图

输入：data (dict), chart_type (str)
输出：image_path (str)
"""

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)


class VisualizationSkill:
    """
    可视化绘图技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责生成训练/推理结果的可视化图表：
    1. 预测 vs 真实值对比图（折线图）
    2. 训练 loss 曲线（train_loss + val_loss 双线图）
    3. 多轮迭代指标对比图（柱状图 / 趋势图）

    使用 matplotlib 绘制，保存为 PNG 到指定目录。
    ---
    """

    PROMPT = """生成训练/推理结果的可视化图表：
1. 预测 vs 真实值对比图（折线图）
2. 训练 loss 曲线（train_loss + val_loss 双线图）
3. 多轮迭代指标对比图（柱状图/趋势图）

使用 matplotlib 绘制，保存为 PNG。"""

    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = output_dir

    def plot_predictions(
        self,
        ground_truth: list[float],
        predictions: list[float],
        title: str = "Prediction vs Ground Truth",
    ) -> str:
        """
        绘制预测 vs 真实值对比图

        Args:
            ground_truth: 真实值序列
            predictions: 预测值序列
            title: 图表标题

        Returns:
            image_path: 保存的图片路径
        """
        logger.info(f"[VisualizationSkill] 绘制预测对比图: {title}")
        # TODO: matplotlib 绘制
        return f"{self.output_dir}/pred_vs_gt.png"

    def plot_loss_curve(
        self,
        train_loss: list[float],
        val_loss: list[float],
        title: str = "Training Loss Curve",
    ) -> str:
        """
        绘制 loss 曲线

        Args:
            train_loss: 训练 loss 序列
            val_loss: 验证 loss 序列
            title: 图表标题

        Returns:
            image_path: 保存的图片路径
        """
        logger.info(f"[VisualizationSkill] 绘制 loss 曲线: {title}")
        return f"{self.output_dir}/loss_curve.png"

    def plot_iteration_comparison(
        self,
        history: list[dict[str, Any]],
        metric_name: str = "mse",
        title: str = "Iteration Comparison",
    ) -> str:
        """
        绘制多轮迭代指标对比图

        Args:
            history: 历史迭代记录列表
            metric_name: 要对比的指标名
            title: 图表标题

        Returns:
            image_path: 保存的图片路径
        """
        logger.info(f"[VisualizationSkill] 绘制迭代对比图: {title}")
        return f"{self.output_dir}/iteration_comparison.png"
