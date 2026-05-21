"""
Log Parser Skill — 日志解析

职责：
- 解析训练日志文件（.log / .csv / result.txt）
- 提取 loss 曲线、评估指标、运行时间等

输入：log_path (str) — 目录路径（results 目录）或文件路径
输出：{"train_loss": [...], "val_loss": [...], "mse": float, "mae": float, ...}
"""

import os
import re
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
    2. 从 result.txt 中提取最终指标（MSE, MAE, RMSE, MAPE）
    3. 计算训练总耗时

    日志格式示例：
    Epoch 1/10 - train_loss: 0.5234 - val_loss: 0.6123 - time: 12.3s
    Epoch 2/10 - train_loss: 0.3456 - val_loss: 0.4567 - time: 11.8s
    ---
    """

    PROMPT = """解析模型训练/推理产生的日志文件：
1. 从 .log 文件中正则提取每轮的 train_loss / val_loss
2. 从 result.txt 中提取最终指标（MSE, MAE, RMSE, MAPE）
3. 计算训练总耗时"""

    # 正则：Epoch: N cost time: X.XXs
    _RE_EPOCH_TIME = re.compile(r"Epoch:\s*(\d+)\s*cost time:\s*([\d.]+)")

    # 正则：Epoch: N, Steps: M | Train Loss: X.XXXX Vali Loss: X.XXXX Test Loss: X.XXXX
    _RE_EPOCH_LOSS = re.compile(
        r"Epoch:\s*(\d+).*?Train Loss:\s*([\d.eE+-]+)\s+Vali Loss:\s*([\d.eE+-]+)\s+Test Loss:\s*([\d.eE+-]+)"
    )

    # 正则：result.txt 中的指标行：mse:0.123, mae:0.456, rse:0.789
    _RE_METRICS_LINE = re.compile(r"(mse|mae|rmse|mape|rse|corr)\s*:\s*([\d.eE+-]+)")

    def parse(self, log_path: str) -> dict[str, Any]:
        """
        解析日志目录或文件

        Args:
            log_path: 结果目录路径（包含 pred.npy）或 result.txt 文件路径

        Returns:
            {
                "train_loss": list[float],
                "val_loss": list[float],
                "test_loss": list[float],
                "mse": float,
                "mae": float,
                "rmse": float,
                "mape": float,
                "rse": float,
                "total_time": float,
                "best_epoch": int,
                "raw_summary": str,
            }
        """
        logger.info(f"[LogParserSkill] 解析日志: {log_path}")

        result: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "rse": 0.0,
            "total_time": 0.0,
            "best_epoch": 0,
            "raw_summary": "",
        }

        # 确定实际文件路径
        log_file = self._resolve_log_path(log_path)
        if not log_file or not os.path.isfile(log_file):
            logger.warning(f"[LogParserSkill] 日志文件不存在: {log_path}")
            return result

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"[LogParserSkill] 读取日志失败: {e}")
            return result

        result["raw_summary"] = content[:2000]  # 截取前 2000 字符

        # 1. 解析每轮的 loss
        for match in self._RE_EPOCH_LOSS.finditer(content):
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            test_loss = float(match.group(4))
            result["train_loss"].append(train_loss)
            result["val_loss"].append(val_loss)
            result["test_loss"].append(test_loss)
            result["best_epoch"] = epoch

        # 2. 解析总耗时
        total_time = 0.0
        for match in self._RE_EPOCH_TIME.finditer(content):
            total_time += float(match.group(2))
        result["total_time"] = total_time

        # 3. 解析最终指标（从 result.txt 或日志末尾的指标行）
        metrics_found: dict[str, float] = {}
        for match in self._RE_METRICS_LINE.finditer(content):
            key = match.group(1)
            val = float(match.group(2))
            # 取最后一次出现（最终值）
            metrics_found[key] = val

        if "mse" in metrics_found:
            result["mse"] = metrics_found["mse"]
        if "mae" in metrics_found:
            result["mae"] = metrics_found["mae"]
        if "rse" in metrics_found:
            result["rse"] = metrics_found["rse"]
        if "rmse" in metrics_found:
            result["rmse"] = metrics_found["rmse"]
        if "mape" in metrics_found:
            result["mape"] = metrics_found["mape"]

        logger.info(
            f"[LogParserSkill] 解析完成: "
            f"epochs={len(result['train_loss'])}, "
            f"mse={result['mse']:.4f}, "
            f"mae={result['mae']:.4f}, "
            f"total_time={result['total_time']:.1f}s"
        )
        return result

    def parse_result_txt(self, result_txt_path: str) -> dict[str, float]:
        """
        专门解析 result.txt 中的指标行

        Args:
            result_txt_path: result.txt 文件路径

        Returns:
            {"mse": float, "mae": float, "rse": float, ...}
        """
        metrics: dict[str, float] = {}
        if not os.path.isfile(result_txt_path):
            return metrics

        try:
            with open(result_txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            for match in self._RE_METRICS_LINE.finditer(content):
                metrics[match.group(1)] = float(match.group(2))
        except Exception as e:
            logger.warning(f"[LogParserSkill] 解析 result.txt 失败: {e}")

        return metrics

    # -----------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------

    @staticmethod
    def _resolve_log_path(log_path: str) -> str | None:
        """
        智能定位日志文件：
        - 如果是目录，依次尝试 result.txt / training.log / stdout.log
        - 如果是文件，直接使用
        """
        if os.path.isfile(log_path):
            return log_path

        if os.path.isdir(log_path):
            for candidate in ("result.txt", "training.log", "stdout.log", "log.txt"):
                full = os.path.join(log_path, candidate)
                if os.path.isfile(full):
                    return full
            # 仍没找到，尝试找任何 .txt / .log 文件
            for fname in os.listdir(log_path):
                if fname.endswith(".txt") or fname.endswith(".log"):
                    return os.path.join(log_path, fname)

        return None
