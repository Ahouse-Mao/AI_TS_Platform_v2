"""
Checkpoint Skill — 模型检查点查找

职责：
- 在本地 checkpoints 目录中查找最优/最新的模型权重文件
- 支持按 model_name, dataset, pred_len 等条件过滤

输入：model_name, dataset, pred_len（可选）
输出：checkpoint_path (str), metadata (dict)
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 从 setting 目录名中提取字段的正则
# 格式: {data}_{seq_len}_{pred_len}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_...
_SETTING_RE = re.compile(
    r"^(?P<dataset>[^_]+)_\d+_\d+_(?P<model>[^_]+)"
    r".*_sl(?P<seq_len>\d+)"
    r"_ll\d+"
    r"_pl(?P<pred_len>\d+)"
)

# 结果目录中的 result.txt（与 checkpoints 平级）
# exp.test() 使用相对路径 "result.txt" 写入，CWD=项目根目录
_RESULT_TXT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "result.txt"
)


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

        遍历 checkpoints 目录，匹配 model_name + dataset + pred_len，
        从 result.txt 中读取指标，按 MSE 升序返回最优。

        Args:
            model_name: 模型名称（如 "DLinear", "PatchTST"）
            dataset: 数据集名称（如 "ETTh1", "ETTm1"）
            pred_len: 预测长度（可选，不传则匹配任意长度）

        Returns:
            {
                "checkpoint_path": str,
                "model_name": str,
                "dataset": str,
                "pred_len": int,
                "seq_len": int,
                "metrics": dict,
                "timestamp": str,
            }
        """
        logger.info(f"[CheckpointSkill] 查找: model={model_name}, dataset={dataset}, pred_len={pred_len}")

        candidates: list[dict[str, Any]] = []

        if not self.checkpoints_root.is_dir():
            logger.warning(f"[CheckpointSkill] 检查点目录不存在: {self.checkpoints_root}")
            return self._empty_result(model_name, dataset, pred_len)

        for entry in self.checkpoints_root.iterdir():
            if not entry.is_dir():
                continue

            match = _SETTING_RE.match(entry.name)
            if not match:
                continue

            g = match.groupdict()
            # 过滤条件
            if g["model"] != model_name:
                continue
            if g["dataset"] != dataset:
                continue
            if pred_len is not None and int(g["pred_len"]) != pred_len:
                continue

            checkpoint_file = entry / "checkpoint.pth"
            if not checkpoint_file.is_file():
                continue

            # 读取指标
            metrics = self._read_metrics(entry.name)

            # 获取修改时间
            mtime = os.path.getmtime(checkpoint_file)

            candidates.append({
                "checkpoint_path": str(checkpoint_file.resolve()),
                "model_name": model_name,
                "dataset": dataset,
                "seq_len": int(g["seq_len"]),
                "pred_len": int(g["pred_len"]),
                "metrics": metrics,
                "timestamp": str(mtime),
                "_dir_name": entry.name,
            })

        if not candidates:
            logger.warning(
                f"[CheckpointSkill] 未找到匹配的检查点: "
                f"model={model_name}, dataset={dataset}, pred_len={pred_len}"
            )
            return self._empty_result(model_name, dataset, pred_len)

        # 按 MSE 升序排序（MSE 越小越好），MSE 缺失时按时间降序（最新优先）
        def sort_key(c: dict) -> tuple:
            mse = c["metrics"].get("mse")
            if mse is not None:
                return (0, mse)
            return (1, -float(c["timestamp"]))

        candidates.sort(key=sort_key)
        best = candidates[0]

        logger.info(
            f"[CheckpointSkill] 找到最优检查点: {best['_dir_name']}, "
            f"mse={best['metrics'].get('mse', 'N/A')}"
        )
        # 清理内部字段
        best.pop("_dir_name", None)
        return best

    def find_latest(
        self,
        model_name: str,
        dataset: str,
        pred_len: Optional[int] = None,
    ) -> dict[str, Any]:
        """查找最新检查点（按文件修改时间降序）"""
        logger.info(f"[CheckpointSkill] 查找最新: model={model_name}, dataset={dataset}")

        candidates: list[dict[str, Any]] = []

        if not self.checkpoints_root.is_dir():
            return self._empty_result(model_name, dataset, pred_len)

        for entry in self.checkpoints_root.iterdir():
            if not entry.is_dir():
                continue

            match = _SETTING_RE.match(entry.name)
            if not match:
                continue

            g = match.groupdict()
            if g["model"] != model_name:
                continue
            if g["dataset"] != dataset:
                continue
            if pred_len is not None and int(g["pred_len"]) != pred_len:
                continue

            checkpoint_file = entry / "checkpoint.pth"
            if not checkpoint_file.is_file():
                continue

            metrics = self._read_metrics(entry.name)
            mtime = os.path.getmtime(checkpoint_file)

            candidates.append({
                "checkpoint_path": str(checkpoint_file.resolve()),
                "model_name": model_name,
                "dataset": dataset,
                "seq_len": int(g["seq_len"]),
                "pred_len": int(g["pred_len"]),
                "metrics": metrics,
                "timestamp": str(mtime),
                "_mtime": mtime,
            })

        if not candidates:
            return self._empty_result(model_name, dataset, pred_len)

        candidates.sort(key=lambda c: c["_mtime"], reverse=True)
        best = candidates[0]
        best.pop("_mtime", None)
        return best

    # -----------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------

    @staticmethod
    def _read_metrics(setting: str) -> dict[str, float]:
        """
        从 result.txt 中读取指定 setting 对应的指标

        result.txt 格式：
            {setting}
            mse:{val}, mae:{val}, rse:{val}
        """
        result_txt = Path(_RESULT_TXT)
        if not result_txt.is_file():
            return {}

        try:
            with open(result_txt, "r") as f:
                lines = f.readlines()

            # 从后往前遍历，取最后一个匹配（最新的运行结果）
            for i in range(len(lines) - 1, -1, -1):
                if setting in lines[i] and i + 1 < len(lines):
                    metrics_line = lines[i + 1].strip()
                    metrics: dict[str, float] = {}
                    for part in metrics_line.split(","):
                        part = part.strip()
                        if ":" in part:
                            k, v = part.split(":", 1)
                            try:
                                metrics[k.strip()] = float(v.strip())
                            except ValueError:
                                pass
                    return metrics
        except Exception as e:
            logger.warning(f"[CheckpointSkill] 读取 result.txt 失败: {e}")

        return {}

    @staticmethod
    def _empty_result(
        model_name: str,
        dataset: str,
        pred_len: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            "checkpoint_path": "",
            "model_name": model_name,
            "dataset": dataset,
            "pred_len": pred_len or 96,
            "seq_len": 96,
            "metrics": {},
            "timestamp": "",
        }
