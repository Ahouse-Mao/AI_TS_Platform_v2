"""
Task Manager — 后台训练/推理任务管理器

职责：
- 在后台线程中运行训练/推理任务
- 提供任务状态查询接口
- 收集训练日志、指标、检查点路径
- 支持并发多个任务同时运行
"""

import os
import sys
import time
import uuid
import json
import logging
import threading
import traceback
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---- 常量 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SRC_DIR = os.path.join(SCRIPT_DIR, "model_src")
CHECKPOINTS_DIR = os.path.join(MODEL_SRC_DIR, "checkpoints")
RESULTS_DIR = os.path.join(MODEL_SRC_DIR, "results")

# exp.test() 使用相对路径 "result.txt" 写入，CWD=项目根目录
RESULT_TXT_PATH = os.path.join(SCRIPT_DIR, "..", "result.txt")


# ============================================================
# 任务状态定义
# ============================================================

@dataclass
class TaskRecord:
    """单个任务的完整记录"""
    task_id: str
    task_type: str                          # "train" | "inference"
    status: str                             # "pending" | "running" | "completed" | "failed"
    progress: int                           # 0-100
    params: dict[str, Any]                  # 用户传入的参数副本
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None   # 训练后生成的检查点路径
    log_path: Optional[str] = None          # 日志或结果文件路径
    metrics: dict[str, Any] = field(default_factory=dict)   # MSE, MAE, RMSE 等
    predictions_path: Optional[str] = None  # 推理结果文件路径
    created_at: str = ""
    completed_at: Optional[str] = None
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": self.progress,
            "params": self.params,
            "error": self.error,
            "checkpoint_path": self.checkpoint_path,
            "log_path": self.log_path,
            "metrics": self.metrics,
            "predictions_path": self.predictions_path,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ============================================================
# 参数转换：API JSON → argparse Namespace
# ============================================================

def build_args(params: dict[str, Any], is_training: bool = True) -> object:
    """
    将 API JSON 参数映射为与 run_longExp.py 兼容的命名空间对象。
    未指定的参数使用默认值。
    """
    import argparse
    import torch

    class_namespace = {
        # random seed
        "random_seed": params.get("random_seed", 2021),
        # basic config
        "is_training": 1 if is_training else 0,
        "model_id": None,
        "model": params.get("model_name", "DLinear"),
        # data loader
        "data": params.get("dataset", "ETTh1"),
        "root_path": os.path.join(MODEL_SRC_DIR, "dataset"),
        "data_path": params.get("data_path", f"{params.get('dataset', 'ETTh1')}.csv"),
        "features": params.get("features", "M"),
        "target": params.get("target", "OT"),
        "freq": params.get("freq", "h"),
        "checkpoints": CHECKPOINTS_DIR,
        # forecasting task
        "seq_len": params.get("seq_len", 96),
        "label_len": params.get("label_len", 48),
        "pred_len": params.get("pred_len", 96),
        # PatchTST specific
        "fc_dropout": params.get("fc_dropout", 0.05),
        "head_dropout": params.get("head_dropout", 0.0),
        "patch_len": params.get("patch_len", 16),
        "stride": params.get("stride", 8),
        "padding_patch": params.get("padding_patch", "end"),
        "revin": params.get("revin", 1),
        "affine": params.get("affine", 0),
        "subtract_last": params.get("subtract_last", 0),
        "decomposition": params.get("decomposition", 0),
        "kernel_size": params.get("kernel_size", 25),
        "individual": params.get("individual", 0),
        # Formers
        "embed_type": params.get("embed_type", 0),
        "enc_in": params.get("enc_in", 7),
        "dec_in": params.get("dec_in", 7),
        "c_out": params.get("c_out", 7),
        "d_model": params.get("d_model", 512),
        "n_heads": params.get("n_heads", 8),
        "e_layers": params.get("e_layers", 2),
        "d_layers": params.get("d_layers", 1),
        "d_ff": params.get("d_ff", 2048),
        "moving_avg": params.get("moving_avg", 25),
        "factor": params.get("factor", 1),
        "distil": params.get("distil", True),
        "dropout": params.get("dropout", 0.05),
        "embed": params.get("embed", "timeF"),
        "activation": params.get("activation", "gelu"),
        "output_attention": params.get("output_attention", False),
        "do_predict": params.get("do_predict", False),
        # optimization
        "num_workers": params.get("num_workers", 0),
        "itr": params.get("itr", 1),
        "train_epochs": params.get("epochs", 50),
        "batch_size": params.get("batch_size", 64),
        "patience": params.get("patience", 5),
        "learning_rate": params.get("learning_rate", 0.005),
        "des": params.get("des", "Exp"),
        "loss": params.get("loss", "mse"),
        "lradj": params.get("lradj", "type3"),
        "pct_start": params.get("pct_start", 0.3),
        "use_amp": params.get("use_amp", False),
        # GPU
        "use_gpu": params.get("use_gpu", True if torch.cuda.is_available() else False),
        "gpu": params.get("gpu", 0),
        "use_multi_gpu": params.get("use_multi_gpu", False),
        "devices": params.get("devices", "0,1,2,3"),
        "test_flop": params.get("test_flop", False),
    }

    # 自动生成 model_id
    data = class_namespace["data"]
    seq_len = class_namespace["seq_len"]
    pred_len = class_namespace["pred_len"]
    class_namespace["model_id"] = f"{data}_{seq_len}_{pred_len}"

    # 兼容 run_longExp.py 的 argparse 行为
    class Args:
        pass

    args = Args()
    for k, v in class_namespace.items():
        setattr(args, k, v)

    # GPU 多卡设置
    if args.use_gpu and args.use_multi_gpu:
        device_ids = args.devices.replace(" ", "").split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    return args


def _make_setting(args: object, ii: int = 0) -> str:
    """生成与 run_longExp.py 一致的 setting 字符串"""
    return "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii,
    )


# ============================================================
# 核心训练/推理函数
# ============================================================

def _run_training(params: dict[str, Any], task: TaskRecord) -> None:
    """
    在后台线程中执行训练任务。
    直接修改 task 对象的字段来报告进度。
    """
    import random

    try:
        task.status = "running"
        task.progress = 5

        args = build_args(params, is_training=True)
        setting = _make_setting(args, ii=0)
        logger.info(f"[TaskManager] 训练启动: setting={setting}")

        # 随机种子
        fix_seed = args.random_seed
        random.seed(fix_seed)
        np.random.seed(fix_seed)
        import torch
        torch.manual_seed(fix_seed)

        # 设置 GPU
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(" ", "")
            device_ids = args.devices.split(",")
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        # 创建实验实例
        from backend.model_src.exp.exp_main import Exp_Main
        exp = Exp_Main(args)

        # 训练
        task.progress = 10
        logger.info(f"[TaskManager] >>>>>>> 开始训练: {setting}")
        exp.train(setting)

        task.progress = 60

        # 测试
        logger.info(f"[TaskManager] >>>>>>> 开始测试: {setting}")
        exp.test(setting)

        task.progress = 85

        # 记录结果
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, setting, "checkpoint.pth")
        result_path = os.path.join(RESULTS_DIR, setting, "pred.npy")

        # 从 result.txt 读取指标（exp.test 会将指标写入该文件）
        metrics = _parse_result_txt(setting)
        if not metrics and os.path.exists(result_path):
            # 如果没有 result.txt，尝试从 npy 文件计算
            logger.info(f"[TaskManager] 从 result.txt 读取指标失败，尝试从 npy 计算")
            metrics = _compute_metrics_from_npy(result_path)

        task.checkpoint_path = checkpoint_path
        task.log_path = os.path.join(RESULTS_DIR, setting)
        task.metrics = metrics
        task.status = "completed"
        task.progress = 100
        task.completed_at = datetime.now().isoformat()
        logger.info(f"[TaskManager] 训练完成: setting={setting}, metrics={metrics}")

    except Exception as e:
        task.status = "failed"
        task.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(f"[TaskManager] 训练失败: {e}")
        task.completed_at = datetime.now().isoformat()


def _run_inference(params: dict[str, Any], task: TaskRecord) -> None:
    """
    在后台线程中执行推理任务。
    需要提供 checkpoint_path，否则会尝试自动查找。
    """
    import random

    try:
        task.status = "running"
        task.progress = 5

        args = build_args(params, is_training=False)
        setting = _make_setting(args, ii=0)
        logger.info(f"[TaskManager] 推理启动: setting={setting}")

        # 随机种子
        fix_seed = args.random_seed
        random.seed(fix_seed)
        np.random.seed(fix_seed)
        import torch
        torch.manual_seed(fix_seed)

        # 设置 GPU
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(" ", "")
            device_ids = args.devices.split(",")
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        # 检查点路径
        checkpoint_path = params.get("checkpoint_path", "")
        if not checkpoint_path:
            # 自动查找
            default_path = os.path.join(CHECKPOINTS_DIR, setting, "checkpoint.pth")
            if os.path.exists(default_path):
                checkpoint_path = default_path
                logger.info(f"[TaskManager] 自动找到检查点: {checkpoint_path}")

        # 创建实验实例
        from backend.model_src.exp.exp_main import Exp_Main
        exp = Exp_Main(args)

        task.progress = 20

        # 执行测试（用验证/测试集评估）
        exp.test(setting, test=1)

        task.progress = 60

        # 读取结果
        result_path = os.path.join(RESULTS_DIR, setting, "pred.npy")
        if os.path.exists(result_path):
            predictions = np.load(result_path)
            task.predictions_path = result_path

        # 指标
        metrics = _parse_result_txt(setting)
        if not metrics and os.path.exists(result_path):
            metrics = _compute_metrics_from_npy(result_path)

        task.checkpoint_path = checkpoint_path
        task.log_path = os.path.join(RESULTS_DIR, setting)
        task.metrics = metrics
        task.status = "completed"
        task.progress = 100
        task.completed_at = datetime.now().isoformat()
        logger.info(f"[TaskManager] 推理完成: setting={setting}, metrics={metrics}")

    except Exception as e:
        task.status = "failed"
        task.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(f"[TaskManager] 推理失败: {e}")
        task.completed_at = datetime.now().isoformat()


# ============================================================
# 辅助函数
# ============================================================

def _parse_result_txt(setting: str) -> dict[str, float]:
    """
    从 result.txt 解析指标的辅助函数。

    result.txt 格式（由 exp.test 追加写入）：
        {setting}
        mse:0.123, mae:0.456, rse:1.234

    注意：同一个 setting 可能出现多次（不同超参数），
    所以从后往前遍历，取最后一次匹配的结果。
    """
    result_txt_path = RESULT_TXT_PATH
    if not os.path.exists(result_txt_path):
        logger.warning(f"[TaskManager] result.txt 不存在: {result_txt_path}")
        return {}

    try:
        with open(result_txt_path, "r") as f:
            lines = f.readlines()

        # 从后往前遍历，找最后一个匹配项（最近一次运行的结果）
        for i in range(len(lines) - 1, -1, -1):
            if setting in lines[i] and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                metrics: dict[str, float] = {}
                parts = next_line.split(",")
                for part in parts:
                    part = part.strip()
                    if ":" in part:
                        key, val = part.split(":", 1)
                        try:
                            metrics[key.strip()] = float(val.strip())
                        except ValueError:
                            pass
                if metrics:
                    logger.info(
                        f"[TaskManager] result.txt 取最后匹配: "
                        f"行 {i+1} → {metrics}"
                    )
                    return metrics
    except Exception as e:
        logger.warning(f"[TaskManager] 解析 result.txt 失败: {e}")

    return {}


def _compute_metrics_from_npy(pred_path: str) -> dict[str, float]:
    """
    从 pred.npy 计算指标（兜底方案）。
    需要对应的 true 值文件。
    """
    from backend.model_src.utils.metrics import metric

    try:
        preds = np.load(pred_path)
        # 尝试找 true.npy
        true_path = pred_path.replace("pred.npy", "true.npy")
        if not os.path.exists(true_path):
            true_path = pred_path.replace("pred.npy", "trues.npy")
        if os.path.exists(true_path):
            trues = np.load(true_path)
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
                "rse": float(rse),
            }
    except Exception as e:
        logger.warning(f"[TaskManager] 从 npy 计算指标失败: {e}")

    return {}


# ============================================================
# 任务管理器（单例）
# ============================================================

class TaskManager:
    """全局任务管理器，管理运行中的后台任务"""

    _instance: Optional["TaskManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "TaskManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: dict[str, TaskRecord] = {}
                    cls._instance._tasks_lock = threading.Lock()
        return cls._instance

    def create_task(self, task_type: str, params: dict[str, Any]) -> TaskRecord:
        """创建并启动一个新任务"""
        task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
        task = TaskRecord(
            task_id=task_id,
            task_type=task_type,
            status="pending",
            progress=0,
            params=params,
            created_at=datetime.now().isoformat(),
        )

        with self._tasks_lock:
            self._tasks[task_id] = task

        # 在后台线程中启动
        target = _run_training if task_type == "train" else _run_inference
        thread = threading.Thread(target=target, args=(params, task), daemon=True)
        task._thread = thread
        thread.start()

        logger.info(f"[TaskManager] 任务已创建: {task_id} ({task_type})")
        return task

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """获取任务记录"""
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        """列出所有任务，可选按状态过滤"""
        with self._tasks_lock:
            tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        return [t.to_dict() for t in tasks]


# 全局单例
task_manager = TaskManager()
