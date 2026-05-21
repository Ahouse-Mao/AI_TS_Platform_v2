"""
训练 API 路由 — POST /api/train, GET /api/status/{task_id}

符合 api_skill.py 中定义的 API 规范：
- POST /api/train  启动训练
- GET  /api/status/{task_id}  查询任务状态
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.task_manager import task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["训练"])


# ============================================================
# 请求/响应模型
# ============================================================

class TrainRequest(BaseModel):
    """训练任务请求参数"""
    model_name: str = Field("DLinear", description="模型名称 (DLinear/PatchTST/Autoformer/...)")
    dataset: str = Field("ETTh1", description="数据集名称")
    seq_len: int = Field(96, ge=1, le=4096, description="输入序列长度")
    pred_len: int = Field(96, ge=1, le=4096, description="预测长度")
    label_len: int = Field(48, ge=0, description="起始 token 长度")
    batch_size: int = Field(64, ge=1, description="批次大小")
    learning_rate: float = Field(0.005, gt=0, lt=1, description="学习率")
    epochs: int = Field(50, ge=1, description="训练轮数")
    patience: int = Field(5, ge=1, description="早停耐心值")
    features: str = Field("M", pattern="^(M|S|MS)$", description="预测任务类型")
    target: str = Field("OT", description="目标特征名")
    freq: str = Field("h", description="时间频率")
    use_gpu: bool = Field(True, description="是否使用 GPU")
    enc_in: int = Field(7, ge=1, description="编码器输入特征数")
    dec_in: int = Field(7, ge=1, description="解码器输入特征数")
    c_out: int = Field(7, ge=1, description="输出特征数")
    d_model: int = Field(512, ge=16, description="模型维度")
    n_heads: int = Field(8, ge=1, description="注意力头数")
    e_layers: int = Field(2, ge=1, description="编码器层数")
    d_layers: int = Field(1, ge=1, description="解码器层数")
    d_ff: int = Field(2048, ge=32, description="前馈网络维度")
    dropout: float = Field(0.05, ge=0, le=1, description="Dropout 比率")
    embed: str = Field("timeF", description="时间特征编码方式")
    lradj: str = Field("type3", description="学习率调整策略")
    use_amp: bool = Field(False, description="是否使用混合精度训练")
    # PatchTST 参数
    patch_len: int = Field(16, ge=1, description="Patch 长度")
    stride: int = Field(8, ge=1, description="Patch 步长")
    # 其他通用参数
    do_predict: bool = Field(False, description="是否预测未来数据")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "DLinear",
                "dataset": "ETTh1",
                "seq_len": 96,
                "pred_len": 96,
                "batch_size": 64,
                "learning_rate": 0.005,
                "epochs": 50,
            }
        }


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    task_type: str
    status: str
    progress: int
    params: dict[str, Any] = {}
    error: str | None = None
    checkpoint_path: str | None = None
    log_path: str | None = None
    metrics: dict[str, Any] = {}
    predictions_path: str | None = None
    created_at: str = ""
    completed_at: str | None = None


class TaskCreateResponse(BaseModel):
    """任务创建响应"""
    message: str
    task_id: str
    status: str
    progress: int


# ============================================================
# 路由
# ============================================================

@router.post("/train", response_model=TaskCreateResponse)
async def start_training(req: TrainRequest):
    """
    启动训练任务

    接收模型参数，在后台启动训练，返回 task_id 用于轮询状态。
    """
    params = req.model_dump()
    task = task_manager.create_task("train", params)

    logger.info(f"[TrainAPI] 训练任务已创建: {task.task_id}")
    return TaskCreateResponse(
        message="训练任务已创建",
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    查询任务状态

    返回任务当前状态、进度、结果（已完成时）。
    """
    task = task_manager.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    return TaskStatusResponse(**task.to_dict())


@router.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(status: str | None = None):
    """
    列出所有任务

    - status (可选): 按状态过滤 (pending/running/completed/failed)
    """
    tasks = task_manager.list_tasks(status=status)
    return [TaskStatusResponse(**t) for t in tasks]
