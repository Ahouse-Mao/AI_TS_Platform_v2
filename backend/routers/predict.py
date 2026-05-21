"""
推理 API 路由 — POST /api/infer

符合 api_skill.py 中定义的 API 规范：
- POST /api/infer  启动推理
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.task_manager import task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["推理"])


# ============================================================
# 请求/响应模型
# ============================================================

class InferRequest(BaseModel):
    """推理任务请求参数"""
    model_name: str = Field("DLinear", description="模型名称")
    dataset: str = Field("ETTh1", description="数据集名称")
    seq_len: int = Field(96, ge=1, description="输入序列长度")
    pred_len: int = Field(96, ge=1, description="预测长度")
    label_len: int = Field(48, ge=0, description="起始 token 长度")
    batch_size: int = Field(64, ge=1, description="批次大小")
    features: str = Field("M", pattern="^(M|S|MS)$", description="预测任务类型")
    target: str = Field("OT", description="目标特征名")
    freq: str = Field("h", description="时间频率")
    use_gpu: bool = Field(True, description="是否使用 GPU")
    checkpoint_path: Optional[str] = Field(None, description="检查点路径（留空则自动查找）")
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
    # PatchTST 参数
    patch_len: int = Field(16, ge=1, description="Patch 长度")
    stride: int = Field(8, ge=1, description="Patch 步长")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "DLinear",
                "dataset": "ETTh1",
                "seq_len": 96,
                "pred_len": 96,
                "checkpoint_path": None,
            }
        }


class InferResponse(BaseModel):
    """推理任务创建响应"""
    message: str
    task_id: str
    status: str
    progress: int


# ============================================================
# 路由
# ============================================================

@router.post("/infer", response_model=InferResponse)
async def start_inference(req: InferRequest):
    """
    启动推理任务

    接收模型参数和检查点路径（可选），在后台执行推理。
    返回 task_id 用于轮询状态。
    """
    params = req.model_dump()
    task = task_manager.create_task("inference", params)

    logger.info(f"[InferAPI] 推理任务已创建: {task.task_id}")
    return InferResponse(
        message="推理任务已创建",
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
    )
