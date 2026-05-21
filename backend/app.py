"""
AI-TS-Platform Backend — FastAPI 主入口

提供训练/推理 API 服务，供 Agent 集群调用。

启动方式：
    uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.train import router as train_router
from backend.routers.predict import router as infer_router

# ---- 日志配置 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期钩子"""
    logger.info("=" * 50)
    logger.info("AI-TS-Platform Backend 启动")
    logger.info(f"训练检查点目录: backend/model_src/checkpoints")
    logger.info(f"数据集目录: backend/model_src/dataset")
    logger.info("=" * 50)
    yield
    logger.info("AI-TS-Platform Backend 关闭")


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="AI-TS-Platform Backend",
    description="时序预测智能体集群 — 训练/推理 API 服务",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(train_router)
app.include_router(infer_router)


# ============================================================
# 根路径
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "AI-TS-Platform Backend",
        "version": "0.1.0",
        "endpoints": {
            "POST /api/train": "启动训练任务",
            "POST /api/infer": "启动推理任务",
            "GET  /api/status/{task_id}": "查询任务状态",
            "GET  /api/tasks": "列所有任务",
            "GET  /health": "健康检查",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
