"""app/main.py — FastAPI 服务骨架（手册 §3.5 feat/app-skeleton）。

路由：
- POST /api/checkup：接收项目描述 → 体检管线 → 四段式报告（结构化 JSON＋渲染
  文本），响应显式携带免责声明（红线：免责声明由渲染层强制注入，此处复核）；
- GET /health：存活检查＋当前管线名。

配置与装配：
- 管线经 create_app(pipeline=...) 注入（依赖倒置：骨架默认 MockCheckupPipeline，
  真实适配器——rag 检索接索引／rules 接数据仓／qwen API——实现同一 CheckupPipeline
  协议后于此替换，路由零改动）；
- 配置走环境变量 / config.json（不硬编码 key）；日志用 logging。

验收标准（开发手册 §3.5）：curl 一次体检端到端返回四段式报告——自动化形态见
tests/test_app_skeleton.py（TestClient 端到端）。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from app.pipeline import CheckupPipeline, MockCheckupPipeline, PipelineResult
from gen.schema import DISCLAIMER_TEXT

logger = logging.getLogger(__name__)

# 免责声明哨兵＝完整产品级原文（渲染层注入的即此文本，逐字比对最严格）
DISCLAIMER_SENTINEL = DISCLAIMER_TEXT


class CheckupRequest(BaseModel):
    """体检请求体：项目描述（简介／资金用途／建设内容）。"""

    description: str

    @field_validator("description")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("description 不能为空")
        if len(v.strip()) < 20:
            raise ValueError("description 过短（至少 20 字），请按「简介＋资金用途＋建设内容」补全")
        return v.strip()


class CheckupResponse(BaseModel):
    """体检响应：pipeline 名＋结构化四段＋渲染文本＋降级标记＋免责声明。"""

    pipeline: str
    report: dict
    rendered_text: str
    degraded: bool
    attempts: int
    failed_sections: list[str]
    disclaimer: str


def create_app(pipeline: CheckupPipeline | None = None) -> FastAPI:
    """装配 FastAPI 实例（pipeline 缺省为 Mock——骨架阶段，真实适配器后置替换）。"""
    pipe = pipeline or MockCheckupPipeline()
    app = FastAPI(
        title="粤策哨 · 申报前体检 API",
        version="0.1.0-skeleton",
        description="县域强村公司与合作社申报前自查辅助（演示骨架；不构成合规判定）",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "pipeline": pipe.name}

    @app.post("/api/checkup", response_model=CheckupResponse)
    def checkup(req: CheckupRequest) -> CheckupResponse:
        logger.info("体检请求：%d 字", len(req.description))
        try:
            result: PipelineResult = pipe.run(req.description)
        except Exception as exc:  # noqa: BLE001 —— 骨架阶段兜底成 500＋日志，不把堆栈泄给客户端
            logger.exception("体检管线执行失败")
            raise HTTPException(status_code=500, detail="体检执行失败，请稍后重试") from exc
        # 免责声明最后防线：管线产物缺失即 500（渲染层保证其存在，此处兜底复核）
        if DISCLAIMER_SENTINEL not in result.rendered_text:
            logger.error("管线产物缺少免责声明，拒绝返回")
            raise HTTPException(status_code=500, detail="服务内部校验失败")
        return CheckupResponse(
            pipeline=pipe.name,
            report=result.to_api_dict()["report"],
            rendered_text=result.rendered_text,
            degraded=result.degraded,
            attempts=result.attempts,
            failed_sections=list(result.failed_sections),
            disclaimer=result.disclaimer,
        )

    return app


app = create_app()
