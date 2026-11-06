"""tests/test_app_skeleton.py — feat/app-skeleton 端到端测试（手册 §3.5 验收）。

验收标准：curl 一次体检端到端返回四段式报告——本文件为其自动化形态
（FastAPI TestClient），并覆盖：健康检查、空/短描述 422、免责声明强制在
响应、A9 降级路径端到端、管线注入（协议替换点）。

红线遵守：Mock 夹具全部〔2099〕人造文号＋自造条款原文（语料不入代码仓）。
"""

from __future__ import annotations

import json

from app.main import create_app
from app.pipeline import MockCheckupPipeline, PipelineResult
from fastapi.testclient import TestClient

from gen.schema import DISCLAIMER_TEXT, CheckupReport

client = TestClient(create_app())  # 默认 Mock 管线


# ---------------------------------------------------------------------------
# 健康检查与请求校验
# ---------------------------------------------------------------------------

def test_health_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["pipeline"] == "mock-skeleton"


def test_empty_description_422() -> None:
    assert client.post("/api/checkup", json={"description": "   "}).status_code == 422


def test_too_short_description_422() -> None:
    r = client.post("/api/checkup", json={"description": "修一条路"})
    assert r.status_code == 422 and "20" in r.json()["detail"][0]["msg"]


# ---------------------------------------------------------------------------
# 端到端（验收标准：一次请求返回四段式报告）
# ---------------------------------------------------------------------------

VALID = {"description": (
    "我村拟申请帮扶资金 60 万元，用于修建 2 公里产业道路和配套灌溉设施，"
    "由村集体经济组织作为实施主体，资金按项目进度拨付使用。"
)}


def test_checkup_end_to_end_returns_four_sections() -> None:
    r = client.post("/api/checkup", json=VALID)
    assert r.status_code == 200
    body = r.json()
    # 四段＋免责声明全部在渲染文本中（端到端验收核心断言）
    for section in ("① 适用政策清单", "② 负面清单命中提示", "③ 材料缺项清单", "④ 修改建议"):
        assert section in body["rendered_text"]
    assert body["rendered_text"].rstrip().endswith(f"免责声明：{DISCLAIMER_TEXT}")
    assert body["disclaimer"] == DISCLAIMER_TEXT
    # 结构化报告：置信由系统侧填充（LLM 不自评）、非降级
    assert body["degraded"] is False
    assert body["attempts"] == 1
    item = body["report"]["policy_items"][0]
    assert item["confidence"] == "明确适用"
    assert item["citation"] == "[C1]"          # 结构化层保留标记
    assert "[C1]" not in body["rendered_text"]  # 文本层已解析为引用行
    assert "粤测农〔2099〕1号 第四条（省，有效）" in body["rendered_text"]


def test_response_schema_fields_complete() -> None:
    body = client.post("/api/checkup", json=VALID).json()
    assert set(body) == {"pipeline", "report", "rendered_text", "degraded",
                         "attempts", "failed_sections", "disclaimer"}


# ---------------------------------------------------------------------------
# A9 降级路径端到端（注入两版生成：首版夹带编造引语→重试→次版通过）
# ---------------------------------------------------------------------------

def test_retry_path_recovers_before_degradation() -> None:
    bad = json.dumps({
        "policy_items": [{"citation": "[C1]",
                          "summary": "条款明确「帮扶资金可以用于建造办公楼」，需注意",
                          "confidence": None}],
        "missing_items": [{"citation": "[C2]", "name": "项目申报书", "note": "按指南核对"}],
        "suggestions": [{"text": "建议写明道路长度"}],
    }, ensure_ascii=False)
    pipe = MockCheckupPipeline(generation_outputs=(bad,))  # 单版坏输出→耗尽→降级
    r = TestClient(create_app(pipe)).post("/api/checkup", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True and body["attempts"] == 3
    assert "①适用政策" in body["failed_sections"]
    assert "① 适用政策清单〔已降级：未通过完整性校验，仅供参考〕" in body["rendered_text"]
    # 降级也不丢免责声明
    assert body["rendered_text"].rstrip().endswith(f"免责声明：{DISCLAIMER_TEXT}")


# ---------------------------------------------------------------------------
# 管线注入（协议替换点：真实适配器实现 CheckupPipeline 即可接入）
# ---------------------------------------------------------------------------

def test_custom_pipeline_injection() -> None:
    @dataclass_free_result
    class StubPipeline:
        name = "stub"

        def run(self, description: str):
            report = CheckupReport(
                project_summary=description,
                policy_items=(),
                negative_hits=(),
                missing_items=(),
                suggestions=(),
                uncovered_sections=("①适用政策", "③材料缺项"),
            )
            return PipelineResult(
                report=report,
                rendered_text=f"体检输入：{description}\n\n免责声明：{DISCLAIMER_TEXT}",
                degraded=False, attempts=1,
            )

    r = TestClient(create_app(StubPipeline())).post("/api/checkup", json=VALID)
    assert r.status_code == 200
    assert r.json()["pipeline"] == "stub"


def dataclass_free_result(cls):
    """测试辅助：直接返回类对象（仅用于注入 stub）。"""
    return cls
