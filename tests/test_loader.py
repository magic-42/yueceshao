"""rag.loader 的单元测试。

覆盖两类需求（对应《开发手册 v1.0》§3.1 PR#1 验收标准）：
1. 过滤：审计公告必须被剔除，且统计数据准确；
2. 字段映射：中文键 → 英文属性不串位，缺失字段有确定退化行为。

另有一条**可选**的集成测试，用真实数据仓验证 733 → 719 的清单规模。
数据仓不存在时自动跳过，因此 CI 与本地都能跑。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rag.loader import (
    AUDIT_CATEGORY,
    DocRecord,
    LoadResult,
    RegistryFormatError,
    build_doc_record,
    load_registry,
    read_registry,
)

FIXTURE = Path(__file__).parent / "fixtures" / "registry_sample.json"


@pytest.fixture()
def fixture_records() -> list[dict]:
    """夹具里的 5 条登记记录（4 条可用 + 1 条审计公告）。"""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------- 过滤行为


def test_audit_documents_are_excluded() -> None:
    """默认加载时审计公告被剔除：5 条进、4 条出。"""
    result = load_registry(FIXTURE)

    assert isinstance(result, LoadResult)
    assert result.total == 5
    assert result.excluded_audit == 1
    assert result.kept == 4
    assert all(not doc.is_audit for doc in result.documents)


def test_audit_can_be_kept_for_debug() -> None:
    """include_audit=True 时保留全部，且不记剔除数（仅调试用）。"""
    result = load_registry(FIXTURE, include_audit=True)

    assert result.kept == 5
    assert result.excluded_audit == 0
    assert sum(1 for doc in result.documents if doc.is_audit) == 1


def test_audit_detection_uses_category_only() -> None:
    """审计公告的判定只看「类别」，与标题、路径无关。"""
    payload = {
        "序号": 999,
        "标题": "关于乡村振兴资金的审计结果公告（标题被改成普通文件）",
        "类别": AUDIT_CATEGORY,
        "清洗文本": "03_清洗文本/999_审计.txt",
    }
    assert build_doc_record(payload).is_audit is True


# --------------------------------------------------------------------- 字段映射


def test_field_mapping() -> None:
    """逐字段比对第一条（粤府〔2023〕34号），确认中文键与英文属性不串位。"""
    result = load_registry(FIXTURE)
    doc = next(d for d in result.documents if d.serial == 2)

    assert isinstance(doc, DocRecord)
    assert doc.doc_no == "粤府〔2023〕34号"
    assert doc.title.startswith("广东省人民政府关于印发")
    assert doc.category == "资金管理办法"
    assert doc.validity == "有效"
    assert doc.issuer == "广东省人民政府"
    assert doc.publisher == "广东省财政厅"
    assert doc.publish_date == "2023-05-19"
    assert doc.source_url == "https://czt.gd.gov.cn/czfg/content/post_4183847.html"
    assert doc.clean_text_path == "03_清洗文本/02_粤府〔2023〕34号.txt"
    assert doc.clean_text_chars == 6866
    assert doc.verify_status == "已人工核验"
    assert doc.fetched_date == "2026-10-29"
    assert doc.valid_until == ""


def test_raw_keeps_all_original_fields() -> None:
    """整条原始记录被保留，未显式映射的字段也能取到。"""
    result = load_registry(FIXTURE)
    doc = next(d for d in result.documents if d.serial == 2)

    assert doc.raw["来源栏目"] == "财政法规"
    assert doc.raw["正文长度"] == 21301
    assert doc.raw["类别依据"] == "标题含「专项资金管理办法」"


def test_missing_fields_degrade_safely() -> None:
    """缺键、None、非数字时不抛异常，退化为空串或 0。"""
    doc = build_doc_record({"序号": "7", "标题": None, "清洗文本字符数": "约三千"})

    assert doc.serial == 7
    assert doc.title == ""
    assert doc.clean_text_chars == 0
    assert doc.doc_no == ""


def test_serial_accepts_string_and_float() -> None:
    """序号在数据仓里可能被存成字符串或浮点，都应正确转成整数。"""
    assert build_doc_record({"序号": "22"}).serial == 22
    assert build_doc_record({"序号": 22.0}).serial == 22
    assert build_doc_record({"序号": " 22 "}).serial == 22


# ----------------------------------------------------------------- 派生语义属性


def test_has_doc_no_rejects_placeholder_text() -> None:
    """数据仓用「无（原因）」表示没有文号，不能当真实文号渲染。"""
    real = build_doc_record({"文号": "粤府〔2023〕34号"})
    placeholder = build_doc_record({"文号": "无（审计结果公告通常无发文字号）"})
    empty = build_doc_record({"文号": ""})

    assert real.has_doc_no is True
    assert placeholder.has_doc_no is False
    assert empty.has_doc_no is False


def test_is_current_excludes_pending_verification() -> None:
    """待人工核验不等于现行有效，避免把未定状态当有效用。"""
    assert build_doc_record({"效力状态": "有效"}).is_current is True
    assert build_doc_record({"效力状态": "已失效"}).is_current is False
    assert build_doc_record({"效力状态": "待人工核验"}).is_current is False


def test_resolve_clean_text_uses_pathlib(tmp_path: Path) -> None:
    """清洗文本路径按数据仓根目录解析，且不在代码里写死分隔符。"""
    doc = build_doc_record({"清洗文本": "03_清洗文本/02_粤府〔2023〕34号.txt"})
    resolved = doc.resolve_clean_text(tmp_path)

    assert resolved == tmp_path / "03_清洗文本" / "02_粤府〔2023〕34号.txt"


# ------------------------------------------------------------------- 异常与边界


def test_missing_registry_raises(tmp_path: Path) -> None:
    """文件不存在时明确报错，不返回空列表糊弄调用方。"""
    with pytest.raises(FileNotFoundError):
        read_registry(tmp_path / "registry.json")


def test_top_level_must_be_list(tmp_path: Path) -> None:
    """顶层是对象而非数组时报格式错误。"""
    bad = tmp_path / "registry.json"
    bad.write_text('{"序号": 1}', encoding="utf-8")

    with pytest.raises(RegistryFormatError):
        read_registry(bad)


def test_record_must_be_mapping() -> None:
    """条目是字符串等非对象时报格式错误。"""
    with pytest.raises(RegistryFormatError):
        build_doc_record("粤府〔2023〕34号")  # type: ignore[arg-type]


# --------------------------------------------------------------- 真实数据集成测试


def _find_real_registry() -> Path | None:
    """在常见位置找真实 registry.json；找不到返回 None（测试自动跳过）。"""
    candidates: list[Path] = []

    env = os.environ.get("YUECESHAO_DATA_REPO")
    if env:
        candidates.append(Path(env))

    project_root = Path(__file__).resolve().parents[1]  # 代码仓根目录
    for name in ("yueceshao-data", "yueceshao-data.partial", "yueceshao-data-full"):
        candidates.append(project_root.parent / name)

    for base in candidates:
        candidate = base / "00_工作区" / "registry.json"
        if candidate.is_file():
            return candidate
    return None


REAL_REGISTRY = _find_real_registry()


@pytest.mark.skipif(REAL_REGISTRY is None, reason="本机未克隆数据仓，跳过集成测试")
def test_real_registry_list_size() -> None:
    """验收标准：数据仓登记 733 篇，剔除 14 篇审计公告后保留 719 篇。

    数字来自《开发手册 v1.0》§3.1 PR#1。若数据仓后续增补文件，本断言应当
    同步更新——它同时也是「是否有新增文件未纳入登记」的哨兵。
    """
    assert REAL_REGISTRY is not None
    result = load_registry(REAL_REGISTRY)

    assert result.total == 733
    assert result.excluded_audit == 14
    assert result.kept == 719
    assert len({doc.serial for doc in result.documents}) == 719, "序号应唯一"
    assert all(doc.clean_text_path for doc in result.documents), "每篇都应有清洗文本路径"
    assert all(not doc.is_audit for doc in result.documents), "结果里不应残留审计公告"
