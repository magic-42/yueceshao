"""tests/test_index_verify.py — 真实索引验证工具的单元测试＋本机集成冒烟。

单元测试覆盖 verify_index 的纯函数口径（必备字段、审计公告泄漏判定、结论汇聚）；
真实索引集成用例沿用本仓既有的 skip 模式：``config.json`` 与 ``.chroma/chroma.sqlite3``
不在位（CI／未解压索引的机器）时自动跳过，保证 CI 全绿。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.verify_index import (
    DEFAULT_EXPECTED_COUNT,
    REQUIRED_METADATA,
    IndexVerification,
    SampleResult,
    check_required_fields,
    find_audit_leaks,
    verify_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
INDEX_DIR = PROJECT_ROOT / ".chroma"


def _index_ready() -> bool:
    """本机是否具备真实索引（config.json＋.chroma/chroma.sqlite3）。"""
    if not CONFIG_PATH.is_file():
        return False
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    persist = Path(str((payload.get("index") or {}).get("persist_dir") or ".chroma"))
    if not persist.is_absolute():
        persist = PROJECT_ROOT / persist
    return (persist / "chroma.sqlite3").is_file()


READY = _index_ready()


# ---------------------------------------------------------------------------
# 纯函数：必备字段与审计公告泄漏
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_complete_row_passes(self) -> None:
        row = {
            "doc_no": "粤财农〔2026〕100号",
            "clause": "第四条",
            "validity": "有效",
            "is_current": True,
        }
        assert check_required_fields(row) == ()

    def test_missing_fields_are_listed_in_required_order(self) -> None:
        assert check_required_fields({"doc_no": "x"}) == ("clause", "validity", "is_current")

    def test_required_metadata_is_the_task_contract(self) -> None:
        assert REQUIRED_METADATA == ("doc_no", "clause", "validity", "is_current")


class TestAuditLeaks:
    def test_audit_category_is_a_leak_even_if_serial_looked_kept(self) -> None:
        records = [(1, "资金管理办法", "文件甲"), (2, "审计公告", "公告乙")]
        assert find_audit_leaks(records, {1, 2}) == ((2, "审计公告", "公告乙"),)

    def test_serial_not_kept_by_loader_is_a_leak(self) -> None:
        records = [(1, "资金管理办法", "文件甲"), (9, "资金管理办法", "文件丙")]
        assert find_audit_leaks(records, {1}) == ((9, "资金管理办法", "文件丙"),)

    def test_clean_index_has_no_leak(self) -> None:
        records = [(1, "资金管理办法", "文件甲"), (3, "申报指南", "指南丙")]
        assert find_audit_leaks(records, {1, 3}) == ()


class TestVerdict:
    def _result(self, **overrides) -> IndexVerification:
        defaults: dict = {
            "persist_dir": Path(".chroma"),
            "collection": "yueceshao_clauses",
            "count": DEFAULT_EXPECTED_COUNT,
            "expected_count": DEFAULT_EXPECTED_COUNT,
            "samples": (SampleResult("73-0001", ()),),
            "doc_count": 717,
            "audit_leaks": (),
            "warnings": (),
        }
        defaults.update(overrides)
        return IndexVerification(**defaults)

    def test_ok_when_count_matches_and_no_problems(self) -> None:
        assert self._result().ok is True
        assert str(DEFAULT_EXPECTED_COUNT) in self._result().summary()

    def test_count_mismatch_fails(self) -> None:
        assert self._result(count=1).ok is False

    def test_sample_missing_field_fails(self) -> None:
        bad = SampleResult("73-0001", ("validity",))
        assert self._result(samples=(bad,)).ok is False

    def test_audit_leak_fails(self) -> None:
        assert self._result(audit_leaks=((162, "审计公告", "某公告"),)).ok is False


# ---------------------------------------------------------------------------
# 真实索引集成（本机可选）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not READY, reason="本机无真实 Chroma 索引（config.json/.chroma 不在位），跳过集成冒烟")
def test_real_index_count_and_samples() -> None:
    """真实索引：块数=18262、抽检 10 块 metadata 齐全、审计公告零泄漏。"""
    result = verify_index(config_path=CONFIG_PATH)
    assert result.count == DEFAULT_EXPECTED_COUNT
    assert result.doc_count == 717
    assert all(sample.ok for sample in result.samples), result.summary()
    assert result.audit_leaks == (), result.summary()
