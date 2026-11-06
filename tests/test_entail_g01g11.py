"""tests/test_entail_g01g11.py — G01-G11 禁止用途载体加载（方向4收尾，TDD 先红后绿）。

覆盖：G01-G11 载体判据、按原文去重（同文保留层级最具体）、命中块 origin 标记、
渲染按 origin 过滤、真实载体加载（集成冒烟，无数据仓时 skip）。

红线：合成夹具用〔2099〕人造文号与自造条款原文。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.real_components import (
    ORIGIN_NEGATIVE,
    ORIGIN_RETRIEVAL,
    ORIGIN_RULES,
    RealComponents,
    dedup_carriers,
    is_g01g11_carrier,
    render_context_blocks,
    rule_orchestration_hit,
)

from rag.retriever import Hit, RetrievalResult
from rules.engine import NegRuleMatcher
from rules.loader import records_from_dicts
from rules.schema import RuleRecord, RuleStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"


def _record(**overrides) -> RuleRecord:
    defaults: dict = {
        "clause_text": "项目资金不得用于修建楼堂馆所。",
        "source_ref": "粤测农〔2099〕1号 第四条",
        "clause_no": "四",
        "carries_enumeration": False,
        "level": "省",
        "effect_status": "有效",
        "group_id": "G04",
        "group_name": "楼堂馆所",
        "group_type": "锚词组",
        "is_negative_list_file": True,
        "doc_index": 1,
        "doc_no": "粤测农〔2099〕1号",
        "doc_title": "测试用资金管理办法",
        "doc_issuer": "测试机关",
        "doc_date": "2099-01-01",
        "doc_category": "资金管理办法",
        "group_decision": "接受",
    }
    defaults.update(overrides)
    return RuleRecord(**defaults)


def _hit() -> Hit:
    return Hit(
        chunk_id="73-0010",
        text="申报主体应当提交项目申报书。",
        score=0.9,
        distance=0.1,
        metadata={
            "doc_no": "粤测农〔2099〕2号", "clause": "第三条", "clause_index": 3,
            "title": "测试用申报指南", "serial": 73, "category": "申报指南",
            "validity": "有效", "is_current": True, "has_doc_no": True,
        },
    )


class FakeRetriever:
    def __init__(self, hits: tuple[Hit, ...]) -> None:
        self._hits = hits

    def retrieve(self, query: str, **_kwargs) -> RetrievalResult:
        return RetrievalResult(query=query, hits=list(self._hits), where={"is_current": True})


# ---------------------------------------------------------------------------
# 单元：载体判据
# ---------------------------------------------------------------------------

class TestCarrierPredicate:
    def test_g01g11_with_negative_word_is_carrier(self) -> None:
        assert is_g01g11_carrier(
            _record(group_id="G04", clause_text="不得用于修建楼堂馆所")) is True

    def test_g01g11_without_negative_word_not_carrier(self) -> None:
        # 承接列举「修建楼堂馆所支出」无禁止性措辞，不进载体（禁止语义由含「不得」的父条覆盖）
        assert is_g01g11_carrier(
            _record(group_id="G04", clause_text="修建楼堂馆所支出")) is False

    def test_outside_g01g11_not_carrier(self) -> None:
        assert is_g01g11_carrier(
            _record(group_id="G17", clause_text="不得化整为零规避招标")) is False

    def test_negative_word_variants(self) -> None:
        for word in ("禁止", "严禁", "不得"):
            assert is_g01g11_carrier(
                _record(group_id="G01", clause_text=f"{word}用于发放工资")) is True


# ---------------------------------------------------------------------------
# 单元：去重（同文保留层级最具体）
# ---------------------------------------------------------------------------

class TestDedup:
    def test_keeps_most_specific_level(self) -> None:
        province = _record(doc_index=1, level="省", doc_no="粤测农〔2099〕1号")
        county = _record(doc_index=2, level="县", doc_no="粤测农〔2099〕2号")
        result = dedup_carriers([province, county])  # 同一条款原文，保留县
        assert len(result) == 1
        assert result[0].level == "县"

    def test_keeps_distinct_texts(self) -> None:
        a = _record(clause_text="不得用于修建楼堂馆所")
        b = _record(clause_text="不得用于弥补预算缺口")
        assert len(dedup_carriers([a, b])) == 2


# ---------------------------------------------------------------------------
# 单元：origin 标记与渲染过滤
# ---------------------------------------------------------------------------

class TestOrigin:
    def test_rule_orchestration_hit_accepts_negative_origin(self) -> None:
        mapped = rule_orchestration_hit(_record(), "[C2]", origin=ORIGIN_NEGATIVE)
        assert mapped["metadata"]["origin"] == ORIGIN_NEGATIVE
        assert mapped["metadata"]["doc_no"] == "粤测农〔2099〕1号"

    def test_default_origin_is_rules(self) -> None:
        assert rule_orchestration_hit(_record(), "[C2]")["metadata"]["origin"] == ORIGIN_RULES

    def test_render_context_blocks_filters_by_origin(self) -> None:
        hits = (
            rule_orchestration_hit(_record(), "[C1]", origin=ORIGIN_RULES),
            rule_orchestration_hit(
                _record(clause_text="不得用于弥补预算缺口", doc_no="粤测农〔2099〕3号"),
                "[C2]", origin=ORIGIN_NEGATIVE),
        )
        blocks = render_context_blocks(hits, origin=ORIGIN_NEGATIVE)
        assert "[C2]" in blocks
        assert "[C1]" not in blocks


# ---------------------------------------------------------------------------
# 编排：载体并入命中列表（合成规则库）
# ---------------------------------------------------------------------------

class TestPlanIncludesCarriers:
    def test_negative_carriers_appended_to_hits(self) -> None:
        store = RuleStore(records_from_dicts([{
            "条款原文逐字": "项目资金不得用于修建楼堂馆所。",
            "出处文号+条款号": "粤测农〔2099〕1号 第四条",
            "条款号": "四", "承接列举": False, "层级": "省",
            "所在文件效力状态": "有效",
            "等价组建议": {"组号": "G04", "组名": "楼堂馆所", "组类型": "锚词组"},
            "负面清单载体文件": True, "溯源_文件序号": 1, "溯源_文号": "粤测农〔2099〕1号",
            "溯源_标题": "测试用办法", "溯源_发布机关": "测试机关", "溯源_发布日期": "2099-01-01",
            "溯源_类别": "资金管理办法", "组裁定": "接受",
        }]))
        matcher = NegRuleMatcher(store)
        components = RealComponents(
            CONFIG_PATH, retriever=FakeRetriever((_hit(),)), matcher=matcher)
        hits = components.retrieve("我村申报产业项目", None)
        carriers = [h for h in hits if h["metadata"].get("origin") == ORIGIN_NEGATIVE]
        assert len(carriers) == 1
        assert carriers[0]["metadata"]["doc_no"] == "粤测农〔2099〕1号"
        assert carriers[0]["chunk_id"].startswith("[C")


# ---------------------------------------------------------------------------
# 载体不进①段生成上下文（P2 回归）
# ---------------------------------------------------------------------------

class TestCarriersExcludedFromGeneration:
    @staticmethod
    def _components() -> RealComponents:
        store = RuleStore(records_from_dicts([{
            "条款原文逐字": "项目资金不得用于修建楼堂馆所。",
            "出处文号+条款号": "粤测农〔2099〕1号 第四条",
            "条款号": "四", "承接列举": False, "层级": "省",
            "所在文件效力状态": "有效",
            "等价组建议": {"组号": "G04", "组名": "楼堂馆所", "组类型": "锚词组"},
            "负面清单载体文件": True, "溯源_文件序号": 1, "溯源_文号": "粤测农〔2099〕1号",
            "溯源_标题": "测试用办法", "溯源_发布机关": "测试机关", "溯源_发布日期": "2099-01-01",
            "溯源_类别": "资金管理办法", "组裁定": "接受",
        }]))
        return RealComponents(
            CONFIG_PATH, retriever=FakeRetriever((_hit(),)), matcher=NegRuleMatcher(store))

    def test_carrier_markers_absent_from_context_blocks(self) -> None:
        components = self._components()
        description = "我村申报产业项目"
        hits = components.retrieve(description, None)
        carrier_markers = {
            h["chunk_id"] for h in hits if h["metadata"].get("origin") == ORIGIN_NEGATIVE}
        assert carrier_markers, "夹具应产出至少一个 G01-G11 载体"
        blocks = components.context_blocks(description)
        for marker in carrier_markers:
            assert marker not in blocks, f"载体 {marker} 混进了生成上下文（应只服务②段复核）"

    def test_retrieval_and_rule_blocks_still_in_context(self) -> None:
        components = self._components()
        description = "我村申报产业项目"
        hits = components.retrieve(description, None)
        keep = {h["chunk_id"] for h in hits
                if h["metadata"].get("origin") in (ORIGIN_RETRIEVAL, ORIGIN_RULES)}
        blocks = components.context_blocks(description)
        for marker in keep:
            assert marker in blocks, f"{marker} 不应被过滤掉"


# ---------------------------------------------------------------------------
# 集成冒烟：真实数据仓的载体数量（无数据仓 skip）
# ---------------------------------------------------------------------------

def _data_ready() -> bool:
    if not CONFIG_PATH.is_file():
        return False
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_repo = Path(str(payload.get("data_repo") or ""))
    return (data_repo / "00_工作区" / "负面清单_条款归并.json").is_file()


@pytest.mark.skipif(not _data_ready(), reason="本机无数据仓，跳过真实载体数量冒烟")
def test_real_negative_carrier_count() -> None:
    """真实 G01-G11 载体：去重后应在 ~11-30 项区间（决策1口径）。"""
    from app.real_components import _build_matcher

    matcher = _build_matcher(CONFIG_PATH)
    carriers = dedup_carriers(
        [r for r in matcher.store.valid_records() if is_g01g11_carrier(r)])
    assert 10 <= len(carriers) <= 35
    for c in carriers:
        assert c.group_id in {f"G{i:02d}" for i in range(1, 12)}
        assert any(w in c.clause_text for w in ("不得", "禁止", "严禁"))
