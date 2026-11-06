"""tests/test_rules_engine.py — feat/rules-engine 单元测试（TDD 先红后绿）。

夹具：tests/fixtures/rules_sample.json（10 条人造样例规则，含人造等价组）。
红线遵守：不使用真实政策语料原文、不引用真实文号。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rules.engine import MatchResult, NegRuleMatcher, render_hits
from rules.loader import load_records, records_from_dicts
from rules.schema import RuleStore

FIXTURE = Path(__file__).parent / "fixtures" / "rules_sample.json"


@pytest.fixture(scope="module")
def matcher() -> NegRuleMatcher:
    return NegRuleMatcher(RuleStore(load_records(FIXTURE)))


def test_concept_cooccurrence_hit(matcher: NegRuleMatcher) -> None:
    """描述侧术语与条款侧术语同概念组即命中，命中词可解释。"""
    result = matcher.match("拟将项目资金用于测试性楼堂馆所建设")
    assert len(result.hits) == 3
    hit = result.hits[0]
    assert "楼堂馆所" in hit.matched_terms
    assert "楼堂馆所" in hit.matched_description_terms
    assert "楼堂馆所" in hit.concept


def test_synonym_cooccurrence_hit(matcher: NegRuleMatcher) -> None:
    """同义术语跨词命中：描述「公务用车」↔ 条款「交通工具」（同一概念组）。"""
    result = matcher.match("申报人拟购置公务用车2辆用于项目通勤")
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.rule.doc_no == "〔测试样例〕市D-010号"
    assert "公务用车" in hit.matched_description_terms
    assert "交通工具" in hit.matched_terms


def test_no_terms_no_hit(matcher: NegRuleMatcher) -> None:
    """无禁止用途词的申报场景不产生命中（引擎是提示器，不是判定器）。"""
    assert matcher.match("建设高标准农田1500亩，衬砌渠道6.2公里").hits == ()
    assert matcher.match("").hits == ()


def test_term_present_but_no_matching_rule(matcher: NegRuleMatcher) -> None:
    """描述含禁止用途词、但无任何现行规则同概念组 → 零命中（有词≠有命中）。"""
    result = matcher.match("项目拟偿还银行贷款本息")
    assert result.hits == ()
    assert "贷款" in result.description_terms


def test_structure_annotation_group_excluded() -> None:
    """结构标注组（总括句）不参与匹配，结构噪音不进命中提示。"""
    record = {
        "条款原文逐字":
            "测试样例条款：帮扶资金不得用于以下测试性禁止用途："
            "楼堂馆所建设（人造样例，非真实政策原文）。",
        "出处文号+条款号": "〔测试样例〕省E-011号 第一条",
        "条款号": "一",
        "承接列举": False,
        "层级": "省",
        "所在文件效力状态": "有效",
        "等价组建议": {"组号": "测试总括句", "组名": "总括句", "组类型": "结构标注"},
        "负面清单载体文件": True,
        "溯源_文件序号": 11,
        "溯源_文号": "〔测试样例〕省E-011号",
        "溯源_标题": "测试样例文件十一（人造数据）",
        "溯源_发布机关": "测试样例机关",
        "溯源_发布日期": "2024-11-01",
        "溯源_类别": "测试样例类别",
        "组裁定": "接受（人造等价组）",
    }
    matcher = NegRuleMatcher(RuleStore(records_from_dicts([record])))
    result = matcher.match("拟将项目资金用于测试性楼堂馆所建设")
    assert result.hits == ()
    assert "楼堂馆所" in result.description_terms


def test_invalidated_rule_excluded(matcher: NegRuleMatcher) -> None:
    """已失效条款不参与匹配（描述侧识别到术语也不命中）。"""
    result = matcher.match("拟安排发放工资及人员经费")
    assert result.hits == ()
    assert "工资" in result.description_terms


def test_hits_sorted_by_display_priority(matcher: NegRuleMatcher) -> None:
    """命中排序：现行有效优先，同效力时层级更具体优先（县>市>省>国家）。"""
    result = matcher.match("拟将项目资金用于测试性楼堂馆所建设")
    assert [h.rule.source_ref for h in result.hits] == [
        "〔测试样例〕省A-002号 第二条",   # 有效（省）
        "〔测试样例〕县A-009号 第三条",   # 待人工核验（县含乡镇，层级优先于市）
        "〔测试样例〕市A-003号 第三条",   # 待人工核验（市）
    ]


def test_hit_includes_equivalence_peers(matcher: NegRuleMatcher) -> None:
    """命中输出带等价组对照（同组其他层级条款）。"""
    hit = matcher.match("拟将项目资金用于测试性楼堂馆所建设").hits[0]
    assert {r.level for r in hit.group_peers} == {"国家", "市", "县", "县（含乡镇）"}
    assert all(r is not hit.rule for r in hit.group_peers)


def test_render_wording_discipline(matcher: NegRuleMatcher) -> None:
    """输出文案只作命中提示/建议对照原文，禁出现判定性措辞（姿态红线）。"""
    lines = render_hits(matcher.match("拟将项目资金用于测试性楼堂馆所建设"))
    text = "\n".join(lines)
    assert "命中提示" in text and "建议对照原文" in text
    for banned in ("违规", "不合规", "可以申报", "合规"):
        assert banned not in text
    assert "〔测试样例〕省A-002号二（省，有效）" in text  # 命中条款带层级与效力标注
    assert "测试性楼堂馆所" in text        # 条款原文逐字展示


def test_render_empty_result() -> None:
    """无命中时给出中性提示（不判定）。"""
    assert render_hits(
        MatchResult(hits=(), description_terms=())
    ) == ["未识别到禁止用途词，暂无命中提示。"]
