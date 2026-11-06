"""tests/test_gen_citation.py — feat/gen-citation 单元测试（手册 §3.3 PR2）。

覆盖：元数据视图（鸭子类型/占位文号判据/缺失字段降级）、引用解析与退化格式、
置信填充三档（确定性信号）、组装防线（失效剔除/标记缺失丢弃/LLM 文本红线）、
文本渲染（免责声明强制附加/诚实降级占位/渲染成品不含元数据表之外的文号）。

红线遵守：夹具只用〔2099〕人造文号与自造标题，不使用真实政策语料原文。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from gen.citation import (
    ChunkMeta,
    ChunkMetaImportError,
    build_meta_table,
    build_report,
    format_reference,
    render_hit_line,
    render_report,
    resolve_citation,
)
from gen.schema import (
    CONFIDENCE_EXPLICIT,
    CONFIDENCE_SUGGEST,
    DISCLAIMER_TEXT,
    HIT_RULE,
    HIT_SUSPECT,
    GeneratedSections,
    MissingItem,
    NegativeHit,
    PolicyItem,
    SuggestionItem,
)

# ---- 人造元数据（文号均为〔2099〕系列，杜绝真实文号）----

def _meta(cid="[C1]", **over: Any) -> ChunkMeta:
    base = dict(
        citation_id=cid,
        doc_no="粤测农〔2099〕1号",
        clause_no="第四条",
        effective_date="2099-01-01",
        validity="有效",
        level="省",
        title="测试用涉农资金管理办法",
        is_current=True,
    )
    base.update(over)
    return ChunkMeta(**base)


TABLE = {
    "[C1]": _meta("[C1]"),
    "[C2]": _meta("[C2]", doc_no="无（申报通知未载文号）", clause_no="第三条",
                  title="测试用申报指南", validity="有效"),
    "[C3]": _meta("[C3]", validity="待人工核验", is_current=True),
    "[C4]": _meta("[C4]", validity="已失效", is_current=False),
}


@dataclass
class FakeHit:
    """模拟检索侧命中（鸭子类型目标：只要有 chunk_id/metadata）。"""

    chunk_id: str
    metadata: dict


# ---------------------------------------------------------------------------
# 元数据视图
# ---------------------------------------------------------------------------

class TestChunkMeta:
    def test_from_hit_duck_typing(self) -> None:
        hit = FakeHit("[C1]", {"doc_no": "粤测农〔2099〕1号", "clause_no": "第四条",
                                "validity": "有效", "level": "省", "title": "测试办法",
                                "is_current": True})
        meta = ChunkMeta.from_hit(hit)
        assert meta.has_doc_no and meta.is_current

    def test_from_hit_rejects_missing_fields(self) -> None:
        with pytest.raises(ChunkMetaImportError):
            ChunkMeta.from_hit(FakeHit("", {}))
        with pytest.raises(ChunkMetaImportError):
            ChunkMeta.from_hit(FakeHit("[C1]", "不是映射"))

    def test_placeholder_doc_no_is_not_real(self) -> None:
        """占位串（非空但非文号）不得被判为有真文号。"""
        assert not TABLE["[C2]"].has_doc_no
        assert not _meta(doc_no="无（页面未载/仅正文引用，不计）").has_doc_no

    def test_is_current_fallback_to_validity(self) -> None:
        """缺 is_current 字段时按效力推断（保守）。"""
        m = ChunkMeta.from_mapping(
            {"doc_no": "粤测农〔2099〕1号", "validity": "有效"}, citation_id="[C9]")
        assert m.is_current
        m2 = ChunkMeta.from_mapping(
            {"doc_no": "粤测农〔2099〕1号", "validity": "已失效"}, citation_id="[C8]")
        assert not m2.is_current

    def test_build_table_rejects_mapping_without_id(self) -> None:
        with pytest.raises(ChunkMetaImportError):
            build_meta_table([{"doc_no": "粤测农〔2099〕1号"}])


# ---------------------------------------------------------------------------
# 引用解析与格式
# ---------------------------------------------------------------------------

class TestFormatReference:
    def test_real_doc_no_format(self) -> None:
        s = format_reference(TABLE["[C1]"])
        assert s == "粤测农〔2099〕1号 第四条（省，有效）"

    def test_placeholder_degrades_to_title(self) -> None:
        """无真文号 → 退化用 registry 标题（权威字段，非编造）。"""
        s = format_reference(TABLE["[C2]"])
        assert s.startswith("《测试用申报指南》") and "第三条" in s
        assert "无（" not in s  # 占位串永不进成品

    def test_resolve_miss_returns_none(self) -> None:
        assert resolve_citation("[C99]", TABLE) is None


# ---------------------------------------------------------------------------
# 报告组装（置信/剔除/防线）
# ---------------------------------------------------------------------------

def _sections(policy=(), missing=(), suggestions=(), uncovered=()) -> GeneratedSections:
    return GeneratedSections(
        policy_items=tuple(policy),
        missing_items=tuple(missing),
        suggestions=tuple(suggestions),
        uncovered_sections=tuple(uncovered),
    )


class TestBuildReport:
    def test_confidence_from_deterministic_signals(self) -> None:
        """置信三档由时效/命中类型驱动：有效→明确适用；待核验→建议确认。"""
        report = build_report(
            "某村测试项目",
            _sections(policy=(PolicyItem("[C1]", "与公厕建设相关", None),
                              PolicyItem("[C3]", "待核验条款", None)),
                      missing=(MissingItem("[C2]", "项目申报书", "按指南核对"),),
                      suggestions=(SuggestionItem("建议写明点位"),)),
            negative_hits=(),
            table=TABLE,
        )
        assert report.policy_items[0].confidence == CONFIDENCE_EXPLICIT
        assert report.policy_items[1].confidence == CONFIDENCE_SUGGEST
        assert report.missing_items[0].name == "项目申报书"

    def test_invalid_clause_dropped_and_section_uncovered(self) -> None:
        """已失效条款不进①段（渲染层双保险）；剔空转未覆盖。"""
        report = build_report(
            "某村测试项目",
            _sections(policy=(PolicyItem("[C4]", "失效条款", None),)),
            negative_hits=(),
            table=TABLE,
        )
        assert report.policy_items == ()
        assert "①适用政策" in report.uncovered_sections

    def test_unknown_citation_dropped(self) -> None:
        """标记查不到 → 条目丢弃（渲染层诚实降级，不猜不补）。"""
        report = build_report(
            "某村测试项目",
            _sections(policy=(PolicyItem("[C99]", "查无此标记", None),)),
            negative_hits=(),
            table=TABLE,
        )
        assert report.policy_items == ()

    def test_llm_text_with_doc_number_rejected(self) -> None:
        """红线防线：绕过 parse 直接构造、summary 夹带文号 → 渲染入口拒绝。"""
        with pytest.raises(Exception, match="文号"):
            build_report(
                "某村测试项目",
                _sections(policy=(PolicyItem("[C1]", "依据粤测农〔2099〕77号", None),)),
                negative_hits=(),
                table=TABLE,
            )

    def test_llm_text_with_posture_word_rejected(self) -> None:
        with pytest.raises(Exception, match="禁语|合规|可以申报"):
            build_report(
                "某村测试项目",
                _sections(suggestions=(SuggestionItem("该项目可以申报"),)),
                negative_hits=(),
                table=TABLE,
            )

    def test_negative_hits_pass_through_with_type(self) -> None:
        hits = (NegativeHit("[C1]", HIT_RULE, (("债务", "偿还债务"),)),
                NegativeHit("[C2]", HIT_SUSPECT, ()))
        report = build_report("某村测试项目",
                              _sections(policy=(PolicyItem("[C1]", "相关", None),),
                                        missing=(MissingItem("[C2]", "材料", "依据"),)),
                              negative_hits=hits, table=TABLE)
        assert report.negative_hits == hits


# ---------------------------------------------------------------------------
# 文本渲染
# ---------------------------------------------------------------------------

class TestRenderReport:
    def _full_report(self):
        return build_report(
            "某村人居环境整治项目",
            _sections(policy=(PolicyItem("[C1]", "该条款与项目的公厕建设内容相关", None),),
                      missing=(MissingItem("[C2]", "项目申报书", "按指南建议核对"),),
                      suggestions=(SuggestionItem("建议在描述中写明公厕数量与点位"),)),
            negative_hits=(NegativeHit("[C1]", HIT_RULE, (("债务", "偿还债务"),)),),
            table=TABLE,
        )

    def test_disclaimer_always_appended(self) -> None:
        """免责声明由渲染层强制附加（唯一出口），任何报告都以它结尾。"""
        text = render_report(self._full_report(), TABLE)
        assert text.rstrip().endswith(f"免责声明：{DISCLAIMER_TEXT}")

    def test_citation_resolved_into_reference(self) -> None:
        text = render_report(self._full_report(), TABLE)
        assert "粤测农〔2099〕1号 第四条（省，有效）" in text
        assert "[C1]" not in text  # 成品不残留原始标记
        assert "[C2]" not in text

    def test_unknown_citation_renders_honest_placeholder(self) -> None:
        """渲染层遇到表外标记 → 诚实占位，绝不编造文号。"""
        report = build_report(
            "某项目",
            _sections(policy=(PolicyItem("[C1]", "相关", None),),
                      missing=(MissingItem("[C2]", "材料", "依据"),)),
            negative_hits=(NegativeHit("[C77]", HIT_SUSPECT, ()),),
            table=TABLE,
        )
        text = render_report(report, TABLE)
        assert "依据标注缺失" in text and "[C77]" in text
        # 成品中不得出现元数据表之外的任何文号
        assert "粤测农〔2099〕2号" not in text

    def test_hit_line_shows_type_and_words(self) -> None:
        line = render_hit_line(NegativeHit("[C1]", HIT_RULE, (("债务", "偿还债务"),)), TABLE)
        assert HIT_RULE in line and "债务」↔「偿还债务" in line

    def test_uncovered_section_declared(self) -> None:
        report = build_report(
            "某项目",
            _sections(policy=(PolicyItem("[C1]", "相关", None),),
                      missing=(MissingItem("[C2]", "材料", "依据"),),
                      uncovered=("④修改建议",)),
            negative_hits=(),
            table=TABLE,
        )
        text = render_report(report, TABLE)
        assert "④修改建议" in text and "未能可靠作答" in text
