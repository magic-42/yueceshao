"""tests/test_gen_prompts.py — feat/gen-prompts 单元测试（手册 §3.3 PR1）。

覆盖：四段式报告/三段 LLM 输出结构校验（置信纪律、标记格式、禁语文号防线、
枚举、诚实声明互斥）、parse_* JSON 解析（含围栏剥离与失败定位）、提示词模板
红线（temperature=0、禁文号/禁结论指令在位、占位符渲染、模板零真实文号）。

红线遵守：测试只用 [C1]/[C2] 人造标记与人造文本，不使用真实政策文号与条款原文。
"""

from __future__ import annotations

import json

import pytest

from gen.prompts import (
    CONTEXT_BLOCK_FORMAT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_TEMPERATURE,
    GENERATION_USER_TEMPLATE,
    NEGATIVE_ENTAILMENT_SYSTEM_PROMPT,
    NEGATIVE_ENTAILMENT_USER_TEMPLATE,
    PROMPT_CONTRACT,
    render_prompt,
)
from gen.schema import (
    CONFIDENCE_EXPLICIT,
    DISCLAIMER_TEXT,
    HIT_RULE,
    HIT_SUSPECT,
    EntailmentVerdict,
    GeneratedSections,
    MissingItem,
    NegativeHit,
    PolicyItem,
    QueryExtraction,
    SchemaError,
    SuggestionItem,
    parse_entailments,
    parse_extraction,
    parse_generated,
)

ALLOWED = frozenset({"[C1]", "[C2]", "[C3]"})


# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    def test_temperature_is_zero(self) -> None:
        """评测纪律：全部生成调用 temperature=0。"""
        assert GENERATION_TEMPERATURE == 0.0

    @pytest.mark.parametrize("prompt", [
        EXTRACTION_SYSTEM_PROMPT,
        NEGATIVE_ENTAILMENT_SYSTEM_PROMPT,
        GENERATION_SYSTEM_PROMPT,
    ], ids=["抽取", "蕴含复核", "生成"])
    def test_three_system_prompts_carry_redline_rules(self, prompt: str) -> None:
        """三段系统提示词都带：只输出 JSON、禁文号指令、禁结论性措辞指令。"""
        assert "JSON" in prompt
        assert "文号" in prompt and "禁止" in prompt
        assert "可以申报" in prompt  # 以禁令句式出现

    def test_generation_prompt_confidence_rule(self) -> None:
        """生成提示词明写：不输出 confidence（LLM 不自评置信度）。"""
        assert "confidence" in GENERATION_SYSTEM_PROMPT
        assert "确定性信号" in GENERATION_SYSTEM_PROMPT
        assert PROMPT_CONTRACT["confidence_rule"]

    def test_templates_have_no_real_doc_numbers(self) -> None:
        """模板与格式约定零真实文号（示例只用人造标记）。"""
        import re
        pattern = re.compile(r"〔\d{4}〕\s*\d+\s*号")
        for text in (EXTRACTION_SYSTEM_PROMPT, NEGATIVE_ENTAILMENT_SYSTEM_PROMPT,
                     GENERATION_SYSTEM_PROMPT, CONTEXT_BLOCK_FORMAT):
            assert not pattern.search(text)

    def test_render_prompt_substitutes_and_keeps_braces(self) -> None:
        """$ 占位渲染成功；JSON 示例的花括号不被破坏。"""
        out = render_prompt(GENERATION_USER_TEMPLATE,
                            description="某村人居环境整治项目",
                            extraction_json='{"subject_type": "其他"}',
                            context_blocks="[C1]｜现行有效\n原文（逐字）：示例文本")
        assert "某村人居环境整治项目" in out
        assert '{"subject_type": "其他"}' in out
        assert "$" not in out

    def test_render_prompt_missing_placeholder_raises(self) -> None:
        with pytest.raises(ValueError, match="缺占位符"):
            render_prompt(
                GENERATION_USER_TEMPLATE,
                description="只给了 description，缺另外两个占位符")

    def test_entailment_and_extraction_templates_render(self) -> None:
        assert "[C1]" in render_prompt(
            NEGATIVE_ENTAILMENT_USER_TEMPLATE,
            description="拟将部分资金用于其他开支",
            clauses="[C1]｜现行有效\n原文（逐字）：资金不得用于……")
        assert "建厕所" in render_prompt(
            EXTRACTION_USER_TEMPLATE, description="我村想建厕所和买吸污车")


# ---------------------------------------------------------------------------
# 结构校验
# ---------------------------------------------------------------------------

class TestQueryExtraction:
    def test_valid(self) -> None:
        q = QueryExtraction(
            "村集体经济组织", ("公厕新建",), ("设备购置",),
            ("建厕所 → 农村人居环境整治",))
        assert q.validate() is q

    def test_bad_subject_type(self) -> None:
        with pytest.raises(SchemaError, match="主体类型非法"):
            QueryExtraction("股份公司", ("x",), (), ()).validate()

    def test_both_item_lists_empty_rejected(self) -> None:
        with pytest.raises(SchemaError, match="不能同时为空"):
            QueryExtraction("其他", (), (), ()).validate()

    def test_item_with_doc_number_rejected(self) -> None:
        with pytest.raises(SchemaError, match="文号"):
            QueryExtraction("其他", ("依据某〔2099〕1号建设",), (), ()).validate()


class TestEntailmentVerdict:
    def test_valid(self) -> None:
        v = EntailmentVerdict("[C1]", "蕴含", "描述拟将资金用于其他开支，落入该条款禁止范围")
        assert v.validate(ALLOWED) is v

    @pytest.mark.parametrize("verdict", ["是", "违规", "maybe"])
    def test_bad_verdict(self, verdict: str) -> None:
        with pytest.raises(SchemaError, match="蕴含结论非法"):
            EntailmentVerdict("[C1]", verdict, "依据").validate(ALLOWED)

    def test_reason_with_conclusive_word_rejected(self) -> None:
        with pytest.raises(SchemaError, match="姿态禁语"):
            EntailmentVerdict("[C1]", "蕴含", "该项目违规使用资金").validate(ALLOWED)

    def test_citation_not_in_allowed_set(self) -> None:
        with pytest.raises(SchemaError, match="不在本次提供的上下文"):
            EntailmentVerdict("[C9]", "不确定", "依据不足").validate(ALLOWED)


class TestGeneratedSections:
    def test_valid_full(self) -> None:
        g = GeneratedSections(
            policy_items=(PolicyItem("[C1]", "该条款与项目的公厕建设相关", None),),
            missing_items=(MissingItem("[C2]", "项目申报书", "按指南建议核对"),),
            suggestions=(SuggestionItem("建议在描述中写明公厕数量与点位"),),
        )
        assert g.validate(ALLOWED) is g

    def test_invalid_confidence_label_rejected(self) -> None:
        """非法置信值拒绝（LLM 自评置信的拦截在 parse_generated 置信纪律闸，另有用例）。"""
        with pytest.raises(SchemaError, match="置信标签非法"):
            PolicyItem("[C1]", "说明", "高置信").validate(ALLOWED)

    def test_summary_with_doc_number_rejected(self) -> None:
        with pytest.raises(SchemaError, match="文号"):
            PolicyItem("[C1]", "依据某〔2099〕99号的条款", None).validate(ALLOWED)

    def test_bad_citation_format(self) -> None:
        with pytest.raises(SchemaError, match="格式非法"):
            PolicyItem("C1", "说明", None).validate(ALLOWED)

    def test_uncovered_declared_with_items_rejected(self) -> None:
        g = GeneratedSections(
            policy_items=(PolicyItem("[C1]", "说明", None),),
            missing_items=(),
            suggestions=(),
            uncovered_sections=("①适用政策",),
        )
        with pytest.raises(SchemaError, match="不得同时给出"):
            g.validate(ALLOWED)

    def test_empty_without_uncovered_rejected(self) -> None:
        """A11 诚实纪律：既无条目也未声明未覆盖＝沉默失败，拒绝。"""
        g = GeneratedSections(policy_items=(), missing_items=(), suggestions=())
        with pytest.raises(SchemaError, match="不知道要说不知道"):
            g.validate(ALLOWED)

    def test_uncovered_only_is_valid(self) -> None:
        g = GeneratedSections(
            policy_items=(), missing_items=(), suggestions=(),
            uncovered_sections=("①适用政策", "③材料缺项"),
        )
        assert g.validate(ALLOWED) is g

    def test_bad_uncovered_name_rejected(self) -> None:
        g = GeneratedSections(
            policy_items=(PolicyItem("[C1]", "说明", None),),
            missing_items=(MissingItem("[C2]", "材料", "依据"),),
            suggestions=(),
            uncovered_sections=("②负面命中",),
        )
        with pytest.raises(SchemaError, match="未覆盖段声明非法"):
            g.validate(ALLOWED)


class TestReportStructures:
    def test_negative_hit_enums(self) -> None:
        hit = NegativeHit("[C3]", HIT_SUSPECT, (("补助", "补助"),))
        assert hit.validate(ALLOWED) is hit
        with pytest.raises(SchemaError, match="命中类型非法"):
            NegativeHit("[C3]", "命中", ()).validate(ALLOWED)

    def test_final_report_requires_system_confidence(self) -> None:
        """最终报告的①段必须带系统侧置信（渲染层组装后的状态）。"""
        from gen.schema import CheckupReport
        ok = CheckupReport(
            project_summary="某村人居环境整治项目",
            policy_items=(PolicyItem("[C1]", "该条款与公厕建设相关", CONFIDENCE_EXPLICIT),),
            negative_hits=(NegativeHit("[C2]", HIT_RULE, (("债务", "偿还债务"),)),),
            missing_items=(MissingItem("[C3]", "项目申报书", "按指南建议核对"),),
            suggestions=(SuggestionItem("建议写明建设点位"),),
        )
        assert ok.validate(ALLOWED) is ok
        bad = CheckupReport(
            project_summary="某村项目",
            policy_items=(PolicyItem("[C1]", "说明", None),),
            negative_hits=(), missing_items=(MissingItem("[C2]", "材料", "依据"),),
            suggestions=(),
        )
        with pytest.raises(SchemaError, match="系统侧置信标签"):
            bad.validate(ALLOWED)

    def test_disclaimer_anchor_text(self) -> None:
        """免责声明锚点＝README 产品级原文（渲染层强制注入用）。"""
        assert "不构成合规判定" in DISCLAIMER_TEXT


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_extraction_roundtrip(self) -> None:
        raw = json.dumps({
            "subject_type": "强村公司",
            "construction_items": ["产业园道路配套"],
            "use_items": ["园区管网建设"],
            "terms": ["修路 → 道路基础设施"],
        }, ensure_ascii=False)
        q = parse_extraction(raw)
        assert q.subject_type == "强村公司" and q.use_items == ("园区管网建设",)

    def test_parse_extraction_strips_fences(self) -> None:
        raw = ("```json\n{\"subject_type\": \"其他\", \"construction_items\": [\"建公厕\"],"
               " \"use_items\": [], \"terms\": []}\n```")
        assert parse_extraction(raw).construction_items == ("建公厕",)

    def test_parse_extraction_bad_json_raises(self) -> None:
        with pytest.raises(SchemaError, match="JSON 解析失败"):
            parse_extraction("这不是JSON")

    def test_parse_entailments_roundtrip(self) -> None:
        raw = json.dumps([
            {"citation": "[C1]", "verdict": "不蕴含",
             "reason": "描述为建设内容，与该条款禁止的用途无关"},
            {"citation": "[C2]", "verdict": "不确定", "reason": "描述未说明资金去向"},
        ], ensure_ascii=False)
        verdicts = parse_entailments(raw, ALLOWED)
        assert len(verdicts) == 2 and verdicts[1].verdict == "不确定"

    def test_parse_entailments_top_level_object_rejected(self) -> None:
        with pytest.raises(SchemaError, match="顶层需要 JSON 数组"):
            parse_entailments('{"a": 1}')

    def test_parse_generated_roundtrip(self) -> None:
        raw = json.dumps({
            "policy_items": [{"citation": "[C1]", "summary": "与公厕建设相关", "confidence": None}],
            "missing_items": [{"citation": "[C2]", "name": "项目申报书", "note": "按指南核对"}],
            "suggestions": [{"text": "建议写明点位"}],
            "uncovered_sections": [],
        }, ensure_ascii=False)
        g = parse_generated(raw, ALLOWED)
        assert g.policy_items[0].citation == "[C1]"

    def test_parse_generated_llm_confidence_rejected(self) -> None:
        """置信纪律闸：LLM 自评置信在解析口拦下。"""
        raw = json.dumps({
            "policy_items": [{"citation": "[C1]", "summary": "相关", "confidence": "明确适用"}],
            "missing_items": [{"citation": "[C2]", "name": "材料", "note": "依据"}],
            "suggestions": [],
        }, ensure_ascii=False)
        with pytest.raises(SchemaError, match="不自评置信度"):
            parse_generated(raw, ALLOWED)

    def test_parse_generated_conclusive_word_rejected(self) -> None:
        raw = json.dumps({
            "policy_items": [{"citation": "[C1]", "summary": "该项目可以申报"}],
            "missing_items": [{"citation": "[C2]", "name": "材料", "note": "依据"}],
            "suggestions": [],
        }, ensure_ascii=False)
        with pytest.raises(SchemaError, match="姿态禁语"):
            parse_generated(raw, ALLOWED)
