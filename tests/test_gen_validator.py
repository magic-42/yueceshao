"""tests/test_gen_validator.py — feat/gen-validator 单元测试（手册 §3.3 PR3，A9）。

本模块是四个「先红后绿」关键路径之一（引用校验器＝幻觉最后防线）：
测试先于实现编写（TDD），覆盖手册验收标准——标记缺失/禁语/包含性/免责声明
四类检查各有用例，降级路径可达（重试≤2→显式降级标注）。

红线遵守：夹具只用〔2099〕人造文号与自造条款原文，不使用真实政策语料。
"""

from __future__ import annotations

import json

import pytest

from gen.schema import (
    DISCLAIMER_TEXT,
    GeneratedSections,
    MissingItem,
    PolicyItem,
    SuggestionItem,
)
from gen.validator import (
    DEGRADED_NOTICE,
    RetryOutcome,
    apply_degradation,
    check_citation_existence,
    check_containment,
    check_disclaimer,
    check_posture,
    run_with_retry,
    validate_all,
)

# ---- 夹具：chunk 原文（自造，与「引语」用例逐字对齐）----
CHUNK_TEXTS = {
    "[C1]": "帮扶资金主要用于产业道路、灌溉设施等生产性基础设施建设，不得用于楼堂馆所建设。",
    "[C2]": "申报主体应当提交项目申报书、实施方案和村集体经济组织成员大会决议材料。",
}
ALLOWED = frozenset(CHUNK_TEXTS)


def _sections(policy=(), missing=(), suggestions=(), uncovered=()) -> GeneratedSections:
    return GeneratedSections(
        policy_items=tuple(policy),
        missing_items=tuple(missing),
        suggestions=tuple(suggestions),
        uncovered_sections=tuple(uncovered),
    )


# ---------------------------------------------------------------------------
# 检查一：标记存在性（LLM 引用的标记必须在上下文集合内）
# ---------------------------------------------------------------------------

class TestCitationExistence:
    def test_all_in_allowed_passes(self) -> None:
        s = _sections(policy=(PolicyItem("[C1]", "相关", None),),
                      missing=(MissingItem("[C2]", "材料", "依据"),))
        assert check_citation_existence(s, ALLOWED) == ()

    def test_unknown_mark_flagged(self) -> None:
        s = _sections(policy=(PolicyItem("[C9]", "相关", None),),
                      missing=(MissingItem("[C2]", "材料", "依据"),))
        issues = check_citation_existence(s, ALLOWED)
        assert len(issues) == 1
        assert issues[0].check == "标记存在性" and "[C9]" in issues[0].message
        assert issues[0].section == "①适用政策"

    def test_malformed_mark_flagged(self) -> None:
        s = _sections(missing=(MissingItem("C2", "材料", "依据"),))
        issues = check_citation_existence(s, ALLOWED)
        assert issues and issues[0].check == "标记存在性" and issues[0].section == "③材料缺项"


# ---------------------------------------------------------------------------
# 检查二：包含性（成句引语必须逐字出现在所引 chunk 原文中）
# ---------------------------------------------------------------------------

class TestContainment:
    def test_verbatim_quote_passes(self) -> None:
        s = _sections(policy=(PolicyItem(
            "[C1]", "条款明确帮扶资金「不得用于楼堂馆所建设」，与本项目无关", None),))
        assert check_containment(s, CHUNK_TEXTS) == ()

    def test_fabricated_quote_flagged(self) -> None:
        """编造引语（原文没有的「成句引文」）＝引用失真，必须拦下。"""
        s = _sections(policy=(
            PolicyItem("[C1]", "条款规定「帮扶资金可以用于建房」，需注意", None),))
        issues = check_containment(s, CHUNK_TEXTS)
        assert len(issues) == 1 and issues[0].check == "包含性"

    def test_quote_from_wrong_chunk_flagged(self) -> None:
        """引语真实存在但引错了条款（[C2] 的原文放到 [C1] 名下）。"""
        s = _sections(missing=(MissingItem("[C1]", "材料",
                                                  "指南要求提交「项目申报书、实施方案」"),))
        issues = check_containment(s, CHUNK_TEXTS)
        assert len(issues) == 1 and issues[0].check == "包含性"

    def test_short_phrase_not_treated_as_quote(self) -> None:
        """短于 8 字的「」短语视为普通强调，不进引语核验（防误报）。"""
        s = _sections(policy=(PolicyItem("[C1]", "这就是所谓的「生产性基础设施」类别", None),))
        assert check_containment(s, CHUNK_TEXTS) == ()

    def test_unresolvable_chunk_flagged(self) -> None:
        """引语所引标记无原文映射 → 无法核验＝issue（宁严勿漏）。"""
        s = _sections(policy=(PolicyItem("[C9]", "原文说「帮扶资金主要用于产业道路建设」", None),))
        issues = check_containment(s, CHUNK_TEXTS)
        assert issues and issues[0].check == "包含性"

    def test_no_quote_passes(self) -> None:
        s = _sections(suggestions=(SuggestionItem("建议写明建设点位与规模"),))
        assert check_containment(s, CHUNK_TEXTS) == ()


# ---------------------------------------------------------------------------
# 检查三：姿态禁语（LLM 字段独立重扫，不信任上游）
# ---------------------------------------------------------------------------

class TestPosture:
    @pytest.mark.parametrize(
        "text", ["该项目合规", "本项目可以申报", "此做法违规", "属于不合规行为"])
    def test_forbidden_words_flagged(self, text: str) -> None:
        s = _sections(suggestions=(SuggestionItem(text),))
        issues = check_posture(s)
        assert len(issues) == 1 and issues[0].check == "姿态禁语"

    def test_clean_passes(self) -> None:
        s = _sections(policy=(PolicyItem("[C1]", "与项目的灌溉设施建设相关", None),),
                      suggestions=(SuggestionItem("建议核对条款原文"),))
        assert check_posture(s) == ()


# ---------------------------------------------------------------------------
# 检查四：免责声明（渲染成品的最后防线）
# ---------------------------------------------------------------------------

class TestDisclaimer:
    def test_present_passes(self) -> None:
        assert check_disclaimer(f"报告正文……\n免责声明：{DISCLAIMER_TEXT}") == ()

    def test_missing_flagged(self) -> None:
        issues = check_disclaimer("报告正文但没有免责声明")
        assert len(issues) == 1 and issues[0].check == "免责声明"


# ---------------------------------------------------------------------------
# 汇总校验
# ---------------------------------------------------------------------------

class TestValidateAll:
    def test_clean_sections_pass(self) -> None:
        s = _sections(policy=(PolicyItem("[C1]", "与灌溉设施建设相关", None),),
                      missing=(MissingItem("[C2]", "项目申报书", "按指南核对"),),
                      suggestions=(SuggestionItem("建议写明点位"),))
        report = validate_all(s, ALLOWED, CHUNK_TEXTS)
        assert report.passed and report.issues == ()

    def test_issues_carry_section(self) -> None:
        s = _sections(policy=(PolicyItem("[C7]", "相关", None),))
        report = validate_all(s, ALLOWED, CHUNK_TEXTS)
        assert not report.passed
        assert report.issues[0].section == "①适用政策"


# ---------------------------------------------------------------------------
# 重试与降级（A9：重试≤2 → 降级显式标注）
# ---------------------------------------------------------------------------

class TestRunWithRetry:
    def test_first_pass_no_retry(self) -> None:
        good = json.dumps({
            "policy_items": [
                {"citation": "[C1]", "summary": "与灌溉设施建设相关", "confidence": None}],
            "missing_items": [{"citation": "[C2]", "name": "项目申报书", "note": "按指南核对"}],
            "suggestions": [{"text": "建议写明点位"}],
        }, ensure_ascii=False)
        calls = []

        def gen(feedback):
            calls.append(feedback)
            return good

        outcome = run_with_retry(gen, ALLOWED, CHUNK_TEXTS)
        assert isinstance(outcome, RetryOutcome)
        assert outcome.delivered is not None and not outcome.degraded
        assert outcome.attempts == 1 and len(calls) == 1 and calls[0] is None

    def test_retry_then_pass(self) -> None:
        """第一次有禁语，带着失败反馈重试后通过——attempts=2。"""
        bad = json.dumps({"suggestions": [{"text": "该项目合规"}],
                          "policy_items": [], "missing_items": [],
                          "uncovered_sections": ["①适用政策", "③材料缺项"]}, ensure_ascii=False)
        good = json.dumps({"suggestions": [{"text": "建议写明点位"}],
                           "policy_items": [], "missing_items": [],
                           "uncovered_sections": ["①适用政策", "③材料缺项"]}, ensure_ascii=False)
        payloads = [bad, good]

        def gen(feedback):
            assert feedback is None or "姿态禁语" in feedback  # 重试必须携带失败原因
            return payloads.pop(0)

        outcome = run_with_retry(gen, ALLOWED, CHUNK_TEXTS)
        assert not outcome.degraded and outcome.attempts == 2
        assert outcome.delivered.suggestions[0].text == "建议写明点位"

    def test_exhausted_retries_degrade(self) -> None:
        """三次全失败（首次＋重试2）→ 降级：失败段清空＋显式标注。

        坏输出用「编造引语」构造——它能通过 parse 层（schema 不核引语），
        只挂 validator 的包含性检查，从而真测到分段定位与选择性保留：
        ①段失败清空、③段无问题保留。禁语类坏输出在 parse 口即被拦
        （整体失败），由 test_parse_failure_also_retried 覆盖那条路径。
        """
        bad = json.dumps({
            "policy_items": [
                {"citation": "[C1]",
                 "summary": "条款明确「帮扶资金可以用于建造办公楼」，需注意",
                 "confidence": None}],
            "missing_items": [{"citation": "[C2]", "name": "材料", "note": "依据"}],
            "suggestions": [],
        }, ensure_ascii=False)

        outcome = run_with_retry(lambda fb: bad, ALLOWED, CHUNK_TEXTS)
        assert outcome.degraded and outcome.attempts == 3
        assert outcome.failed_sections == ("①适用政策",)  # 精确到失败段
        # 降级段条目清空且声明未覆盖（可进渲染）
        assert outcome.delivered.policy_items == ()
        assert "①适用政策" in outcome.delivered.uncovered_sections
        # 其余通过的段保留（③段无问题应保留）
        assert outcome.delivered.missing_items != ()

    def test_cross_round_fix_not_over_degraded(self) -> None:
        """A1 回归：降级段只取最后一轮失败——跨轮已修复的段不得被历史 issues 牵连清空。

        r1 ①段编造引语（挂）→ r2 ①修好、③段引语编造（挂）→ r3 同 r2 → 耗尽。
        正确行为：只降级③段；①段最后一轮干净，必须保留。
        """
        bad_policy = json.dumps({
            "policy_items": [
                {"citation": "[C1]",
                 "summary": "条款明确「帮扶资金可以用于建造办公楼」，需注意",
                 "confidence": None}],
            "missing_items": [{"citation": "[C2]", "name": "项目申报书", "note": "按指南核对"}],
            "suggestions": [],
        }, ensure_ascii=False)
        bad_missing = json.dumps({
            "policy_items": [{"citation": "[C1]", "summary": "与产业道路建设相关",
                              "confidence": None}],
            "missing_items": [
                {"citation": "[C2]", "name": "项目申报书",
                 "note": "应提交「项目申报书、实施方案和决议材料」"}],
            "suggestions": [],
        }, ensure_ascii=False)
        payloads = iter([bad_policy, bad_missing, bad_missing])
        outcome = run_with_retry(lambda fb: next(payloads), ALLOWED, CHUNK_TEXTS)
        assert outcome.degraded and outcome.attempts == 3
        assert outcome.failed_sections == ("③材料缺项",)
        assert outcome.delivered.policy_items != ()      # ①段跨轮修好，保留
        assert outcome.delivered.missing_items == ()     # ③段最后一轮失败，降级
        assert "③材料缺项" in outcome.delivered.uncovered_sections

    def test_parse_failure_also_retried(self) -> None:
        """连 JSON 都解析失败的输出同样计入重试（校验器不信任任何环节）。"""
        payloads = ["这不是JSON", "也不是", "还不是"]

        def gen(feedback):
            return payloads.pop(0)

        outcome = run_with_retry(gen, ALLOWED, CHUNK_TEXTS)
        assert outcome.degraded and outcome.attempts == 3


class TestApplyDegradation:
    def test_degraded_section_cleared_and_declared(self) -> None:
        s = _sections(policy=(PolicyItem("[C1]", "相关", None),),
                      missing=(MissingItem("[C2]", "材料", "依据"),),
                      suggestions=(SuggestionItem("建议"),))
        out = apply_degradation(s, ("①适用政策",))
        assert out.policy_items == ()
        assert "①适用政策" in out.uncovered_sections
        assert out.missing_items == s.missing_items  # 未失败段保留
        assert out.suggestions == s.suggestions

    def test_section_four_degrades_to_empty(self) -> None:
        """④段无未覆盖枚举值，降级＝清空（渲染层显示「暂无」）。"""
        s = _sections(suggestions=(SuggestionItem("建议"),),
                      uncovered=("①适用政策", "③材料缺项"))
        out = apply_degradation(s, ("④修改建议",))
        assert out.suggestions == ()

    def test_degradation_hardens_incomplete_input(self) -> None:
        """完备性兜底：对残缺输入（①空却未声明）降级后仍产出合法结构。"""
        s = _sections(missing=(MissingItem("[C2]", "材料", "依据"),))
        out = apply_degradation(s, ("④修改建议",))
        out.validate()  # 不抛即合法
        assert "①适用政策" in out.uncovered_sections

    def test_degraded_notice_constant(self) -> None:
        """降级标注文案存在且为诚实口径（渲染层引用）。"""
        assert "完整性校验" in DEGRADED_NOTICE and "仅供参考" in DEGRADED_NOTICE

class TestRenderDegradedAnnotation:
    def test_render_marks_degraded_sections(self) -> None:
        """A9 渲染落点：降级段在成品文本中带显式标注（DEGRADED_NOTICE 口径）。"""
        from gen.citation import ChunkMeta, build_report, render_report
        meta_table = {
            cid: ChunkMeta(citation_id=cid, doc_no="粤测农〔2099〕1号", clause_no="第一条",
                           effective_date="2099-01-01", validity="有效", level="省",
                           title="测试办法", is_current=True)
            for cid in CHUNK_TEXTS
        }
        report = build_report(
            "某项目",
            _sections(policy=(PolicyItem("[C1]", "与灌溉设施建设相关", None),),
                      missing=(MissingItem("[C2]", "项目申报书", "按指南核对"),),
                      uncovered=("④修改建议",)),
            negative_hits=(),
            table=meta_table,
        )
        text = render_report(report, meta_table, degraded_sections=("④修改建议",))
        assert "④ 修改建议〔已降级：未通过完整性校验，仅供参考〕" in text
        assert "① 适用政策清单\n" in text  # 未降级段无标注
        assert "① 适用政策清单〔已降级" not in text
