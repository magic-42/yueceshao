"""tests/test_graph_entail_retry.py — feat/langgraph-entail-retry（方向4）TDD 先红后绿。

覆盖两件事：
1. entail 节点接真实 LLM 蕴含复核：②段 = 规则命中 ∪ LLM「蕴含」命中（并集）；
2. 图级校验失败条件重试边：A9 重试从 generate 内部抽到图级（generate→validate→
   失败回 generate / 通过或耗尽进 render）。

红线：测试夹具只用〔2099〕人造文号与自造条款原文。
"""

from __future__ import annotations

import json

from app.graph import OrchestrationComponents, build_checkup_graph
from app.pipeline import MOCK_GENERATION_JSON, MOCK_HITS

from gen.schema import (
    ENTAILMENT_NO,
    ENTAILMENT_UNSURE,
    ENTAILMENT_YES,
    HIT_RULE,
    HIT_SUSPECT,
    NegativeHit,
)

VALID = (
    "我村拟申请帮扶资金 60 万元，用于修建 2 公里产业道路和配套灌溉设施，"
    "由村集体经济组织作为实施主体。"
)

EXTRACTION = json.dumps({
    "subject_type": "村集体经济组织",
    "construction_items": ["产业道路", "灌溉设施"],
    "use_items": [],
    "terms": [],
}, ensure_ascii=False)


def _components(entail_json: str, rule_hits: tuple = (), generation=MOCK_GENERATION_JSON):
    """构造注入组件：retrieve 用 Mock 命中，entail_review 返回给定 JSON。

    ``generation`` 可为固定 JSON 字符串，或按调用次数逐次返回的可调用对象。
    """
    gen = generation if callable(generation) else (lambda d, ej, cb, fb: generation)
    return OrchestrationComponents(
        extract=lambda d: EXTRACTION,
        retrieve=lambda d, e: MOCK_HITS,
        rule_scan=lambda d: rule_hits,
        entail_review=lambda d, h, fb=None: entail_json,
        generate=gen,
    )


# ---------------------------------------------------------------------------
# 1. entail 蕴含合并
# ---------------------------------------------------------------------------

class TestEntailMerge:
    def test_entail_yes_becomes_suspect_hit(self) -> None:
        """LLM 判「蕴含」→ ②段新增「疑似命中」，与规则命中并集。"""
        entail = json.dumps([
            {"citation": "[C1]", "verdict": ENTAILMENT_YES, "reason": "描述涉及禁止用途"},
        ], ensure_ascii=False)
        final = build_checkup_graph(_components(entail)).invoke({"description": VALID})
        hits = final["negative_hits"]
        assert len(hits) == 1
        assert hits[0].citation == "[C1]"
        assert hits[0].hit_type == HIT_SUSPECT
        assert hits[0].matched_words == ()

    def test_entail_no_is_discarded(self) -> None:
        entail = json.dumps([
            {"citation": "[C1]", "verdict": ENTAILMENT_NO, "reason": "与禁止用途无关"},
        ], ensure_ascii=False)
        final = build_checkup_graph(_components(entail)).invoke({"description": VALID})
        assert final["negative_hits"] == ()

    def test_entail_unsure_is_discarded(self) -> None:
        entail = json.dumps([
            {"citation": "[C1]", "verdict": ENTAILMENT_UNSURE, "reason": "依据不足"},
        ], ensure_ascii=False)
        final = build_checkup_graph(_components(entail)).invoke({"description": VALID})
        assert final["negative_hits"] == ()

    def test_rule_hit_takes_precedence_on_same_citation(self) -> None:
        """同一 citation 规则命中 + LLM 蕴含 → 保留规则命中（并集按 citation 去重）。"""
        rule_hit = NegativeHit(
            citation="[C1]", hit_type=HIT_RULE, matched_words=(("补助", "补助"),))
        entail = json.dumps([
            {"citation": "[C1]", "verdict": ENTAILMENT_YES, "reason": "描述涉及禁止用途"},
        ], ensure_ascii=False)
        final = build_checkup_graph(_components(entail, rule_hits=(rule_hit,))).invoke(
            {"description": VALID})
        hits = final["negative_hits"]
        assert len(hits) == 1
        assert hits[0].hit_type == HIT_RULE

    def test_rule_and_llm_hits_union(self) -> None:
        """规则命中 [C1] + LLM 蕴含 [C2] → 两条都在（并集）。"""
        rule_hit = NegativeHit(
            citation="[C1]", hit_type=HIT_RULE, matched_words=(("补助", "补助"),))
        entail = json.dumps([
            {"citation": "[C2]", "verdict": ENTAILMENT_YES, "reason": "描述涉及禁止用途"},
        ], ensure_ascii=False)
        final = build_checkup_graph(_components(entail, rule_hits=(rule_hit,))).invoke(
            {"description": VALID})
        citations = {h.citation for h in final["negative_hits"]}
        assert citations == {"[C1]", "[C2]"}


# ---------------------------------------------------------------------------
# 2. entail 失败降级
# ---------------------------------------------------------------------------

class TestEntailFailure:
    def test_entail_failure_degrades_to_rule_only(self) -> None:
        """entail 调真实 LLM 抛错 → ②段退化为仅规则命中 + 显式标注，不阻断管线。"""
        rule_hit = NegativeHit(
            citation="[C1]", hit_type=HIT_RULE, matched_words=(("补助", "补助"),))

        def broken_entail(d, h):
            raise RuntimeError("LLM 超时")

        components = OrchestrationComponents(
            extract=lambda d: EXTRACTION,
            retrieve=lambda d, e: MOCK_HITS,
            rule_scan=lambda d: (rule_hit,),
            entail_review=broken_entail,
            generate=lambda d, ej, cb, fb: MOCK_GENERATION_JSON,
        )
        final = build_checkup_graph(components).invoke({"description": VALID})
        assert final["negative_hits"] == (rule_hit,)
        assert final["entail_failed"] is True
        assert "LLM 蕴含复核未完成" in final["rendered_text"]


# ---------------------------------------------------------------------------
# 3. 图级条件重试边
# ---------------------------------------------------------------------------

def _multi_generate(outputs: tuple[str, ...]):
    """按调用次数逐次返回生成 JSON；耗尽后重复最后一个。"""
    served: list[int] = [0]

    def generate(d, ej, cb, fb):
        index = served[0]
        served[0] += 1
        return outputs[min(index, len(outputs) - 1)]

    return generate


class TestConditionalRetryEdge:
    def test_fail_then_pass_retries_through_graph(self) -> None:
        """首版非法 → 图级条件边回 generate 带反馈 → 第二版通过，attempts=2。"""
        bad = json.dumps({
            "policy_items": [{
                "citation": "[C99]", "summary": "引用不存在的标记", "confidence": None,
            }],
            "missing_items": [{"citation": "[C2]", "name": "申报书", "note": "核对"}],
            "suggestions": [{"text": "建议"}],
        }, ensure_ascii=False)
        components = _components("[]", generation=_multi_generate((bad, MOCK_GENERATION_JSON)))
        final = build_checkup_graph(components).invoke({"description": VALID})
        assert final["outcome"].degraded is False
        assert final["outcome"].attempts == 2
        assert final["gen_attempts"] == 2

    def test_exhaustion_degrades_after_three_attempts(self) -> None:
        """连续 3 次非法 → 耗尽降级，attempts=3，报告仍出免责声明。"""
        bad = json.dumps({
            "policy_items": [{
                "citation": "[C99]", "summary": "引用不存在的标记", "confidence": None,
            }],
            "missing_items": [{"citation": "[C2]", "name": "申报书", "note": "核对"}],
            "suggestions": [{"text": "建议"}],
        }, ensure_ascii=False)
        components = _components("[]", generation=_multi_generate((bad, bad, bad)))
        final = build_checkup_graph(components).invoke({"description": VALID})
        assert final["outcome"].degraded is True
        assert final["outcome"].attempts == 3
        assert final["gen_attempts"] == 3
        assert "免责声明" in final["rendered_text"]
