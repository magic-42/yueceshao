"""tests/test_entail_retry.py — entail 节点重试（方向4补齐，TDD 先红后绿）。

背景：真 LLM 蕴含复核实测踩到一个 gap——qwen 在 reason 里写「违规」被姿态校验拒收，
而 entail 节点只做一次调用＋解析，失败即降级为「仅规则命中」，②段永远出不来「疑似命中」。
本文件把 entail 对齐 generate 的 A9 语义：解析失败（SchemaError）→ 带反馈重试 ≤2 → 仍失败才降级。

红线：测试夹具用〔2099〕人造文号与自造条款原文。
"""

from __future__ import annotations

import json

from app.graph import OrchestrationComponents, build_checkup_graph
from app.pipeline import MOCK_GENERATION_JSON, MOCK_HITS

from gen.schema import HIT_SUSPECT

VALID = "我村拟申请帮扶资金，翻新改建村委会旧瓦房为会议室和接待室。"

EXTRACTION = json.dumps({
    "subject_type": "村集体经济组织",
    "construction_items": ["翻新改建村委会"],
    "use_items": [],
    "terms": [],
}, ensure_ascii=False)

# 姿态违规的蕴含 JSON（reason 含「违规」→ parse_entailments 拒收）
BAD_ENTAIL = json.dumps([
    {"citation": "[C1]", "verdict": "蕴含", "reason": "描述明确违规，落入条款禁止范围"},
], ensure_ascii=False)

# 合规的蕴含 JSON（中性口径「落入条款禁止范围」，通过校验）
CLEAN_ENTAIL = json.dumps([
    {"citation": "[C1]", "verdict": "蕴含", "reason": "描述落入条款禁止范围"},
], ensure_ascii=False)


def _multi_entail(outputs: tuple[str, ...]):
    """按调用次数逐次返回蕴含 JSON；耗尽后重复最后一个。"""
    served: list[int] = [0]

    def entail(description, hits, feedback=None):
        index = served[0]
        served[0] += 1
        return outputs[min(index, len(outputs) - 1)]

    return entail


def _components(entail) -> OrchestrationComponents:
    return OrchestrationComponents(
        extract=lambda d: EXTRACTION,
        retrieve=lambda d, e: MOCK_HITS,
        rule_scan=lambda d: (),
        entail_review=entail,
        generate=lambda d, ej, cb, fb: MOCK_GENERATION_JSON,
    )


class TestEntailRetry:
    def test_posture_violation_retries_with_feedback_then_succeeds(self) -> None:
        """首版 reason 含「违规」→ 带反馈重试 → 第二版中性口径 → ②段出现疑似命中。"""
        final = build_checkup_graph(
            _components(_multi_entail((BAD_ENTAIL, CLEAN_ENTAIL)))
        ).invoke({"description": VALID})
        assert final.get("entail_failed") is not True
        hits = final["negative_hits"]
        assert len(hits) == 1
        assert hits[0].hit_type == HIT_SUSPECT
        assert hits[0].citation == "[C1]"

    def test_retry_exhausted_degrades_to_rule_only(self) -> None:
        """连续 3 次姿态违规 → 重试耗尽 → ②段退化为仅规则命中 + 显式标注。"""
        final = build_checkup_graph(
            _components(_multi_entail((BAD_ENTAIL, BAD_ENTAIL, BAD_ENTAIL)))
        ).invoke({"description": VALID})
        assert final["entail_failed"] is True
        assert final["negative_hits"] == ()
        assert "LLM 蕴含复核未完成" in final["rendered_text"]

    def test_network_failure_degrades_without_retry(self) -> None:
        """非 SchemaError（如网络异常）→ 立即降级，不重试。"""
        def broken_entail(d, h, fb=None):
            raise RuntimeError("LLM 超时")

        final = build_checkup_graph(_components(broken_entail)).invoke({"description": VALID})
        assert final["entail_failed"] is True
        assert final["negative_hits"] == ()
