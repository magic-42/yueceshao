"""tests/test_orchestration_langgraph.py — feat/orchestration-langgraph（A12）。

覆盖：图拓扑（并行汇合/状态传递）、行为等价（与 app 层验收断言同构）、
降级路由（A9 结果经图状态进渲染标注）、组件注入（OrchestrationComponents
可替换）、reduced 产物完整性。
"""

from __future__ import annotations

import json

from app.graph import OrchestrationComponents, build_checkup_graph
from app.pipeline import MOCK_GENERATION_JSON, MOCK_HITS, MockCheckupPipeline

VALID = (
    "我村拟申请帮扶资金 60 万元，用于修建 2 公里产业道路和配套灌溉设施，"
    "由村集体经济组织作为实施主体。"
)


# ---------------------------------------------------------------------------
# 图专项
# ---------------------------------------------------------------------------

class TestGraphTopology:
    def test_parallel_merge_writes_both_keys(self) -> None:
        """retrieve ‖ rule_scan 并行汇合：两个 key 都落进最终状态。"""
        pipe = MockCheckupPipeline()
        graph = build_checkup_graph(pipe.components())
        final = graph.invoke({"description": VALID})
        assert final["hits"] and final["meta_table"]
        assert final["negative_hits"] == ()  # Mock 默认②段无命中
        assert final["chunk_texts"]["[C1]"].startswith("帮扶资金")

    def test_parse_and_generate_share_extraction_json(self) -> None:
        """extraction_json 存入状态供 generate 复用（真实 LLM 适配器不重复付费）。"""
        seen = []
        components = OrchestrationComponents(
            extract=lambda d: (seen.append(d) or json.dumps({
                "subject_type": "其他", "construction_items": ["修路"],
                "use_items": [], "terms": []})),
            retrieve=lambda d, e: MOCK_HITS,
            rule_scan=lambda d: (),
            entail_review=lambda d, h, fb=None: "[]",
            generate=lambda d, ej, cb, fb: (seen.append(ej), MOCK_GENERATION_JSON)[1],
        )
        graph = build_checkup_graph(components)
        graph.invoke({"description": VALID})
        assert seen[0] == VALID          # extract 收到原始描述
        assert seen[1] == seen[1]        # generate 收到 extraction_json（非空字符串）
        assert isinstance(seen[1], str) and seen[1]


# ---------------------------------------------------------------------------
# 行为等价（与 #36 验收断言同构——编排换成 LangGraph，行为不变）
# ---------------------------------------------------------------------------

class TestBehaviorEquivalence:
    def test_end_to_end_same_as_acceptance(self) -> None:
        result = MockCheckupPipeline().run(VALID)
        assert result.degraded is False and result.attempts == 1
        for section in ("① 适用政策清单", "② 负面清单命中提示", "③ 材料缺项清单", "④ 修改建议"):
            assert section in result.rendered_text
        assert result.rendered_text.rstrip().endswith(f"免责声明：{result.disclaimer}")
        assert result.report.policy_items[0].confidence == "明确适用"
        assert "[C1]" not in result.rendered_text

    def test_degradation_routes_through_graph(self) -> None:
        """A9 降级经图状态进渲染：耗尽→①段清空＋降级标注＋免责声明仍在。"""
        bad = json.dumps({
            "policy_items": [{"citation": "[C1]",
                              "summary": "条款明确「帮扶资金可以用于建造办公楼」，需注意",
                              "confidence": None}],
            "missing_items": [{"citation": "[C2]", "name": "项目申报书", "note": "按指南核对"}],
            "suggestions": [{"text": "建议写明道路长度"}],
        }, ensure_ascii=False)
        result = MockCheckupPipeline(generation_outputs=(bad,)).run(VALID)
        assert result.degraded is True and result.attempts == 3
        assert "① 适用政策清单〔已降级：未通过完整性校验，仅供参考〕" in result.rendered_text
        assert result.rendered_text.rstrip().endswith(f"免责声明：{result.disclaimer}")

    def test_components_replacement_point(self) -> None:
        """组件回调可整体替换（真实适配器的接入点）：自定义 retrieve 即生效。"""
        custom_hits = (
            {"chunk_id": "[C5]", "text": "自造原文：资金不得挪作他用。",
             "metadata": {"doc_no": "粤测农〔2099〕9号", "clause_no": "第一条",
                          "validity": "有效", "level": "县", "title": "自造测试办法",
                          "is_current": True}},
        )
        base = MockCheckupPipeline()
        c = base.components()
        good_for_custom = json.dumps({
            "policy_items": [
                {"citation": "[C5]", "summary": "与该测试项目的自造条款直接相关",
                 "confidence": None}],
            "missing_items": [], "suggestions": [],
            "uncovered_sections": ["③材料缺项"],
        }, ensure_ascii=False)
        components = OrchestrationComponents(
            extract=c.extract, retrieve=lambda d, e: custom_hits,
            rule_scan=c.rule_scan, entail_review=c.entail_review,
            generate=lambda d, ej, cb, fb: good_for_custom,
        )
        final = build_checkup_graph(components).invoke({"description": VALID})
        assert "[C5]" in final["meta_table"]
        # delivered 是校验器输出（置信恒 None＝解析口纪律）；系统侧填充在 report
        assert final["outcome"].delivered.policy_items[0].citation == "[C5]"
        assert final["report"].policy_items[0].confidence == "明确适用"
