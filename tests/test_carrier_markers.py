"""tests/test_carrier_markers.py — 载体标记分配与蕴含复核覆盖（issue #48 回归）。

issue #48 的两个缺陷（#47 终审时量化发现，均由 CodeRabbit Major 引出）：

1. **复用标记的载体漏出蕴含复核**：载体若与稠密检索或规则命中落在同一 (文件, 条款号)，
   旧实现让它复用已有标记、不追加新块 → 它不以 origin=negative-list 出现在 hits 里 →
   而蕴含复核只按 hits 的 origin 过滤取块 → 该载体**永远不会被 LLM 复核看到**。
2. **同文件同条款号不同原文共用键**：旧实现用 (文件序号, 条款号) 作标记键，同一「第X条」
   下多个款级子块（不同原文）会共用第一个标记，后一个**整块静默丢失**。

修法：标记键改为 (文件序号, 条款号, 条款原文)；同时把载体**单独清单化**（``carrier_hits``），
蕴含复核直接喂这份清单，不再依赖 hits 的 origin 过滤。

红线：合成夹具用〔2099〕人造文号与自造条款原文，不含真实政策语料。
"""

from __future__ import annotations

from pathlib import Path

from app.real_components import (
    ORIGIN_NEGATIVE,
    ORIGIN_RULES,
    RealComponents,
    render_context_blocks,
)

from gen.citation import build_meta_table
from rag.retriever import Hit, RetrievalResult
from rules.engine import NegRuleMatcher
from rules.loader import records_from_dicts
from rules.schema import RuleStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"


def _rule_dict(
    clause_text: str,
    *,
    doc_index: int = 1,
    clause_no: str = "四",
    group_id: str = "G04",
    carries_enumeration: bool = False,
) -> dict:
    """合成规则记录（人造文号＋自造条款原文）。"""
    return {
        "条款原文逐字": clause_text,
        "出处文号+条款号": f"粤测农〔2099〕{doc_index}号 {clause_no}",
        "条款号": clause_no,
        "承接列举": carries_enumeration,
        "层级": "省",
        "所在文件效力状态": "有效",
        "等价组建议": {"组号": group_id, "组名": "测试组", "组类型": "锚词组"},
        "负面清单载体文件": True,
        "溯源_文件序号": doc_index,
        "溯源_文号": f"粤测农〔2099〕{doc_index}号",
        "溯源_标题": "测试用办法（人造数据）",
        "溯源_发布机关": "测试机关",
        "溯源_发布日期": "2099-01-01",
        "溯源_类别": "资金管理办法",
        "组裁定": "接受",
    }


CLAUSE = "项目资金不得用于购置公务用车。"
"""夹具条款：含「公务用车」，与描述侧术语同概念组 → 规则可命中；同时满足载体判据。"""

DESC = "我村申报产业项目，拟购置公务用车一辆"
"""夹具描述：命中「交通工具与通讯设备」概念组。"""


def _hit(chunk_id: str = "1-0004", *, serial: int = 1, clause_index: int = 4) -> Hit:
    """假检索命中（与载体条款同文件同条序号，用于验证"不再复用稠密标记"）。"""
    return Hit(
        chunk_id=chunk_id,
        text=CLAUSE,
        score=0.9,
        distance=0.1,
        metadata={
            "doc_no": "粤测农〔2099〕1号",
            "clause": "第四条",
            "clause_index": clause_index,
            "serial": serial,
            "title": "测试用办法（人造数据）",
            "category": "资金管理办法",
            "validity": "有效",
            "is_current": True,
            "has_doc_no": True,
            "structure": "clause",
        },
    )


class FakeRetriever:
    """假检索器：返回固定命中（不编码、不碰向量库）。"""

    def __init__(self, hits: tuple[Hit, ...]) -> None:
        self._hits = hits

    def retrieve(self, query: str, **_kwargs) -> RetrievalResult:
        return RetrievalResult(query=query, hits=list(self._hits), where={"is_current": True})


class CapturingLLM:
    """记录 entail_review 实际收到的块，其余委托给 Mock 夹具。"""

    def __init__(self) -> None:
        self.entail_calls: list[tuple[dict, ...]] = []

    def extract(self, description: str) -> str:
        from app.pipeline import MOCK_EXTRACTION_JSON

        return MOCK_EXTRACTION_JSON

    def entail_review(self, description: str, hits, feedback: str | None = None) -> str:
        from app.pipeline import MOCK_ENTAILMENT_JSON

        self.entail_calls.append(tuple(hits))
        return MOCK_ENTAILMENT_JSON

    def generate(self, description, extraction_json, context_blocks, feedback) -> str:
        from app.pipeline import MOCK_GENERATION_JSON

        return MOCK_GENERATION_JSON


def _components(rules: list[dict], hits: tuple[Hit, ...] = ()) -> RealComponents:
    store = RuleStore(records_from_dicts(rules))
    return RealComponents(
        CONFIG_PATH, retriever=FakeRetriever(hits), matcher=NegRuleMatcher(store))


class TestCarriersReachEntailReview:
    """缺陷 1：复用标记的载体也必须进蕴含复核。"""

    def test_carrier_covered_by_rule_hit_still_reaches_entail(self) -> None:
        """载体条款同时被规则命中（同文）时，仍须出现在载体复核清单里。"""
        text = CLAUSE
        # 不注入稠密命中：让该条款只由规则引擎命中（origin=rules），载体随之复用规则标记——
        # 这正是"复用标记后仍须进复核"的场景（若注入稠密命中，块会变成 origin=retrieval）。
        components = _components([_rule_dict(text)])
        description = DESC
        plan = components._plan(description)

        rule_blocks = [h for h in plan.hits if h["metadata"]["origin"] == ORIGIN_RULES]
        assert rule_blocks, "夹具应产生规则命中块（该条款同时是载体）"
        assert len(plan.carrier_hits) == 1, "载体必须进复核清单，不能因复用标记而被跳过"
        carrier = plan.carrier_hits[0]
        assert carrier["metadata"]["origin"] == ORIGIN_NEGATIVE
        assert carrier["text"] == text

        table = build_meta_table(plan.hits)
        assert carrier["chunk_id"] in table, "载体标记必须能在元数据表解析（②段引用可回溯）"

    def test_entail_review_receives_carrier_list_not_origin_filtered_hits(self) -> None:
        """components() 的 entail 回调必须传载体清单，而不是 hits 里 origin 过滤后的子集。"""
        text = CLAUSE
        components = _components([_rule_dict(text)], (_hit(),))
        llm = CapturingLLM()
        components._llm = llm
        description = DESC

        components.components().entail_review(description, components.retrieve(description, None))

        assert len(llm.entail_calls) == 1
        received = llm.entail_calls[0]
        assert len(received) == len(components._plan(description).carrier_hits)
        assert all(h["metadata"]["origin"] == ORIGIN_NEGATIVE for h in received)
        blocks = render_context_blocks(received)
        assert text in blocks

    def test_carrier_not_reusing_dense_marker_even_when_clause_retrieved(self) -> None:
        """稠密命中落在载体条款上时，载体仍拿自己的 negative-list 块（不复用稠密标记）。"""
        text = CLAUSE
        components = _components([_rule_dict(text)], (_hit(serial=1, clause_index=4),))
        plan = components._plan(DESC)

        dense = [h for h in plan.hits if h["metadata"]["origin"] == "retrieval"]
        assert dense, "夹具应有稠密命中"
        carrier = plan.carrier_hits[0]
        assert carrier["chunk_id"] not in {h["chunk_id"] for h in dense}, (
            "载体不得复用稠密标记，否则 origin 会变回 retrieval 而被复核漏掉")


class TestDistinctTextsGetDistinctMarkers:
    """缺陷 2：同一 (文件, 条款号) 下不同原文不得共用标记。"""

    def test_two_subclauses_same_article_both_kept(self) -> None:
        first = "严禁将项目资金用于股红分配或者举债分红（本句为测试合成样例）。"
        second = "严禁因机构调整、班子换届等情形突击分红（本句为测试合成样例）。"
        components = _components([
            _rule_dict(first, doc_index=601, clause_no="五十三"),
            _rule_dict(second, doc_index=601, clause_no="五十三"),
        ])
        plan = components._plan("拟用帮扶资金向村民分红")

        texts = {h["text"] for h in plan.hits}
        assert first in texts and second in texts, "同条不同原文都必须保留（旧实现会丢后一个）"

        markers = [h["chunk_id"] for h in plan.hits]
        assert len(markers) == len(set(markers)), "每个块必须有独立标记"

    def test_same_text_same_marker(self) -> None:
        """完全相同的原文仍复用同一标记（不因去重键变化而重复分块）。"""
        text = "严禁将项目资金用于股红分配或者举债分红（本句为测试合成样例）。"
        components = _components([
            _rule_dict(text, doc_index=601, clause_no="五十三"),
            _rule_dict(text, doc_index=601, clause_no="五十三"),
        ])
        plan = components._plan("拟用帮扶资金向村民分红")
        same_text_blocks = [h for h in plan.hits if h["text"] == text]
        assert len(same_text_blocks) == 1, "同文同条不应重复分块"


class TestGenerationContextUnaffected:
    """①③④段生成上下文仍不含载体（#47 已修，本轮不得回退）。"""

    def test_generation_context_excludes_carriers(self) -> None:
        """载体必须真拿到自己的 negative-list 块才谈得上“不进①段”，故描述不与载体条款命中。

        反例提醒（终审指出）：若描述恰好命中该载体条款，载体复用规则标记、不会产生
        `origin=negative-list` 的块，``carrier_markers`` 就成了空集，断言会**空转通过**。
        """
        text = CLAUSE
        components = _components([_rule_dict(text, group_id="G04")])
        # 描述只提到「差旅」，与夹具条款（公务用车/交通工具组）无共现 → 规则不命中该条款
        description = "我村申报项目，拟列支项目管理人员的差旅费用"
        plan = components._plan(description)
        assert plan.carrier_hits, "夹具应产出载体复核项"

        carrier_markers = {h["chunk_id"] for h in plan.hits
                           if h["metadata"]["origin"] == ORIGIN_NEGATIVE}
        assert carrier_markers, "本用例要求至少有 1 个载体自有块（否则断言空转）"

        table = build_meta_table(plan.hits)
        for marker in carrier_markers:
            assert marker in table, "载体标记必须能在元数据表解析（②段引用可回溯）"
            assert marker not in components.context_blocks(description), (
                f"载体 {marker} 不得进入①段生成上下文")
        assert text not in components.context_blocks(description), "载体原文不得进入①段上下文"
