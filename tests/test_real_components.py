"""tests/test_real_components.py — 真实检索适配器（feat/real-retrieval）。

三类用例：
1. 纯逻辑单元：命中 dict 形状转换（标记分配/字段别名/溯源字段）、规则库条款号归一、
   matched_words 配对、.env 解析、qwen 适配器缺 key 的清晰报错；
2. 注入假组件跑通「真规则＋假检索」的编排一致性（标记可解析、②段可渲染）；
3. 真实索引／模型／数据仓在位时的集成冒烟（沿用既有 skip 模式，CI 无索引全绿）：
   检索冒烟「产业园申报」断言只回现行有效条款（手册 §3.1 PR#4 验收口径）＋
   端到端（真检索＋真规则＋Mock LLM），全部命中可回溯、免责声明在位。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from app.real_components import (
    QwenKeyMissingError,
    QwenLLM,
    RealCheckupPipeline,
    RealComponents,
    clause_no_to_index,
    format_clause_label,
    hit_to_orchestration_hit,
    matched_word_pairs,
    read_env_file,
    rule_orchestration_hit,
)

from gen.citation import build_meta_table
from gen.schema import DISCLAIMER_TEXT, HIT_RULE
from rag.retriever import Hit, RetrievalResult
from rules.engine import MatchResult, RuleHit
from rules.schema import RuleRecord, RuleStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

def _rule(**overrides) -> RuleRecord:
    """合成一条规则记录（〔2099〕人造文号＋自造条款原文，红线⑤⑥）。"""
    defaults: dict = {
        "clause_text": "帮扶资金不得用于购置公务用车。",
        "source_ref": "粤测农〔2099〕1号 第四条",
        "clause_no": "四",
        "carries_enumeration": False,
        "level": "省",
        "effect_status": "有效",
        "group_id": "G99",
        "group_name": "测试组",
        "group_type": "锚词组",
        "is_negative_list_file": True,
        "doc_index": 99,
        "doc_no": "粤测农〔2099〕1号",
        "doc_title": "测试用资金管理办法",
        "doc_issuer": "测试机关",
        "doc_date": "2099-01-01",
        "doc_category": "资金管理办法",
        "group_decision": "待人工",
    }
    defaults.update(overrides)
    return RuleRecord(**defaults)


def _hit(chunk_id: str = "73-0010", metadata: dict | None = None) -> Hit:
    payload = {
        "chunk_id": chunk_id,
        "doc_no": "粤测农〔2099〕2号",
        "clause": "第三条",
        "clause_index": 3,
        "title": "测试用申报指南",
        "serial": 73,
        "category": "资金管理办法",
        "validity": "有效",
        "is_current": True,
        "has_doc_no": True,
        "publish_date": "2099-02-01",
        "structure": "clause",
    }
    payload.update(metadata or {})
    return Hit(
        chunk_id=chunk_id,
        text="申报主体应当提交项目申报书。",
        score=0.9,
        distance=0.1,
        metadata=payload,
    )


class FakeRetriever:
    """假检索器：返回固定 Hit 列表（不编码、不碰向量库）。"""

    def __init__(self, hits: tuple[Hit, ...]) -> None:
        self._hits = hits

    def retrieve(self, query: str, **_kwargs) -> RetrievalResult:
        return RetrievalResult(query=query, hits=list(self._hits), where={"is_current": True})


class FakeMatcher:
    """假规则匹配器：返回固定 MatchResult；store 为空规则库（无 G01-G11 载体）。"""

    def __init__(self, result: MatchResult) -> None:
        self._result = result
        self.store = RuleStore([])

    def match(self, description: str) -> MatchResult:
        return self._result


def _fake_components() -> tuple[RealComponents, RuleRecord]:
    rule = _rule()
    rule_hit = RuleHit(
        rule=rule,
        matched_terms=("公务用车",),
        matched_description_terms=("公务用车",),
        concept="交通工具与通讯设备",
        group_peers=(),
    )
    components = RealComponents(
        CONFIG_PATH,
        retriever=FakeRetriever((_hit(),)),
        matcher=FakeMatcher(MatchResult(hits=(rule_hit,), description_terms=("公务用车",))),
    )
    return components, rule


# ---------------------------------------------------------------------------
# 单元：条款号归一
# ---------------------------------------------------------------------------

class TestClauseNumbers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("十八", 18), ("第四条", 4), ("二十一", 21), ("二十九", 29), ("4", 4)],
    )
    def test_chinese_and_arabic_forms(self, raw: str, expected: int) -> None:
        assert clause_no_to_index(raw) == expected

    def test_unclaused_returns_none(self) -> None:
        assert clause_no_to_index("未分条（原文无条款号）") is None

    def test_label_normalisation(self) -> None:
        assert format_clause_label("十八") == "第十八条"
        assert format_clause_label("第四条") == "第四条"
        assert format_clause_label("未分条（原文无条款号）") == ""


# ---------------------------------------------------------------------------
# 单元：命中 dict 形状（与 Mock 夹具同构）
# ---------------------------------------------------------------------------

class TestHitMapping:
    def test_chunk_id_becomes_marker_and_real_id_is_kept(self) -> None:
        mapped = hit_to_orchestration_hit(_hit(), "[C1]")
        assert mapped["chunk_id"] == "[C1]"
        assert mapped["metadata"]["source_chunk_id"] == "73-0010"
        assert mapped["metadata"]["origin"] == "retrieval"
        assert mapped["text"] == "申报主体应当提交项目申报书。"

    def test_clause_no_alias_for_render_layer(self) -> None:
        # 索引字段名是 clause，渲染层读 clause_no——适配器必须补别名
        mapped = hit_to_orchestration_hit(_hit(), "[C1]")
        assert mapped["metadata"]["clause_no"] == "第三条"

    def test_meta_table_resolves_doc_no_from_mapping(self) -> None:
        table = build_meta_table([hit_to_orchestration_hit(_hit(), "[C1]")])
        assert table["[C1]"].doc_no == "粤测农〔2099〕2号"
        assert table["[C1]"].clause_no == "第三条"
        assert table["[C1]"].is_current is True

    def test_rule_block_carries_real_traceability_fields(self) -> None:
        mapped = rule_orchestration_hit(_rule(), "[C2]")
        meta = mapped["metadata"]
        assert meta["origin"] == "rules"
        assert meta["doc_no"] == "粤测农〔2099〕1号"
        assert meta["clause_no"] == "第四条"
        assert meta["validity"] == "有效"
        assert meta["level"] == "省"
        assert mapped["text"] == "帮扶资金不得用于购置公务用车。"


class TestMatchedWordPairs:
    def test_pairs_by_concept_group(self) -> None:
        rule_hit = RuleHit(
            rule=_rule(),
            matched_terms=("补助", "补贴"),
            matched_description_terms=("补助", "福利"),
            concept="奖金津贴福利",
            group_peers=(),
        )
        concepts = {"奖金津贴福利": ("补助", "补贴", "福利")}
        assert matched_word_pairs(rule_hit, concepts) == (
            ("补助", "补助"),
            ("补助", "补贴"),
            ("福利", "补助"),
            ("福利", "补贴"),
        )

    def test_falls_back_to_zip_when_concepts_unknown(self) -> None:
        rule_hit = RuleHit(
            rule=_rule(),
            matched_terms=("甲",),
            matched_description_terms=("乙",),
            concept="未知组",
            group_peers=(),
        )
        assert matched_word_pairs(rule_hit, {}) == (("乙", "甲"),)


# ---------------------------------------------------------------------------
# 单元：编排一致性（假检索＋真规则转换）
# ---------------------------------------------------------------------------

class TestPlanConsistency:
    def test_markers_are_sequential_and_rule_block_appended(self) -> None:
        components, _ = _fake_components()
        hits = components.retrieve("我们村申请补助资金购置公务用车", None)
        assert [hit["chunk_id"] for hit in hits] == ["[C1]", "[C2]"]
        assert hits[0]["metadata"]["origin"] == "retrieval"
        assert hits[1]["metadata"]["origin"] == "rules"

    def test_rule_hit_citation_resolves_through_meta_table(self) -> None:
        components, rule = _fake_components()
        description = "我们村申请补助资金购置公务用车"
        hits = components.retrieve(description, None)
        negative = components.rule_scan(description)
        table = build_meta_table(hits)
        assert len(negative) == 1
        assert negative[0].hit_type == HIT_RULE
        assert negative[0].citation == "[C2]"
        assert table[negative[0].citation].doc_no == rule.doc_no
        assert negative[0].matched_words == (("公务用车", "公务用车"),)

    def test_retrieve_and_rule_scan_share_the_same_plan(self) -> None:
        components, _ = _fake_components()
        first = components.retrieve("描述", None)
        second = components.retrieve("描述", None)
        assert first == second  # 同一描述命中计划复用，两次调用返回一致

    def test_duplicate_rule_hits_collapse(self) -> None:
        """同一（标记＋命中词）的重复规则记录只保留一条（②段不出现重复行）。"""
        rule = _rule()
        rule_hit = RuleHit(
            rule=rule,
            matched_terms=("公务用车",),
            matched_description_terms=("公务用车",),
            concept="交通工具与通讯设备",
            group_peers=(),
        )
        components = RealComponents(
            CONFIG_PATH,
            retriever=FakeRetriever((_hit(),)),
            matcher=FakeMatcher(MatchResult(
                hits=(rule_hit, rule_hit), description_terms=("公务用车",))),
        )
        assert len(components.rule_scan("我们村申报产业道路建设项目，计划购置公务用车一辆。")) == 1

    def test_concurrent_branches_compute_plan_once(self) -> None:
        """图里 retrieve 与 rule_scan 并发执行：加锁后只算一次计划。

        不锁时两路会各自算一遍：重复跑模型编码，且两批标记可能不同号（②段引用错位）。
        """
        calls: list[str] = []

        class SlowRetriever(FakeRetriever):
            def retrieve(self, query: str, **kwargs) -> RetrievalResult:
                calls.append(query)
                time.sleep(0.05)
                return super().retrieve(query, **kwargs)

        components = RealComponents(
            CONFIG_PATH,
            retriever=SlowRetriever((_hit(),)),
            matcher=FakeMatcher(MatchResult(hits=(), description_terms=())),
        )
        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def run_retrieve() -> None:
            barrier.wait()
            results["retrieve"] = components.retrieve("我们村申报产业道路建设项目", None)

        def run_rule_scan() -> None:
            barrier.wait()
            results["rule_scan"] = components.rule_scan("我们村申报产业道路建设项目")

        threads = [threading.Thread(target=run_retrieve), threading.Thread(target=run_rule_scan)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(calls) == 1, f"并发两路重复计算了 {len(calls)} 次命中计划"
        assert results["retrieve"] and results["rule_scan"] is not None


# ---------------------------------------------------------------------------
# 单元：.env 与 qwen 适配器骨架
# ---------------------------------------------------------------------------

class TestEnvAndQwen:
    def test_read_env_file_parses_comments_quotes_and_export(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            "# 注释\nDASHSCOPE_API_KEY=sk-test-123\n"
            "export OTHER_KEY='quoted-value'\n\nBADLINE\n",
            encoding="utf-8",
        )
        parsed = read_env_file(env)
        assert parsed == {"DASHSCOPE_API_KEY": "sk-test-123", "OTHER_KEY": "quoted-value"}

    def test_missing_env_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_env_file(tmp_path / "nope.env") == {}

    def test_qwen_raises_clear_error_without_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        llm = QwenLLM({}, env_path=tmp_path / ".env")
        with pytest.raises(QwenKeyMissingError) as excinfo:
            llm.extract("描述")
        message = str(excinfo.value)
        assert "DASHSCOPE_API_KEY" in message
        assert ".env" in message

    def test_qwen_uses_key_from_env_file(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("DASHSCOPE_API_KEY=sk-file\n", encoding="utf-8")
        llm = QwenLLM({}, env_path=env)
        assert llm.api_key == "sk-file"


# ---------------------------------------------------------------------------
# 集成：真实索引/模型/数据仓在位才跑（CI 无索引自动跳过）
# ---------------------------------------------------------------------------

def _real_ready_reason() -> str | None:
    """真实环境是否就位；不就位返回原因，就位返回 None。"""
    payload: dict = {}
    if CONFIG_PATH.is_file():
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    persist = Path(str((payload.get("index") or {}).get("persist_dir") or ".chroma"))
    if not persist.is_absolute():
        persist = PROJECT_ROOT / persist
    if not (persist / "chroma.sqlite3").is_file():
        return "本机无真实 Chroma 索引"
    model_dir = (payload.get("embedding") or {}).get("model_dir") or "models/Qwen3-Embedding-0.6B"
    model = Path(str(model_dir))
    if not model.is_absolute():
        model = PROJECT_ROOT / model
    if not (model / "config.json").is_file():
        return "本机无 embedding 模型权重"
    data_repo = Path(str(payload.get("data_repo") or ""))
    if not data_repo.is_absolute():
        data_repo = PROJECT_ROOT / data_repo
    if not (data_repo / "00_工作区" / "负面清单_条款归并.json").is_file():
        return "本机无数据仓（规则库）"
    return None


READY_REASON = _real_ready_reason()


@pytest.mark.skipif(READY_REASON is not None, reason=f"{READY_REASON}，跳过真实检索集成冒烟")
def test_real_retrieval_smoke_only_current_clauses() -> None:
    """手册 §3.1 PR#4 验收：查「产业园申报」只回现行有效条款（时效硬过滤前置）。"""
    components = RealComponents(CONFIG_PATH)
    hits = components.retrieve("产业园申报需要什么条件", None)
    dense = [hit for hit in hits if hit["metadata"].get("origin") == "retrieval"]
    assert dense, "真实索引检索无命中"
    for hit in dense:
        meta = hit["metadata"]
        assert meta["is_current"] is True
        assert meta["validity"] == "有效"
        assert meta["category"] != "审计公告"
    assert components.retrieval_summary("产业园申报需要什么条件")


@pytest.mark.skipif(READY_REASON is not None, reason=f"{READY_REASON}，跳过真实检索集成冒烟")
def test_real_rule_scan_markers_resolve() -> None:
    """②段真规则命中：NegativeHit 标记全部能在检索上下文元数据表中解析。"""
    components = RealComponents(CONFIG_PATH)
    description = "我村申报产业项目，计划购置公务用车用于项目巡查。"
    hits = components.retrieve(description, None)
    negative = components.rule_scan(description)
    assert negative, "真规则库对该描述应有命中"
    table = build_meta_table(hits)
    for hit in negative:
        assert hit.hit_type == HIT_RULE
        assert hit.citation in table
        meta = table[hit.citation]
        assert meta.doc_no, "规则命中必须带真实文号"


@pytest.mark.skipif(READY_REASON is not None, reason=f"{READY_REASON}，跳过真实端到端")
def test_real_end_to_end_with_mock_llm() -> None:
    """端到端：真检索＋真规则＋Mock LLM，产出四段式报告且免责声明在位。"""
    pipeline = RealCheckupPipeline(CONFIG_PATH)
    description = (
        "我们村集体经济组织申报补改投现代农业产业园试点项目，主导产业为本地特色水稻种植，"
        "生产用地已按法定程序完成流转备案，拟申请补助资金用于产业道路和烘干设施建设。"
    )
    result = pipeline.run(description)
    assert result.report
    assert result.rendered_text
    assert DISCLAIMER_TEXT in result.rendered_text
    assert result.report.negative_hits, "真规则库应命中「补助」，②段非空"
    assert result.report.policy_items, "①段应有真实条款"
    table = build_meta_table(pipeline.components.retrieve(description, None))
    for item in result.report.policy_items:
        assert table[item.citation].doc_no, "①段引用必须来自检索返回的元数据"
