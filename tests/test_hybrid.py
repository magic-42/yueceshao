"""rag.hybrid 的单元测试。

对应《开发手册 v1.0》§3.1 PR#5 验收标准：

1. **BM25（jieba＋政策术语词典）＋RRF 混合**（A5）——本文件覆盖术语不被切碎、
   RRF 公式与权重、两路互补；
2. **混合 vs 纯稠密在样例查询上的对比记录进 PR 描述**——本文件覆盖"混合能召回
   稠密漏掉的块"（``test_hybrid_recovers_chunk_dense_missed``），真实语料上的
   10 条查询对比见 PR 描述。

不装 chromadb／torch 也能跑：稠密一路用满足协议的替身，BM25 只依赖 jieba／rank_bm25
（两者都在 requirements.txt 里，CI 会装）。
"""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from rag.hybrid import (
    BM25Index,
    FusedHit,
    HybridConfig,
    HybridRetriever,
    PolicyTokenizer,
    load_policy_terms,
    rrf_fuse,
)
from rag.retriever import Hit, RetrievalConfig

# ================================================================ 测试替身


def make_record(
    chunk_id: str,
    text: str,
    *,
    is_current: bool = True,
    category: str = "资金管理办法",
    doc_no: str = "粤农农规〔2023〕6号",
    clause: str = "第十条",
    title: str = "广东省现代农业产业园财政资金管理办法",
) -> dict[str, Any]:
    """构造一条块记录（映射形式，与 JSONL 结构一致）。"""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "doc_no": doc_no,
            "title": title,
            "clause": clause,
            "category": category,
            "is_current": is_current,
            "validity": "有效" if is_current else "已失效",
            "publish_date": "2023-12-16",
            "serial": 1,
        },
    }


class FakeDense:
    """稠密一路替身：按预设返回命中，并记录调用参数。"""

    def __init__(self, hits: Sequence[Hit], config: RetrievalConfig | None = None) -> None:
        self.hits = list(hits)
        self.config = config or RetrievalConfig()
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
        count_candidates: bool = False,
    ) -> Any:
        """返回预设命中（截断到 top_k）。"""
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "where": dict(where or {}),
                "temporal": self.config.enable_temporal_filter,
            }
        )
        limit = len(self.hits) if top_k is None else int(top_k)
        return types.SimpleNamespace(query=query, hits=self.hits[:limit])


def make_hit(chunk_id: str, score: float = 0.6, **meta: Any) -> Hit:
    """构造一条稠密命中。"""
    record = make_record(chunk_id, f"{chunk_id} 的正文内容", **meta)
    return Hit(
        chunk_id=chunk_id,
        text=record["text"],
        score=score,
        distance=1.0 - score,
        metadata=record["metadata"],
    )


# ================================================================ 分词与词典


@pytest.fixture(scope="module")
def tokenizer() -> PolicyTokenizer:
    """模块级分词器：jieba 初始化约 1 秒，不必每个用例重建。"""
    return PolicyTokenizer()


def test_policy_terms_file_is_readable() -> None:
    """词典文件必须存在且有内容——它是 A5 里"政策术语词典"的实现。"""
    terms = load_policy_terms()
    assert len(terms) >= 60, f"术语仅 {len(terms)} 条，过少"
    assert "联农带农" in terms
    assert "驻镇帮镇扶村" in terms
    assert "强村公司" in terms


def test_missing_terms_file_degrades_gracefully(tmp_path: Any) -> None:
    """词典缺失只告警、不抛异常：分词退化为通用切分，检索链不能因此挂掉。"""
    assert load_policy_terms(tmp_path / "不存在.txt") == ()


def test_policy_term_kept_whole(tokenizer: PolicyTokenizer) -> None:
    """政策专名必须整体保留——切碎后 BM25 会以碎片匹配，IDF 被稀释。"""
    tokens = tokenizer.cut("联农带农机制和驻镇帮镇扶村工作队的职责")
    assert "联农带农" in tokens, tokens
    assert "驻镇帮镇扶村" in tokens, tokens


def test_stopwords_dropped(tokenizer: PolicyTokenizer) -> None:
    """疑问词与虚词要滤掉，否则口语查询会被"什么""需要"拉偏。"""
    tokens = tokenizer.cut("产业园申报需要什么条件")
    assert "什么" not in tokens
    assert "需要" not in tokens
    assert "产业园" in tokens or "申报" in tokens


def test_stopwords_can_be_kept() -> None:
    """可关闭停用词过滤（对照实验要用同一套口径时）。"""
    keeper = PolicyTokenizer(drop_stopwords=False)
    assert "什么" in keeper.cut("需要什么条件")


# ================================================================ BM25 索引


@pytest.fixture(scope="module")
def bm25_records() -> list[dict[str, Any]]:
    """三块：失效块与查询高度相关、现行块中等相关、无关块。"""
    return [
        make_record(
            "1-0001",
            "产业园申报指南：产业园申报应当提交产业园申报材料，产业园申报窗口为每年六月。"
            "产业园申报主体须为强村公司。",
            is_current=False,
            doc_no="粤办函〔2021〕207号",
        ),
        make_record(
            "1-0002",
            "产业园申报材料清单包括项目实施方案与资金使用计划，申报主体须为强村公司。",
            is_current=True,
        ),
        make_record(
            "1-0003",
            "合作社财务管理应当建立账簿，按期公开财务收支。",
            is_current=True,
            doc_no="粤农农规〔2022〕5号",
        ),
    ]


def test_bm25_returns_relevant_chunk(
    bm25_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """基本召回：查询"产业园申报"应命中两块产业园相关块。"""
    index = BM25Index(bm25_records, tokenizer=tokenizer)
    hits = index.search("产业园申报", k=5, require_current=True)

    assert [chunk_id for chunk_id, _ in hits] == ["1-0002"], "失效块应被时效掩码挡掉"
    assert len(index) == 3


def test_bm25_filters_before_truncating(
    bm25_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """**先过滤、后截断**：k=1 时不能因为失效块分数最高就返回空。

    若实现写成"取 top-1 再筛"，这里会拿到失效块 ``1-0001``、筛完剩空，
    用户看到"没有依据"——而实际有一块现行条款高度相关。这条用例专门把
    写回后置过滤的改动打红。
    """
    index = BM25Index(bm25_records, tokenizer=tokenizer)

    top1 = index.search("产业园申报", k=1, require_current=True)

    assert len(top1) == 1, "独立 k=1 也不该塌陷成空结果"
    assert top1[0][0] == "1-0002"


def test_bm25_temporal_switch_off(
    bm25_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """关掉时效掩码后，失效块确实回到首位（证明开关可观测，能用于消融）。"""
    index = BM25Index(bm25_records, tokenizer=tokenizer)

    hits = index.search("产业园申报", k=3, require_current=False)

    assert hits[0][0] == "1-0001", "失效块与查询高度相关，关掉过滤后应排第一"
    assert hits[0][1] > 0


def test_bm25_excludes_audit_category(tokenizer: PolicyTokenizer) -> None:
    """红线④：审计公告不进稀疏索引（BM25 侧的第一道拦截）。"""
    records = [
        make_record("9-0001", "产业园申报审计发现问题的公告", category="审计公告"),
        make_record("1-0002", "产业园申报材料清单"),
    ]

    index = BM25Index(records, tokenizer=tokenizer)

    assert index.chunk_ids == ("1-0002",)


def test_bm25_deduplicates_ids(tokenizer: PolicyTokenizer) -> None:
    """同一块重复出现只保留第一份，避免分数被重复计入。"""
    record = make_record("1-0001", "产业园申报材料清单")
    index = BM25Index([record, dict(record)], tokenizer=tokenizer)

    assert len(index) == 1


def test_bm25_where_equality(tokenizer: PolicyTokenizer) -> None:
    """等值条件（定文号）在稀疏侧同样生效。"""
    records = [
        make_record("1-0001", "产业园申报材料清单", doc_no="甲〔2023〕1号"),
        make_record("1-0002", "产业园申报材料清单", doc_no="乙〔2023〕2号"),
    ]
    index = BM25Index(records, tokenizer=tokenizer)

    hits = index.search("产业园申报材料", k=5, where={"doc_no": "乙〔2023〕2号"})

    assert [chunk_id for chunk_id, _ in hits] == ["1-0002"]


def test_bm25_rejects_unexpected_where_operator(tokenizer: PolicyTokenizer) -> None:
    """不支持的复合条件必须报错，不能静默忽略（否则调用方以为过滤生效了）。"""
    index = BM25Index([make_record("1-0001", "产业园申报")], tokenizer=tokenizer)

    with pytest.raises(NotImplementedError):
        index.search("产业园申报", where={"$or": [{"doc_no": "甲"}]})


def test_bm25_empty_query_returns_empty(
    bm25_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """空查询／全停用词查询不打分，直接返回空。"""
    index = BM25Index(bm25_records, tokenizer=tokenizer)

    assert index.search("", k=5) == []
    assert index.search("什么 怎么", k=5) == []


def test_bm25_empty_corpus_raises() -> None:
    """空语料要明确报错，而不是建出一个永远返回空的索引。"""
    with pytest.raises(ValueError):
        BM25Index([])


# ================================================================ RRF 融合


def test_rrf_formula() -> None:
    """分数严格等于 ``w / (k + rank)``，累加同一 id 的多路贡献。"""
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], rrf_k=60, weights=[0.5, 0.5]))

    assert fused["a"] == pytest.approx(0.5 / 61)
    assert fused["b"] == pytest.approx(0.5 / 62 + 0.5 / 61)
    assert fused["c"] == pytest.approx(0.5 / 62)
    assert fused["b"] > fused["a"] > fused["c"]


def test_rrf_weights_shift_order() -> None:
    """权重能改变排序：把稀疏一路权重拉满，只在该路出现的块应胜出。"""
    equal = dict(rrf_fuse([["a"], ["b"]], weights=[0.5, 0.5]))
    assert equal["a"] == pytest.approx(equal["b"])

    dense_heavy = dict(rrf_fuse([["a"], ["b"]], weights=[0.9, 0.1]))
    assert dense_heavy["a"] > dense_heavy["b"]


def test_rrf_tie_keeps_first_seen_order() -> None:
    """同分时保持首次出现顺序，避免排序不稳定导致评测结果抖动。"""
    fused = rrf_fuse([["x", "y"], ["y", "x"]])
    assert fused[0][0] == "x"


def test_rrf_limit() -> None:
    """limit 截断在排序之后。"""
    fused = rrf_fuse([["a", "b", "c"]], limit=2)
    assert [item[0] for item in fused] == ["a", "b"]


def test_rrf_rejects_bad_params() -> None:
    """参数校验：k 必须为正、权重长度要匹配、权重不能为负。"""
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], rrf_k=0)
    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], weights=[1.0])
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], weights=[-1.0])


def test_rrf_second_route_can_overtake() -> None:
    """两路排名能叠加反超：``b`` 在两路都排第二，应胜过只在第一路排第一的 ``a``。"""
    fused = dict(rrf_fuse([["a", "b"], ["b"]], weights=[0.5, 0.5]))

    assert fused["b"] > fused["a"], "双路都命中应胜过单路第一"


# ================================================================ 混合检索器


@pytest.fixture(scope="module")
def hybrid_records() -> list[dict[str, Any]]:
    """四块：一块只有稠密能召回（语义相近但用词不同），一块只有 BM25 能召回（原词）。"""
    return [
        make_record("1-0001", "申报主体须为依法登记的强村公司，注册资本不低于一百万元。"),
        make_record("1-0002", "产业园申报材料清单包括项目实施方案与资金使用计划。"),
        make_record("1-0003", "资金支出范围包括产业奖补、基础设施配套与技能培训。"),
        make_record("1-0004", "合作社应当建立财务账簿并定期公开。", doc_no="粤农农规〔2022〕5号"),
    ]


def test_hybrid_orders_by_fusion_score(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """两路都命中的块应排在只有一路命中的前面（RRF 累加的效果）。"""
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001", 0.70), make_hit("1-0002", 0.65)])
    # 让稀疏侧只命中 1-0002 与 1-0003
    retriever = HybridRetriever(dense, sparse, HybridConfig(top_k=3, dense_k=5, bm25_k=5))

    result = retriever.retrieve("产业园申报材料清单")

    ids = [hit.chunk_id for hit in result.hits]
    assert "1-0002" in ids
    assert result.hits[0].fusion_score >= result.hits[-1].fusion_score
    assert result.hits[0].source == "both"


def test_hybrid_recovers_chunk_dense_missed(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """**PR#5 的核心收益**：只有稀疏侧召回的块也能进最终结果。

    稠密侧返回两块的顺序是 [1-0001, 1-0004]，完全没提"产业园申报材料清单"；
    稀疏侧靠原词命中 1-0002。融合后 1-0002 必须出现在结果里——若实现只做
    "稠密 + 稀疏求交集"，这块就丢了，混合等于没做。
    """
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001", 0.70), make_hit("1-0004", 0.66)])
    retriever = HybridRetriever(dense, sparse, HybridConfig(top_k=5, dense_k=5, bm25_k=5))

    result = retriever.retrieve("产业园申报材料清单")

    assert "1-0002" in [hit.chunk_id for hit in result.hits]
    recovered = next(hit for hit in result.hits if hit.chunk_id == "1-0002")
    assert recovered.source == "sparse"
    assert recovered.dense_rank is None
    assert recovered.sparse_rank is not None
    assert recovered.bm25_score and recovered.bm25_score > 0
    assert recovered.text.startswith("产业园申报材料清单"), "正文必须来自块记录，不能为空"


def test_hybrid_syncs_dense_temporal_switch(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """**消融口径一致性**：混合侧关掉时效过滤，稠密侧在检索时必须同步关掉。

    若不同步，跑出来的"关闭对照组"其实只关了一半——稀疏侧放开了失效条款，
    稠密侧还锁着，对比结论直接作废。这是混合架构里最隐蔽的错误。
    """
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001")], config=RetrievalConfig(enable_temporal_filter=True))
    retriever = HybridRetriever(dense, sparse, HybridConfig(enable_temporal_filter=False))

    retriever.retrieve("产业园申报")

    assert dense.calls[0]["temporal"] is False, "稠密侧在检索时必须已被同步为关闭"


def test_hybrid_restores_dense_config_after_retrieve(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """对齐是**临时**的：检索结束后分享进来的 DenseRetriever 必须恢复原状。

    否则一次消融实验会把别人（或后续正式流程）的时效过滤永久关掉——
    只在"先跑消融、再跑正式"的顺序下暴露，极难排查。
    """
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001")], config=RetrievalConfig(enable_temporal_filter=True))
    retriever = HybridRetriever(dense, sparse, HybridConfig(enable_temporal_filter=False))

    retriever.retrieve("产业园申报")

    assert dense.config.enable_temporal_filter is True, "检索结束后开关要还原"


def test_hybrid_restores_dense_config_on_error(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """稠密侧抛异常时也要还原配置（finally 保证）。"""
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)

    class BoomDense(FakeDense):
        def retrieve(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("模拟检索异常")

    dense = BoomDense([], config=RetrievalConfig(enable_temporal_filter=True))
    retriever = HybridRetriever(dense, sparse, HybridConfig(enable_temporal_filter=False))

    with pytest.raises(RuntimeError):
        retriever.retrieve("产业园申报")

    assert dense.config.enable_temporal_filter is True


def test_hybrid_keeps_dense_temporal_when_already_aligned(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """两侧已一致时不做无谓替换（避免每次查询都重建配置对象）。"""
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001")], config=RetrievalConfig(enable_temporal_filter=True))
    retriever = HybridRetriever(dense, sparse, HybridConfig(enable_temporal_filter=True))
    before = dense.config

    retriever.retrieve("产业园申报")

    assert dense.config is before


def test_hybrid_passes_where_to_dense(tokenizer: PolicyTokenizer) -> None:
    """附加条件要原样下发给稠密侧（向量库层才有 where 能力）。"""
    sparse = BM25Index([make_record("1-0001", "产业园申报材料清单")], tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001")])
    retriever = HybridRetriever(dense, sparse)

    retriever.retrieve("产业园申报", where={"doc_no": "粤农农规〔2023〕6号"})

    assert dense.calls[0]["where"] == {"doc_no": "粤农农规〔2023〕6号"}


def test_hybrid_drops_non_retrievable_dense_hit(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """红线④第三道：稠密侧若混进类别缺失的脏块，融合阶段必须丢弃。"""
    dirty = Hit("9-0001", "类别缺失的可疑块", 0.9, 0.1, {"chunk_id": "9-0001", "is_current": True})
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([dirty])
    retriever = HybridRetriever(dense, sparse, HybridConfig(top_k=5))

    result = retriever.retrieve("产业园申报材料清单")

    assert "9-0001" not in [hit.chunk_id for hit in result.hits]


def test_hybrid_empty_query(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """空查询不打任何一路，返回空结果且不抛异常。"""
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001")])
    retriever = HybridRetriever(dense, sparse)

    result = retriever.retrieve("   ")

    assert result.hits == []
    assert dense.calls == []


def test_hybrid_respects_top_k(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """最终条数受 top_k 约束，两路候选池各自更大。"""
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit(f"1-000{index}", 0.7 - index / 100) for index in range(1, 5)])
    retriever = HybridRetriever(dense, sparse, HybridConfig(top_k=10, dense_k=4, bm25_k=4))

    result = retriever.retrieve("产业园申报材料清单", top_k=2)

    assert len(result.hits) <= 2
    assert dense.calls[0]["top_k"] == 4, "稠密一路仍按 dense_k 取候选"


def test_hybrid_source_counts_and_summary(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """来源统计与摘要能反映"混合补了多少"——评测要用这两个数说话。"""
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001", 0.7)])
    retriever = HybridRetriever(dense, sparse, HybridConfig(top_k=5, dense_k=5, bm25_k=5))

    result = retriever.retrieve("产业园申报材料清单")

    counts = result.source_counts()
    assert counts["both"] + counts["dense"] + counts["sparse"] == len(result.hits)
    assert "融合" in result.summary()
    # 本例两路各命中不同的块：稠密命中 1-0001（语义"申报主体"），稀疏命中 1-0002
    # （原词"产业园申报材料清单"）。overlap=0 时两路结果**都要保留**——若写成求交集，
    # 这里会剩 0 条，混合检索的意义就没了。
    assert result.overlap == 0
    assert counts["dense"] == 1
    assert counts["sparse"] == 1
    assert len(result.hits) == 2


def test_fused_hit_serialization(tokenizer: PolicyTokenizer) -> None:
    """``as_dict`` 要同时给出融合信息与原始元数据，评测脚本靠它落 JSONL。"""
    sparse = BM25Index([make_record("1-0001", "产业园申报材料清单")], tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001", 0.7)])
    result = HybridRetriever(dense, sparse).retrieve("产业园申报材料清单")

    payload = result.hits[0].as_dict()

    assert payload["chunk_id"] == "1-0001"
    assert payload["fusion_score"] > 0
    assert payload["source"] == "both"
    assert payload["metadata"]["doc_no"] == "粤农农规〔2023〕6号"
    assert "dense_rank" in payload and "sparse_rank" in payload


def test_hybrid_config_from_config() -> None:
    """配置段读取：缺字段回落默认值，给了就用给的。"""
    default = HybridConfig.from_config(None)
    assert default.rrf_k == 60
    assert default.dense_weight == 0.5

    custom = HybridConfig.from_config(
        {"hybrid": {"top_k": 8, "rrf_k": 40, "dense_weight": 0.7, "sparse_weight": 0.3}}
    )
    assert custom.top_k == 8
    assert custom.rrf_k == 40
    assert custom.dense_weight == 0.7
    assert custom.bm25_k == 40, "未给的字段保持默认"


def test_fused_hit_source_label() -> None:
    """来源标记：两路都有是 both，只有一路则分别是 dense / sparse。"""
    hit = make_hit("1-0001")
    assert FusedHit(hit, 0.1, dense_rank=1, sparse_rank=2).source == "both"
    assert FusedHit(hit, 0.1, dense_rank=1).source == "dense"
    assert FusedHit(hit, 0.1, sparse_rank=2).source == "sparse"


# ================================================================ 预审补测（红线①与边界）

def test_bm25_rejects_operator_where_value(tokenizer: PolicyTokenizer) -> None:
    """where 的值是 Chroma 算子时必须报错。

    若静默按"与 dict 相等"比较，结果恒 False，整路返回空——调用方会以为
    "没有匹配条款"，实际是条件没被理解。
    """
    index = BM25Index([make_record("1-0001", "产业园申报")], tokenizer=tokenizer)

    with pytest.raises(NotImplementedError):
        index.search("产业园申报", where={"category": {"$ne": "审计公告"}})


def test_bm25_skips_blank_text_chunk(tokenizer: PolicyTokenizer) -> None:
    """正文分词为空的块不入索引（会让 BM25Okapi 的 avgdl 除零）。"""
    records = [
        make_record("1-0001", "，。；"),
        make_record("1-0002", "产业园申报材料清单"),
    ]

    index = BM25Index(records, tokenizer=tokenizer)

    assert index.chunk_ids == ("1-0002",)


def test_single_route_fusion_score_is_normalized(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """只有一路有结果时权重归一为 1.0，融合分不被打折（否则跨查询不可比）。"""
    # 方向一：稀疏侧空 → 稠密侧权重归一 1.0
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001")])
    # 用一串在中文语料里不可能出现的词，确保稀疏侧返回空
    retriever = HybridRetriever(dense, sparse, HybridConfig(top_k=5, bm25_k=5))

    result = retriever.retrieve("zzzzqqqq")

    assert result.sparse_count == 0
    assert result.hits[0].fusion_score == pytest.approx(1.0 / 61, rel=1e-6)

    # 方向二：稠密侧空 → 稀疏侧权重同样归一 1.0（对称，不能仍按 0.5 打折）
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([])
    retriever = HybridRetriever(dense, sparse, HybridConfig(top_k=5, bm25_k=5))

    result = retriever.retrieve("产业园申报材料清单")

    assert result.dense_count == 0
    assert result.hits[0].fusion_score == pytest.approx(1.0 / 61, rel=1e-6)


def test_fused_hit_delegates_metadata_properties(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """委托属性必须齐全，下游不该为了取效力状态去摸 metadata。"""
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001")])
    result = HybridRetriever(dense, sparse).retrieve("产业园申报")

    hit = result.hits[0]

    assert hit.title.startswith("广东省现代农业产业园")
    assert hit.has_doc_no is True
    assert hit.validity == "有效"
    assert hit.publish_date == "2023-12-16"
    assert hit.serial == 1
    assert hit.clause


def test_sparse_only_hit_serializes_null_dense_score(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """只有稀疏侧命中的块，``score``／``distance`` 必须为 None 而不是 0.0。

    填 0.0 会让下游按 ``score > 0`` 过滤时把混合救回来的块整批丢掉。
    """
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    dense = FakeDense([make_hit("1-0001", 0.7)])
    result = HybridRetriever(dense, sparse, HybridConfig(top_k=5, bm25_k=5)).retrieve(
        "产业园申报材料清单"
    )

    sparse_only = next(hit for hit in result.hits if hit.source == "sparse")
    payload = sparse_only.as_dict()

    assert payload["score"] is None
    assert payload["distance"] is None
    assert payload["dense_score"] is None


def test_hybrid_passes_where_to_sparse_side(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """附加条件必须同时作用于稀疏侧（之前只测了稠密侧）。"""
    dense = FakeDense([])  # 只留稀疏一路，便于断言条件生效
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    retriever = HybridRetriever(dense, sparse, HybridConfig(top_k=5))

    result = retriever.retrieve("产业园申报", where={"doc_no": "粤农农规〔2023〕6号"})

    assert result.hits, "该文号下应有命中"
    assert all(hit.doc_no == "粤农农规〔2023〕6号" for hit in result.hits)


def test_empty_query_reports_actual_temporal_state(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """空查询返回的结果也要如实报告时效开关状态，消融日志不能说是开的。"""
    sparse = BM25Index(hybrid_records, tokenizer=tokenizer)
    retriever = HybridRetriever(
        FakeDense([]), sparse, HybridConfig(enable_temporal_filter=False)
    )

    result = retriever.retrieve("")

    assert result.temporal_filtered is False
    assert "已关" in result.summary()


def test_placeholder_doc_no_citation_is_dropped(
    hybrid_records: list[dict[str, Any]], tokenizer: PolicyTokenizer
) -> None:
    """红线①：占位文号不得经混合层透传进引用串。

    语料里 6267 块（34%）的 registry 文号是 ``无（…）`` 这类占位串，
    必须在 ``Hit`` 层就被挡住（本用例守住混合层不重新引入）。
    """
    record = make_record("1-0009", "产业园申报材料清单")
    record["metadata"].update(
        {"doc_no": "无（申报通知未载文号）", "has_doc_no": False, "clause": "第四条"}
    )
    sparse = BM25Index([record], tokenizer=tokenizer)
    dense = FakeDense([])  # 只走稀疏侧，命中的元数据来自块记录本身
    result = HybridRetriever(dense, sparse).retrieve("产业园申报材料清单")

    assert result.hits[0].citation == ""
    assert "无（" not in result.hits[0].reference
