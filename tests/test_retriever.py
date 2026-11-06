"""rag.retriever 的单元测试。

对应《开发手册 v1.0》§3.1 PR#4 验收标准：

1. **查"产业园申报"只回现行有效条款**——``test_stale_chunks_do_not_take_top_k_slots``
   用一个刻意构造的场景验证：失效块的相似度全都高于现行块，若实现退化成"取 top-k 再
   后置筛"，结果会是空；只有把条件前置进 where 才拿得到现行条款。
2. **失效文件被过滤有测试**——同一个用例断言返回块全部 ``is_current=True``，另有
   ``test_temporal_filter_can_be_disabled`` 证明开关关掉后失效块确实会回来（消融可观测）。

编码器与后端默认用假实现：CI 不装 sentence-transformers／chromadb 也能跑；
装了 chromadb 时真实集合用例自动启用。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from rag.retriever import (
    AUDIT_CATEGORY,
    ChromaBackend,
    DenseRetriever,
    Hit,
    RetrievalConfig,
    RetrievalResult,
    build_where,
    is_retrievable,
)

# ================================================================ 测试替身


class FakeEmbedder:
    """记录查询串的假编码器。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.queries: list[str] = []

    def encode_query(self, text: str) -> list[float]:
        """记录查询并返回固定向量（后端替身不真的算相似度）。"""
        self.queries.append(text)
        return [0.5] * self.dim


def _match(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """极简 where 求值：只支持本模块会用到的等值与 ``$ne``／``$and``。

    故意写得"笨"但直白——替身要按条件真过滤，否则测不出条件是前置还是后置。
    """
    if not where:
        return True
    for key, expected in where.items():
        if key == "$and":
            return all(_match(metadata, clause) for clause in expected)
        if isinstance(expected, dict):
            if "$ne" in expected and metadata.get(key) == expected["$ne"]:
                return False
            continue
        if metadata.get(key) != expected:
            return False
    return True


class FakeBackend:
    """内存后端：按 where 过滤 → 按 distance 升序 → 取前 top_k。"""

    def __init__(self, rows: list[tuple[str, str, dict[str, Any], float]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    def search(
        self, vector: list[float], top_k: int, where: dict[str, Any]
    ) -> list[tuple[str, str, dict[str, Any], float]]:
        """先过滤再排序——模仿真实向量库的语义。"""
        self.calls.append({"top_k": top_k, "where": dict(where)})
        matched = [row for row in self.rows if _match(row[2], where)]
        matched.sort(key=lambda row: row[3])
        return matched[:top_k]

    def count(self, where: dict[str, Any] | None = None) -> int:
        """满足条件的候选总数。"""
        self.count_calls.append(dict(where or {}))
        return len([row for row in self.rows if _match(row[2], where or {})])


def make_row(
    chunk_id: str,
    distance: float,
    *,
    is_current: bool = True,
    category: str = "财政专项",
    doc_no: str = "粤财农〔2020〕106号",
    clause: str = "第十二条",
    title: str = "关于印发乡村振兴资金管理办法的通知",
) -> tuple[str, str, dict[str, Any], float]:
    """造一条后端行：``(chunk_id, text, metadata, distance)``。"""
    metadata = {
        "chunk_id": chunk_id,
        "doc_no": doc_no,
        "title": title,
        "clause": clause,
        "category": category,
        "is_current": is_current,
        "serial": 106,
        "is_preamble": False,
        "is_appendix": False,
    }
    return (
        chunk_id,
        f"{clause} 产业园建设与基础设施配套补助，不得用于偿还债务。",
        metadata, distance)


# ================================================================ where 构造


def test_build_where_enables_temporal_filter_by_default() -> None:
    """默认配置必须带 ``is_current`` 条件（A6 默认开）。"""
    where = build_where()
    assert {"is_current": True} in where["$and"]


def test_build_where_single_clause_not_wrapped() -> None:
    """只有一条条件时不包 ``$and``——Chroma 对单键 where 更省事。"""
    config = RetrievalConfig(exclude_categories=())
    assert build_where(config) == {"is_current": True}


def test_build_where_empty_when_everything_off() -> None:
    """全部关掉时返回空字典，表示不过滤。"""
    config = RetrievalConfig(enable_temporal_filter=False, exclude_categories=())
    assert build_where(config) == {}


def test_build_where_merges_overrides() -> None:
    """调用方的额外条件与时效条件取交集，而不是覆盖。"""
    where = build_where(RetrievalConfig(), {"doc_no": "河财农〔2020〕82号"})
    assert {"is_current": True} in where["$and"]
    assert {"doc_no": "河财农〔2020〕82号"} in where["$and"]


def test_build_where_excludes_audit_category() -> None:
    """红线④：默认排除审计公告。"""
    assert {"category": {"$ne": AUDIT_CATEGORY}} in build_where()["$and"]


# ================================================================ 时效硬过滤（核心）


def test_temporal_filter_is_pushed_into_where() -> None:
    """条件必须下发给后端，而不是拿回结果后自己筛。"""
    backend = FakeBackend([make_row("1-0000", 0.2)])
    retriever = DenseRetriever(FakeEmbedder(), backend)

    retriever.retrieve("产业园申报条件")

    assert backend.calls, "必须调用后端检索"
    sent = backend.calls[0]["where"]
    assert {"is_current": True} in sent["$and"], "时效条件没有前置到 where，退化成后置过滤了"


def test_stale_chunks_do_not_take_top_k_slots() -> None:
    """验收主用例：失效块更相似时，也不得挤占 top-k 名额。

    构造 2 条现行（distance 0.5/0.6）+ 6 条已失效（distance 0.1~0.35，全都更相似），
    ``top_k=3``。前置过滤的正确结果是 2 条现行块；若有人把它写回"取 top3 再筛"，
    结果会是 0 条——这正是要防的召回塌陷。
    """
    rows = [
        make_row("1-0101", 0.10, is_current=False),
        make_row("1-0102", 0.15, is_current=False),
        make_row("1-0103", 0.20, is_current=False),
        make_row("1-0104", 0.25, is_current=False),
        make_row("1-0105", 0.30, is_current=False),
        make_row("1-0106", 0.35, is_current=False),
        make_row("1-0001", 0.50, is_current=True),
        make_row("1-0002", 0.60, is_current=True),
    ]
    backend = FakeBackend(rows)
    retriever = DenseRetriever(FakeEmbedder(), backend)

    result = retriever.retrieve("产业园申报", top_k=3)

    assert {hit.chunk_id for hit in result.hits} == {"1-0001", "1-0002"}
    assert all(hit.is_current for hit in result.hits), "结果里不得出现已失效条款"
    assert result.temporal_filtered


def test_temporal_filter_can_be_disabled() -> None:
    """消融开关可观测：关掉后失效块确实会回来（否则开关是假的）。"""
    rows = [
        make_row("1-0101", 0.10, is_current=False),
        make_row("1-0001", 0.50, is_current=True),
    ]
    backend = FakeBackend(rows)
    config = RetrievalConfig(enable_temporal_filter=False)
    retriever = DenseRetriever(FakeEmbedder(), backend, config)

    result = retriever.retrieve("产业园申报", top_k=2)

    assert len(result.hits) == 2
    assert result.hits[0].chunk_id == "1-0101", "关掉过滤后最相似的失效块应当排在首位"
    assert not result.temporal_filtered


def test_audit_chunks_never_returned() -> None:
    """红线④：即便集合里混着审计公告，检索端也要挡住。"""
    rows = [
        make_row("9-0001", 0.10, category=AUDIT_CATEGORY),
        make_row("1-0001", 0.50),
    ]
    backend = FakeBackend(rows)
    retriever = DenseRetriever(FakeEmbedder(), backend)

    result = retriever.retrieve("产业园申报", top_k=5)

    assert [hit.chunk_id for hit in result.hits] == ["1-0001"]


def test_audit_chunk_filtered_even_if_backend_ignores_where() -> None:
    """后端若忽略 where（旧索引/异常实现），检索端仍要再挡一次。"""

    class SloppyBackend:
        """故意无视 where 的后端，模拟"集合不干净且后端不可靠"。"""

        def search(self, vector: list[float], top_k: int, where: dict[str, Any]) -> list:
            return [make_row("9-0001", 0.1, category=AUDIT_CATEGORY)]

        def count(self, where: dict[str, Any] | None = None) -> int:
            return 1

    result = DenseRetriever(FakeEmbedder(), SloppyBackend()).retrieve("产业园申报")

    assert result.hits == []


# ================================================================ 排序与数值


def test_scores_are_similarity_descending() -> None:
    """对外暴露的是相似度（1 - distance），且按降序排列。"""
    backend = FakeBackend([make_row("1-0001", 0.50), make_row("1-0002", 0.10)])
    result = DenseRetriever(FakeEmbedder(), backend).retrieve("产业园申报")

    assert [hit.chunk_id for hit in result.hits] == ["1-0002", "1-0001"]
    assert result.hits[0].score == pytest.approx(0.9)
    assert result.hits[0].distance == pytest.approx(0.1)


def test_top_k_override_beats_config() -> None:
    """显式 top_k 覆盖配置值（重排阶段要 k40，不能只吃配置里的 10）。"""
    rows = [make_row(f"1-00{index:02d}", 0.1 * index) for index in range(1, 8)]
    backend = FakeBackend(rows)
    retriever = DenseRetriever(FakeEmbedder(), backend, RetrievalConfig(top_k=2))

    assert len(retriever.retrieve("产业园").hits) == 2, "配置 top_k=2 生效"
    assert len(retriever.retrieve("产业园", top_k=5).hits) == 5
    assert backend.calls[-1]["top_k"] == 5


def test_candidate_count_uses_same_where() -> None:
    """诊断计数必须用同一套条件，否则"候选池 0 但命中 3 条"这种鬼话会出现。"""
    rows = [make_row("1-0001", 0.5, is_current=True), make_row("1-0101", 0.1, is_current=False)]
    backend = FakeBackend(rows)
    result = DenseRetriever(FakeEmbedder(), backend).retrieve("产业园", count_candidates=True)

    assert result.candidates == 1
    assert backend.count_calls[0] == backend.calls[0]["where"]


def test_count_is_off_by_default() -> None:
    """默认不做候选池统计：它要走一次全量 get，放主路径上会拖慢每次检索。"""
    backend = FakeBackend([make_row("1-0001", 0.5)])
    result = DenseRetriever(FakeEmbedder(), backend).retrieve("产业园")

    assert result.candidates is None
    assert backend.count_calls == [], "默认不该调用 count"


# ================================================================ 引用与边界


def test_citation_only_uses_metadata() -> None:
    """红线②：引用串只拼元数据里已有的字段，缺一段就是空，不补不猜。"""
    hit = Hit("1-0001", "正文", 0.9, 0.1, {"doc_no": "粤财农〔2020〕106号", "clause": "第十二条"})
    assert hit.citation == "粤财农〔2020〕106号 第十二条"

    no_clause = Hit("1-0002", "正文", 0.9, 0.1, {"doc_no": "粤财农〔2020〕106号", "clause": ""})
    assert no_clause.citation == "粤财农〔2020〕106号"

    bare = Hit("1-0003", "正文", 0.9, 0.1, {})
    assert bare.citation == "", "没有任何元数据时必须为空，不能编一个出来"


def test_citation_empty_when_doc_no_missing() -> None:
    """没有文号时引用串必须为空，**不能退化成"只有条款号"**。

    全库 6267 块（34%）来自无文号文件（函件、内部通知）。若 citation 给"第十二条"
    这种半截引用，校验器拿去 registry 反查必然查不到，等于把查不动的依据混进报告。
    """
    hit = Hit(
        "1-0001", "正文", 0.9, 0.1,
        {"doc_no": "", "clause": "第十二条", "title": "某内部通知"})

    assert hit.citation == "", "缺文号时严格引用串必须为空"
    assert not hit.has_doc_no


def test_reference_falls_back_to_title() -> None:
    """展示用引用要有兜底：没文号就用标题，读者仍能定位到文件。"""
    hit = Hit(
        "1-0001", "正文", 0.9, 0.1,
        {"doc_no": "", "clause": "第十二条", "title": "某内部通知"})

    assert hit.reference == "某内部通知 第十二条"
    assert hit.has_doc_no is False

    with_doc = Hit(
        "1-0002", "正文", 0.9, 0.1,
        {"doc_no": "粤财农〔2020〕106号", "clause": "第一条"})
    assert with_doc.reference == "粤财农〔2020〕106号 第一条"


def test_references_cover_hits_without_doc_no() -> None:
    """没有文号的命中不会出现在 citations 里，但会出现在 references 里。"""
    result = RetrievalResult(
        query="产业园",
        hits=[
            Hit("1-0001", "正文", 0.9, 0.1, {"doc_no": "A号", "clause": "第一条"}),
            Hit("1-0002", "正文", 0.8, 0.2, {"doc_no": "", "title": "某通知", "clause": "第二条"}),
        ],
    )

    assert result.citations() == ["A号 第一条"]
    assert result.references() == ["A号 第一条", "某通知 第二条"]
    assert result.no_doc_no_count() == 1


def test_hit_exposes_validity_and_publish_date() -> None:
    """A8 要求文号/条款号/日期都由元数据注入，故 Hit 要能直接取到。"""
    hit = Hit(
        "1-0001", "正文", 0.9, 0.1,
        {"doc_no": "A号", "clause": "第一条", "publish_date": "2023-12-05", "validity": "有效"},
    )

    assert hit.publish_date == "2023-12-05"
    assert hit.validity == "有效"


def test_is_current_requires_real_true() -> None:
    """``is_current`` 用 ``is True`` 判定：字符串 "false" 也是字符串，不能当现行。"""
    assert Hit("1-0001", "t", 0.9, 0.1, {"is_current": True}).is_current
    assert not Hit("1-0002", "t", 0.9, 0.1, {"is_current": False}).is_current
    assert not Hit("1-0003", "t", 0.9, 0.1, {"is_current": "false"}).is_current
    assert not Hit("1-0004", "t", 0.9, 0.1, {}).is_current


def test_citations_deduplicated_and_ordered() -> None:
    """同一文件的多个条款，授权串里按出现顺序去重。"""
    result = RetrievalResult(
        query="产业园",
        hits=[
            Hit("1-0001", "正文", 0.9, 0.1, {"doc_no": "A号", "clause": "第一条"}),
            Hit("1-0002", "正文", 0.8, 0.2, {"doc_no": "A号", "clause": "第二条"}),
            Hit("1-0003", "正文", 0.7, 0.3, {"doc_no": "B号", "clause": "第一条"}),
        ],
    )

    assert result.citations() == ["A号 第一条", "A号 第二条", "B号 第一条"]
    assert result.doc_nos() == ["A号", "B号"]


def test_empty_query_returns_empty_without_error() -> None:
    """空查询不该打后端，也不该抛异常。"""
    backend = FakeBackend([make_row("1-0001", 0.5)])
    result = DenseRetriever(FakeEmbedder(), backend).retrieve("   ")

    assert result.hits == []
    assert backend.calls == []


def test_no_hits_when_pool_empty() -> None:
    """候选池为空时返回空结果，summary 仍可打印。"""
    backend = FakeBackend([make_row("1-0101", 0.1, is_current=False)])
    result = DenseRetriever(FakeEmbedder(), backend).retrieve("产业园", count_candidates=True)

    assert result.hits == []
    assert result.candidates == 0
    assert "命中 0 块" in result.summary()


def test_retrieve_many_keeps_order() -> None:
    """批量检索逐条返回，顺序与入参一致。"""
    backend = FakeBackend([make_row("1-0001", 0.5)])
    retriever = DenseRetriever(FakeEmbedder(), backend)

    results = retriever.retrieve_many(["第一个问题", "第二个问题"])

    assert [item.query for item in results] == ["第一个问题", "第二个问题"]


def test_query_is_sent_verbatim_to_embedder() -> None:
    """查询前不擅自加指令前缀——指令由编码器负责，避免叠加两次。"""
    embedder = FakeEmbedder()
    DenseRetriever(embedder, FakeBackend([make_row("1-0001", 0.5)])).retrieve("产业园申报条件")

    assert embedder.queries == ["产业园申报条件"]


# ================================================================ 真实 Chroma 集成


def test_chroma_backend_end_to_end(tmp_path: Path) -> None:
    """真实 Chroma：入库三块（两现行一失效），检索只回现行。"""
    chromadb = pytest.importorskip("chromadb")
    from rag.indexer import ChromaStore, IndexConfig, build_index

    config = IndexConfig(persist_dir=tmp_path / "chroma", collection="test_retriever")
    store = ChromaStore(config, client=chromadb.EphemeralClient())

    class VecEmbedder:
        """把关键词映射到固定向量：含"失效"的落在另一侧，便于构造可分辨的距离。"""

        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            return [
                [1.0, 0.0, 0.0, 0.1] if "失效" not in text else [0.0, 1.0, 0.0, 0.1]
                for text in texts
            ]

        def encode_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.1]

    records = [
        {"chunk_id": "1-0001", "text": "现行：产业园建设补助可以申报基础设施配套。",
         "metadata": {"doc_no": "粤财农〔2020〕106号", "clause": "第十二条",
                      "title": "资金管理办法", "category": "财政专项", "is_current": True,
                      "serial": 106, "is_preamble": False, "is_appendix": False}},
        {"chunk_id": "1-0002", "text": "现行：产业园奖补资金按投资额分档。",
         "metadata": {"doc_no": "河财农〔2020〕82号", "clause": "第八条",
                      "title": "产业奖补细则", "category": "财政专项", "is_current": True,
                      "serial": 82, "is_preamble": False, "is_appendix": False}},
        {"chunk_id": "1-0003", "text": "失效：旧版产业园补助标准已废止。",
         "metadata": {"doc_no": "粤财农〔2016〕9号", "clause": "第五条",
                      "title": "旧补助标准", "category": "财政专项", "is_current": False,
                      "serial": 9, "is_preamble": False, "is_appendix": False}},
    ]
    build_index(records, embedder=VecEmbedder(), store=store, config=config)

    retriever = DenseRetriever(VecEmbedder(), ChromaBackend(store.collection))
    result = retriever.retrieve("产业园申报", top_k=3, count_candidates=True)

    assert {hit.chunk_id for hit in result.hits} == {"1-0001", "1-0002"}
    assert all(hit.is_current for hit in result.hits)
    assert result.candidates == 2, "候选池里就不该有失效块"
    assert "粤财农〔2016〕9号" not in result.doc_nos()


def test_chroma_backend_respects_doc_no_override(tmp_path: Path) -> None:
    """额外条件（限定文号）与时效条件并用，两者都要生效。"""
    chromadb = pytest.importorskip("chromadb")
    from rag.indexer import ChromaStore, IndexConfig, build_index

    config = IndexConfig(persist_dir=tmp_path / "chroma2", collection="test_retriever_filter")
    store = ChromaStore(config, client=chromadb.EphemeralClient())

    class FlatEmbedder:
        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def encode_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    records = [
        {"chunk_id": f"1-{index:04d}", "text": f"产业园补助条款 {index}",
         "metadata": {"doc_no": doc_no, "clause": "第一条", "title": "t",
                      "category": "财政专项", "is_current": True, "serial": index,
                      "is_preamble": False, "is_appendix": False}}
        for index, doc_no in enumerate(["粤财农〔2020〕106号", "河财农〔2020〕82号"], start=1)
    ]
    build_index(records, embedder=FlatEmbedder(), store=store, config=config)

    retriever = DenseRetriever(FlatEmbedder(), ChromaBackend(store.collection))
    result = retriever.retrieve("产业园", where={"doc_no": "河财农〔2020〕82号"})

    assert [hit.doc_no for hit in result.hits] == ["河财农〔2020〕82号"]


# ================================================================ 红线④的 fail-closed 判定


def test_is_retrievable_accepts_normal_category() -> None:
    """正常类别放行。"""
    assert is_retrievable({"category": "资金管理办法"})
    assert is_retrievable({"category": "负面清单"})


def test_is_retrievable_rejects_audit_category() -> None:
    """审计公告一律不放行（红线④）。"""
    assert not is_retrievable({"category": AUDIT_CATEGORY})


def test_is_retrievable_rejects_missing_category() -> None:
    """类别缺失／空串一律不放行——fail-closed。

    为什么不是 fail-open：Chroma 的 ``$ne`` 会把"缺该字段"判为满足条件，所以
    向量库那一层挡不住类别未知的块；若后置兜底也放行，红线④就漏了。
    """
    assert not is_retrievable({})
    assert not is_retrievable({"category": ""})
    assert not is_retrievable({"category": "   "})
    assert not is_retrievable({"category": None})


def test_chunk_without_category_is_dropped_by_retriever() -> None:
    """后端把类别缺失的块返回上来时，检索端必须丢掉它。"""
    rows = [
        ("9-0001", "类别缺失的可疑块", {"chunk_id": "9-0001", "is_current": True}, 0.05),
        make_row("1-0001", 0.50),
    ]
    backend = FakeBackend(rows)

    result = DenseRetriever(FakeEmbedder(), backend).retrieve("产业园申报", top_k=5)

    assert [hit.chunk_id for hit in result.hits] == ["1-0001"]


def test_chroma_ne_where_lets_missing_category_through(tmp_path: Path) -> None:
    """复现 ``$ne`` 的语义陷阱，并证明检索端的 fail-closed 兜底是必要的。

    Chroma 对 ``{"category": {"$ne": "审计公告"}}`` 会**放行缺少 category 的块**
    （实测 chromadb 1.5.9）。所以只靠 where 不足以守住红线④：一条没有类别字段的
    审计公告块会从向量库那一层漏出来，必须靠 :func:`is_retrievable` 在结果侧再挡。
    """
    chromadb = pytest.importorskip("chromadb")
    from rag.indexer import ChromaStore, IndexConfig

    config = IndexConfig(persist_dir=tmp_path / "chroma3", collection="test_ne_trap")
    store = ChromaStore(config, client=chromadb.EphemeralClient())
    collection = store.collection
    # 故意绕开 build_index（它会拦审计公告）直接写一条"没有 category 字段"的块，
    # 模拟旧索引／手工导入的脏数据
    collection.upsert(
        ids=["9-0001"],
        embeddings=[[1.0, 0.0]],
        documents=["（旧索引遗留）审计口径说明"],
        metadatas=[{"chunk_id": "9-0001", "is_current": True, "doc_no": ""}],
    )

    class FlatEmbedder:
        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def encode_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    backend = ChromaBackend(collection)

    # 1) 向量库那一层：$ne 条件确实放行了这条脏块
    leaked = backend.search([1.0, 0.0], 5, build_where(RetrievalConfig()))
    assert [row[0] for row in leaked] == ["9-0001"], \
        "若这里不再放行，说明 chromadb 语义变了，可简化兜底"

    # 2) 检索端兜底：fail-closed 把它拦住
    result = DenseRetriever(FlatEmbedder(), backend).retrieve("审计口径")
    assert result.hits == [], "类别缺失的块不得进入检索结果"


# ================================================================ 占位文号不得混进引用（红线①）

REAL_DOC_NO = re.compile(r"〔\d{4}〕\d+号")


def test_has_doc_no_prefers_metadata_flag() -> None:
    """metadata 的 ``has_doc_no`` 是 loader 用正则算出的权威结论，优先采信。"""
    placeholder = Hit(
        "1-0001",
        "正文",
        0.9,
        0.1,
        {
            "doc_no": "无（申报通知未载文号）",
            "clause": "第四条",
            "title": "关于组织申报的通知",
            "has_doc_no": False,
        },
    )

    assert not placeholder.has_doc_no
    assert placeholder.citation == "", "占位串不得进入引用位"
    assert placeholder.reference == "关于组织申报的通知 第四条", "展示时退回标题"


def test_has_doc_no_falls_back_to_placeholder_regex() -> None:
    """字段缺失时（旧索引／手工导入）用正则兜底，仍然挡住占位串。"""
    legacy = Hit(
        "1-0002",
        "正文",
        0.9,
        0.1,
        {"doc_no": "无（页面未载/仅正文引用，不计）", "clause": "第四条"},
    )

    assert not legacy.has_doc_no
    assert legacy.citation == ""


def test_has_doc_no_true_for_real_doc_no() -> None:
    """真文号放行（``〔年份〕序号号`` 形态）。"""
    hit = Hit("1-0003", "正文", 0.9, 0.1, {"doc_no": "粤农农规〔2023〕6号", "clause": "第十条"})

    assert hit.has_doc_no
    assert hit.citation == "粤农农规〔2023〕6号 第十条"


def test_metadata_flag_overrides_misleading_doc_no() -> None:
    """字段说没有，就算 doc_no 长得很正常也不认——以 loader 的判定为准。"""
    hit = Hit(
        "1-0004",
        "正文",
        0.9,
        0.1,
        {"doc_no": "某某文件", "clause": "第一条", "has_doc_no": False},
    )

    assert not hit.has_doc_no
    assert hit.citation == ""
