"""rag.indexer 的单元测试。

对应《开发手册 v1.0》§3.1 PR#3 验收标准：

1. 全量入库可断点续跑——覆盖「指纹相同跳过」「内容变更重写」「全量重建先删后建（含
   删除失败必须报错）」，真实 719 篇全量跑批的输出见 PR 描述；
2. 抽 10 块元数据完整——``test_check_metadata_flags_missing_field`` 与 Chroma 集成用例；
   空集合／块数不足必须判不通过，不能让验收项假绿。

编码器与向量库默认用假实现：CI 不装 sentence-transformers（权重 1.2GB），
装了 chromadb 时真实向量库用例自动启用，未装则跳过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rag.indexer import (
    AUDIT_CATEGORY,
    CONTENT_HASH_KEY,
    REQUIRED_METADATA,
    ChromaStore,
    ChunkRecord,
    IndexConfig,
    build_index,
    check_metadata,
    content_hash,
    dump_chunks,
    format_query,
    load_chunks,
    scalarize_metadata,
    to_record,
)

# ================================================================ 测试替身


class FakeEmbedder:
    """确定性假编码器：记录每次批大小，可配置前若干次调用故意失败。"""

    def __init__(self, dim: int = 4, fail_times: int = 0) -> None:
        self.dim = dim
        self.batches: list[int] = []
        self.queries: list[str] = []
        self._remaining_failures = fail_times

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """返回按序号递增的固定向量，便于断言写入顺序。"""
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("故意失败：模拟编码器异常")
        self.batches.append(len(texts))
        return [[float(index + 1), 0.1, 0.2, 0.3][: self.dim] for index in range(len(texts))]

    def encode_query(self, text: str) -> list[float]:
        """记录查询串并返回固定向量。"""
        self.queries.append(text)
        return [1.0, 0.1, 0.2, 0.3][: self.dim]


class FakeStore:
    """内存假向量库，方法面与 :class:`~rag.indexer.ChromaStore` 对齐。"""

    def __init__(self) -> None:
        self.rows: dict[str, ChunkRecord] = {}
        self.vectors: dict[str, list[float]] = {}
        self.reset_called = 0

    def count(self) -> int:
        """已入库块数。"""
        return len(self.rows)

    def ids(self) -> set[str]:
        """已入库的 chunk_id。"""
        return set(self.rows)

    def hashes(self) -> dict[str, str]:
        """已有块的内容指纹。"""
        return {
            key: str(value.metadata.get(CONTENT_HASH_KEY, ""))
            for key, value in self.rows.items()
        }

    def upsert(self, records: list[ChunkRecord], vectors: list[list[float]]) -> None:
        """按 chunk_id 幂等写入。"""
        assert len(records) == len(vectors), "记录数与向量数必须一致"
        for record, vector in zip(records, vectors, strict=False):
            self.rows[record.chunk_id] = record
            self.vectors[record.chunk_id] = list(vector)

    def sample(self, limit: int = 10, offset: int = 0) -> list[tuple[str, dict[str, Any]]]:
        """取一段元数据。"""
        items = list(self.rows.items())[offset : offset + limit]
        return [(key, dict(value.metadata)) for key, value in items]

    def purge(self, where: dict[str, Any]) -> int:
        """按元数据条件删除。"""
        doomed = [
            key
            for key, value in self.rows.items()
            if all(value.metadata.get(field) == wanted for field, wanted in where.items())
        ]
        for key in doomed:
            self.rows.pop(key, None)
            self.vectors.pop(key, None)
        return len(doomed)

    def reset(self) -> None:
        """清空。"""
        self.rows.clear()
        self.vectors.clear()
        self.reset_called += 1


@dataclass
class ChunkLike:
    """模仿 chunker 的 Chunk：验证鸭子类型取值与 ``as_metadata()`` 约定。"""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        """chunker 侧的真实接口。"""
        return dict(self.metadata)


def make_record(chunk_id: str = "1-0000", **overrides: Any) -> dict[str, Any]:
    """构造一条最小可用的 chunk 记录（文号取自数据仓真实条目）。"""
    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "serial": 1,
        "doc_no": "粤财农〔2020〕106号",
        "title": "广东省涉农资金统筹整合管理办法",
        "structure": "clause",
        "chapter": "第二章 资金使用",
        "section": "",
        "clause": "第十条",
        "clause_index": 10,
        "part": 1,
        "parts_total": 1,
        "category": "资金管理办法",
        "validity": "有效",
        "is_current": True,
        "has_doc_no": True,
        "publish_date": "2020-06-01",
    }
    metadata.update(overrides)
    return {"chunk_id": chunk_id, "text": f"第十条 正文（{chunk_id}）", "metadata": metadata}


# ================================================================ 记录归一化


def test_to_record_from_mapping() -> None:
    """映射形态（JSONL 行）可直接归一化。"""
    record = to_record(make_record("7-0003"))
    assert record.chunk_id == "7-0003"
    assert record.metadata["doc_no"] == "粤财农〔2020〕106号"


def test_to_record_from_chunk_like_object() -> None:
    """chunker 的 Chunk 对象按属性取值，走 ``as_metadata()``。"""
    chunk = ChunkLike(chunk_id="9-0001", text="第九条 正文", metadata={"category": "申报指南"})
    record = to_record(chunk)
    assert record.chunk_id == "9-0001"
    assert record.metadata["category"] == "申报指南"
    assert record.metadata["chunk_id"] == "9-0001"


def test_to_record_rejects_missing_id() -> None:
    """缺 chunk_id 直接拒绝，避免脏数据拿到随机 id。"""
    with pytest.raises(ValueError, match="chunk_id"):
        to_record({"text": "第十条 正文", "metadata": {}})


def test_to_record_rejects_empty_text() -> None:
    """空块不占检索名额。"""
    with pytest.raises(ValueError, match="入库文本为空"):
        to_record({"chunk_id": "1-0000", "text": "   ", "metadata": {}})


def test_metadata_scalarized_to_chroma_types() -> None:
    """None 转空串、非标量丢弃、bool 不当 int 处理。"""
    flattened = scalarize_metadata(
        {
            "clause": None,
            "is_current": False,
            "serial": 12,
            "score": 0.5,
            "keywords": ["帮扶", "资金"],
            "extra": {"a": 1},
        }
    )
    assert flattened == {
        "clause": "",
        "is_current": False,
        "serial": 12,
        "score": 0.5,
    }
    assert isinstance(flattened["is_current"], bool)


# ================================================================ 入库主流程


def test_build_index_in_batches() -> None:
    """按 batch_size 分批编码，块数与向量一一对应。"""
    embedder = FakeEmbedder()
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(5)]

    report = build_index(
        records, embedder=embedder, store=store, config=IndexConfig(batch_size=2)
    )

    assert report.indexed == 5
    assert report.failed == 0
    assert embedder.batches == [2, 2, 1], "应按 batch_size 切批"
    assert store.count() == 5
    assert report.collection_count == 5


def test_resume_skips_indexed_chunks() -> None:
    """断点续跑：重复执行不重复编码已入库块。"""
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(3)]

    first = build_index(records, embedder=FakeEmbedder(), store=store)
    second = build_index(records, embedder=FakeEmbedder(), store=store)

    assert first.indexed == 3
    assert second.indexed == 0
    assert second.skipped_existing == 3
    assert store.count() == 3, "续跑不得产生副本"


def test_no_resume_reindexes_everything() -> None:
    """``resume=False`` 时全量重写（配合 --full-rebuild）。"""
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(3)]

    build_index(records, embedder=FakeEmbedder(), store=store)
    again = build_index(records, embedder=FakeEmbedder(), store=store, resume=False)

    assert again.indexed == 3
    assert again.skipped_existing == 0
    assert store.count() == 3


def test_duplicate_chunk_id_counted_once() -> None:
    """同一批里出现重复 chunk_id 只入一次。"""
    store = FakeStore()
    records = [make_record("1-0000"), make_record("1-0000")]

    report = build_index(records, embedder=FakeEmbedder(), store=store)

    assert report.indexed == 1
    assert report.skipped_existing == 1


def test_audit_category_blocked() -> None:
    """红线④：审计公告一律拒绝入库，且不送去编码。"""
    embedder = FakeEmbedder()
    store = FakeStore()
    records = [
        make_record("1-0000", category=AUDIT_CATEGORY),
        make_record("2-0000"),
    ]

    report = build_index(records, embedder=embedder, store=store)

    assert report.skipped_audit == 1
    assert report.indexed == 1
    assert "1-0000" not in store.rows
    assert embedder.batches == [1], "审计公告不应进入编码环节"


def test_batch_failure_does_not_stop_run() -> None:
    """单批失败只计失败数，后续批次继续。"""
    embedder = FakeEmbedder(fail_times=1)
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(4)]

    report = build_index(
        records, embedder=embedder, store=store, config=IndexConfig(batch_size=2)
    )

    assert report.failed == 2
    assert report.indexed == 2
    assert store.count() == 2


def test_resume_rewrites_changed_text() -> None:
    """同一 chunk_id 正文变了要重写——只看 id 会让库里留着旧条文。"""
    store = FakeStore()
    build_index([make_record("1-0000")], embedder=FakeEmbedder(), store=store)

    revised = make_record("1-0000")
    revised["text"] = "第十条 正文（已修订：新增产业奖补负面情形）"
    report = build_index([revised], embedder=FakeEmbedder(), store=store)

    assert report.indexed == 0
    assert report.updated == 1
    assert store.rows["1-0000"].text == revised["text"]


def test_resume_rewrites_on_validity_change() -> None:
    """效力改判（有效→已失效）必须重写，否则 A6 硬过滤会放行失效条款。"""
    store = FakeStore()
    build_index([make_record("1-0000", is_current=True)], embedder=FakeEmbedder(), store=store)

    report = build_index(
        [make_record("1-0000", is_current=False, validity="已失效")],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert report.updated == 1
    assert store.rows["1-0000"].metadata["is_current"] is False


def test_full_rebuild_path_writes_after_reset() -> None:
    """全量重建：先 reset 再以 ``resume=False`` 写入，块数不得缩水。"""
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(3)]
    build_index(records, embedder=FakeEmbedder(), store=store)

    store.reset()
    report = build_index(records, embedder=FakeEmbedder(), store=store, resume=False)

    assert store.reset_called == 1
    assert report.indexed == 3
    assert report.skipped_existing == 0
    assert store.count() == 3


def test_purges_legacy_audit_chunks() -> None:
    """红线④：库里遗留的审计公告块必须被清出，不能只拦新写入。"""
    store = FakeStore()
    legacy = to_record(make_record("9-0000", category=AUDIT_CATEGORY))
    store.rows[legacy.chunk_id] = legacy  # 模拟历史遗留（绕过 build_index 直接塞进去）

    report = build_index([make_record("1-0000")], embedder=FakeEmbedder(), store=store)

    assert report.purged_audit == 1
    assert "9-0000" not in store.rows
    assert store.count() == 1


def test_content_hash_ignores_self_field() -> None:
    """指纹不得把自身算进去（否则永远对不上，续跑退化成全量重写）。"""
    metadata = {"clause": "第十条", "is_current": True}
    first = content_hash("正文", metadata)
    metadata[CONTENT_HASH_KEY] = "老指纹"
    assert content_hash("正文", metadata) == first
    assert content_hash("正文改动", metadata) != first


def test_metadata_chunk_id_overrides_mapping_value() -> None:
    """映射自带不一致的 chunk_id 时，以顶层为准，避免 id 与元数据对不上。"""
    record = make_record("1-0007")
    record["metadata"]["chunk_id"] = "错误值"
    assert to_record(record).metadata["chunk_id"] == "1-0007"


# ================================================================ 元数据抽检


def test_check_metadata_passes_for_complete_chunks() -> None:
    """10 块元数据齐全 → 抽检通过（验收项）。"""
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(10)]
    build_index(records, embedder=FakeEmbedder(), store=store)

    audit = check_metadata(store, sample=10)

    assert audit.sampled == 10
    assert audit.ok, audit.summary()


def test_check_metadata_flags_missing_field() -> None:
    """缺 ``validity`` 必须被抓出来——时效过滤靠它，缺了就静默失效。"""
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(10)]
    records[0]["metadata"].pop("validity")
    build_index(records, embedder=FakeEmbedder(), store=store)

    audit = check_metadata(store, sample=10)

    assert not audit.ok
    assert any("validity" in problem for problem in audit.problems), audit.summary()


def test_check_metadata_fails_on_empty_collection() -> None:
    """空集合不得判通过——否则全量入库失败后 ``--check`` 会假绿。"""
    audit = check_metadata(FakeStore(), sample=10)

    assert audit.sampled == 0
    assert not audit.ok
    assert "集合为空" in audit.problems[0]


def test_check_metadata_fails_when_too_few_chunks() -> None:
    """块数不足抽检要求时也要报出来（3 块不能冒充「抽 10 块」）。"""
    store = FakeStore()
    build_index(
        [make_record(f"1-{index:04d}") for index in range(3)],
        embedder=FakeEmbedder(),
        store=store,
    )

    audit = check_metadata(store, sample=10)

    assert audit.sampled == 3
    assert not audit.ok
    assert any("不足抽检要求" in problem for problem in audit.problems)


def test_required_metadata_matches_chunker_fields() -> None:
    """必备字段清单不得漏掉 chunker 实际产出的关键字段。"""
    assert set(REQUIRED_METADATA) <= set(make_record()["metadata"])
    assert "is_current" in REQUIRED_METADATA
    assert "publish_date" in REQUIRED_METADATA, "报告②段要标生效日期"


# ================================================================ JSONL


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """dump → load 后内容一致，元数据仍是扁平标量。"""
    target = tmp_path / "chunks.jsonl"
    chunks = [
        ChunkLike(chunk_id="1-0000", text="第十条 正文", metadata={"is_current": True}),
        make_record("2-0001"),
    ]

    written = dump_chunks(chunks, target)
    loaded = list(load_chunks(target))

    assert written == 2
    assert [record.chunk_id for record in loaded] == ["1-0000", "2-0001"]
    assert loaded[0].metadata["is_current"] is True
    first_line = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    assert set(first_line) == {"chunk_id", "text", "metadata"}


def test_load_chunks_skips_broken_line(tmp_path: Path) -> None:
    """坏行跳过，不中断整批。"""
    target = tmp_path / "chunks.jsonl"
    target.write_text(
        json.dumps(make_record("1-0000"), ensure_ascii=False)
        + "\n{坏行\n"
        + json.dumps(make_record("1-0001"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    assert [record.chunk_id for record in load_chunks(target)] == ["1-0000", "1-0001"]


# ================================================================ 查询编码


def test_format_query_follows_model_card() -> None:
    """Qwen3-Embedding 的查询串格式：Instruct 行 + Query: 前缀。"""
    text = format_query("村里想申请产业园建设资金")
    assert text.startswith("Instruct: ")
    assert "\nQuery:村里想申请产业园建设资金" in text


def test_qwen_embedder_is_lazy() -> None:
    """导入与构造不加载权重（CI 没装 sentence-transformers 也能跑到这里）。"""
    from rag.indexer import EmbedderConfig, Qwen3Embedder

    embedder = Qwen3Embedder(EmbedderConfig(model_dir="models/不存在的目录"))
    assert embedder._model is None  # noqa: SLF001 - 断言懒加载状态


# ================================================================ 命令行


def test_main_requires_a_source() -> None:
    """既不给 --chunks 也不给 --from-corpus 时要直接报错退出，不去碰向量库。"""
    from rag.indexer import main

    with pytest.raises(SystemExit):
        main([])


def test_load_config_absent_returns_defaults(tmp_path: Path) -> None:
    """配置文件不存在时返回空配置，全部走默认值（不静默用错路径）。"""
    from rag.indexer import _load_config

    assert _load_config(tmp_path / "没有这个文件.json") == {}


# ================================================================ Chroma 集成


def test_chroma_store_roundtrip(tmp_path: Path) -> None:
    """真实 Chroma：写入、where 时效过滤、抽检、重置。"""
    chromadb = pytest.importorskip("chromadb")
    config = IndexConfig(persist_dir=tmp_path / "chroma", collection="test_clauses")
    store = ChromaStore(config, client=chromadb.EphemeralClient())
    records = [
        make_record("1-0000", is_current=True),
        make_record("1-0001", is_current=False, validity="已失效"),
        make_record("1-0002", is_current=True),
    ]

    report = build_index(records, embedder=FakeEmbedder(), store=store, config=config)

    assert report.indexed == 3
    assert store.count() == 3
    assert store.ids() == {"1-0000", "1-0001", "1-0002"}

    current = store.collection.get(where={"is_current": True}, include=["metadatas"])
    assert set(current["ids"]) == {"1-0000", "1-0002"}, "A6 时效硬过滤依赖 is_current 元数据"
    assert check_metadata(store, sample=3).ok

    store.reset()
    assert store.count() == 0


def test_chroma_hashes_paginate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``hashes()`` 必须分页读完——18262 块的全量续跑全靠这条路径。"""
    chromadb = pytest.importorskip("chromadb")
    monkeypatch.setattr(ChromaStore, "PAGE_SIZE", 2)
    config = IndexConfig(persist_dir=tmp_path / "chroma", collection="paging")
    store = ChromaStore(config, client=chromadb.EphemeralClient())
    records = [make_record(f"1-{index:04d}") for index in range(5)]
    build_index(records, embedder=FakeEmbedder(), store=store, config=config)

    hashes = store.hashes()

    assert set(hashes) == {record["chunk_id"] for record in records}
    assert all(len(value) == 16 for value in hashes.values()), "每块都要有 16 位指纹"


def test_chroma_reset_raises_when_not_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """删除失败必须报错——静默成功会把「全量重建」降级成「全部跳过」。"""
    chromadb = pytest.importorskip("chromadb")
    config = IndexConfig(persist_dir=tmp_path / "chroma", collection="stuck")
    store = ChromaStore(config, client=chromadb.EphemeralClient())
    build_index(
        [make_record("1-0000")], embedder=FakeEmbedder(), store=store, config=config
    )

    monkeypatch.setattr(store.client, "delete_collection", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="全量重建失败"):
        store.reset()


def test_chroma_purge_removes_audit_chunks(tmp_path: Path) -> None:
    """真实 Chroma 上按 category 删除生效（红线④的 where 条件）。"""
    chromadb = pytest.importorskip("chromadb")
    config = IndexConfig(persist_dir=tmp_path / "chroma", collection="audit_purge")
    store = ChromaStore(config, client=chromadb.EphemeralClient())
    legacy = to_record(make_record("9-0000", category=AUDIT_CATEGORY))
    store.upsert([legacy], [[0.1, 0.2, 0.3, 0.4]])

    removed = store.purge({"category": AUDIT_CATEGORY})

    assert removed == 1
    assert store.count() == 0
