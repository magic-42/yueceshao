"""粤策哨检索管线（rag）。

模块划分与 PR 拆分见《粤策哨开发手册 v1.0》§3.1：

- loader    —— 读数据仓 registry.json → 标准化 DocRecord，过滤审计公告（已落地）
- chunker   —— 结构感知条款分块，层级头注入，条款级溯源（已落地）
- indexer   —— Chroma 入库与 embedding 封装（待开发）
- retriever —— 稠密检索＋时效硬过滤（已落地）
- hybrid    —— BM25＋RRF 混合检索（待开发）
- rerank    —— Cross-Encoder 重排＋质量门（待开发）
"""

from rag.chunker import (
    Chunk,
    ChunkConfig,
    check_clause_continuity,
    chunk_corpus,
    chunk_document,
    clause_sequence,
)
from rag.loader import DocRecord, LoadResult, load_registry

__all__ = [
    "Chunk",
    "ChunkConfig",
    "DocRecord",
    "LoadResult",
    "check_clause_continuity",
    "chunk_corpus",
    "chunk_document",
    "clause_sequence",
    "load_registry",
]
