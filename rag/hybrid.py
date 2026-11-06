"""BM25 与稠密检索的 RRF 融合（《开发手册 v1.0》A5，PR#5）。

职责
----
把「稠密召回」（:mod:`rag.retriever`）与「稀疏召回」（BM25＋jieba＋政策术语词典）
两路结果按 Reciprocal Rank Fusion 融合成一份排序，供下游重排/生成使用。

为什么要混合
------------
稠密检索擅长语义相近，但**对政策专有名词不敏感**——「联农带农」「负面清单」
「项目库」这类词在语料里是固定搭配，用户查的也往往是这些原词，而向量模型会把
它们和语义相邻但不相干的表述混在一起。BM25 恰好相反：原词命中就高分，但换个
说法（"资金能花在哪" vs "支出范围"）就召回不到。两路互补是 A5 的出发点。

三个必须说明的设计决策
----------------------
1. **稀疏侧也要做时效硬过滤**。BM25 是内存索引，没有 Chroma 的 ``where``，
   若只指望稠密侧过滤，失效条款会从稀疏一路漏进融合结果——A6 在混合架构下最
   容易漏的就是这一处。所以本模块在打分前先按 ``is_current`` 与类别做 mask。
   注意顺序：**先对全部块打分、再 mask、最后截断 top-k**，不是"先取 top-k 再筛"
   ——后者同样会塌陷名额。
2. **融合只用排名，不用分数**。BM25 分数与余弦相似度量纲不同（前者无上界、
   随查询长度漂移），加权求和的权重毫无物理意义。RRF 只吃排名，天然免疫这件事，
   这也是 A5 直接给 RRF 而不是加权求和的原因。
3. **两路候选池各自独立取**，融合前不互相裁剪。稠密侧取 ``dense_k``、稀疏侧取
   ``bm25_k``，两边都远大于最终 ``top_k``，才能让"只在某一路排得高"的块有救回的
   机会——这正是混合检索的收益来源。

红线约束
--------
- 不产生任何文号：正文与元数据全部来自块记录，本模块不构造、不拼接、不改写。
- 审计公告拦截三次：稠密侧（:func:`rag.retriever.is_retrievable`）、稀疏侧
  （同一判定做 mask）、融合侧（再次校验）。三条路径任一漏掉都会让评测出卷集
  混进结果。
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from rag.retriever import CURRENT_FIELD, Hit, RetrievalConfig, is_retrievable

logger = logging.getLogger(__name__)

__all__ = [
    "POLICY_TERMS_PATH",
    "PolicyTokenizer",
    "BM25Index",
    "rrf_fuse",
    "HybridConfig",
    "FusedHit",
    "HybridResult",
    "HybridRetriever",
    "build_hybrid_retriever",
]


POLICY_TERMS_PATH = Path(__file__).with_name("resources") / "policy_terms.txt"
"""政策术语词典路径；由 :class:`PolicyTokenizer` 加载进 jieba。"""


# ================================================================ 分词


STOPWORDS: frozenset[str] = frozenset(
    """
    的 了 和 与 及 或 在 是 为 对 等 有 就 都 而 但 也 把 被 让 给 从 到 由 于
    这个 那个 这些 那些 一个 一些 什么 哪些 哪个 怎么 怎样 如何 多少 是否
    可以 能否 需要 应该 必须 我们 他们 你们 自己 以及 并且 或者 还是
    吗 呢 吧 啊 呀 哦 之 其 该 此 本 上述 下列 有关 相关
    """.split()
)
"""查询与语料两侧共用的停用词。

两侧必须用同一套：BM25 的 IDF 建立在语料分词之上，若查询侧额外滤词而语料侧不滤，
词频与文档频率就对不上（虽然 rank_bm25 不会报错，但打分口径已经不一致了）。
只收虚词与疑问词，不动任何专业名词。
"""


def load_policy_terms(path: Path | str | None = None) -> tuple[str, ...]:
    """读取政策术语词典。

    :param path: 词典路径，默认 :data:`POLICY_TERMS_PATH`
    :return: 术语元组（去重保序）；文件不存在时返回空元组并告警，
        不抛异常——词典缺失只会让分词退化为通用切分，不该让整条检索链挂掉
    """
    target = Path(path) if path else POLICY_TERMS_PATH
    if not target.is_file():
        logger.warning("政策术语词典不存在，按 jieba 默认词典分词：%s", target)
        return ()
    terms: list[str] = []
    seen: set[str] = set()
    for line in target.read_text(encoding="utf-8-sig").splitlines():
        term = line.strip()
        if not term or term.startswith("#") or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    logger.debug("政策术语词典加载 %d 条：%s", len(terms), target)
    return tuple(terms)


class PolicyTokenizer:
    """jieba 分词＋政策术语词典＋停用词过滤。

    为什么必须加词典：jieba 默认词典对县域帮扶术语切得很碎——「联农带农」→
    「联农/带农」、「驻镇帮镇扶村」→「驻镇/帮镇/扶村」、「强村公司」→「强村/公司」。
    切碎后每个片段都很常见（IDF 低），BM25 无法把它们当作整体去匹配用户在提问里
    写的同一个词，稀疏召回直接失效。

    :param extra_terms: 追加术语（测评时补充召回用）
    :param terms_path: 词典文件路径，默认 :data:`POLICY_TERMS_PATH`
    :param drop_stopwords: 是否过滤 :data:`STOPWORDS`
    """

    def __init__(
        self,
        extra_terms: Iterable[str] = (),
        terms_path: Path | str | None = None,
        *,
        drop_stopwords: bool = True,
    ) -> None:
        import jieba  # 局部 import：装依赖即用，不影响只需类型检查的场景

        self._jieba = jieba
        self.drop_stopwords = drop_stopwords
        terms = list(load_policy_terms(terms_path)) + [
            term for term in extra_terms if term
        ]
        self.terms: tuple[str, ...] = tuple(dict.fromkeys(terms))
        for term in self.terms:
            # freq 交给 jieba 自己算（suggest_freq）：给死值要么压不过更长词，
            # 要么把 total 抬得过高、扭曲其他词的词频占比。
            self._jieba.add_word(term, freq=None)

    def cut(self, text: str) -> list[str]:
        """切词并去除空白/标点/停用词。

        :param text: 原文
        :return: 词元列表；空输入返回空列表
        """
        if not text:
            return []
        tokens = []
        for token in self._jieba.lcut(text):
            token = token.strip()
            if not token:
                continue
            if len(token) == 1 and not token.isalnum():
                continue  # 单字标点
            if self.drop_stopwords and token in STOPWORDS:
                continue
            tokens.append(token.lower())
        return tokens


# ================================================================ 稀疏索引


class _RecordLike(Protocol):
    """BM25 只要求块有 id／正文／元数据三样。"""

    text: str
    metadata: Mapping[str, Any]


def _chunk_id_of(record: Any) -> str:
    """从块对象／映射里取 chunk_id。"""
    value = getattr(record, "chunk_id", None)
    if isinstance(value, str) and value:
        return value
    if isinstance(record, Mapping):
        raw = record.get("chunk_id") or record.get("id")
        if isinstance(raw, str) and raw:
            return raw
    raise ValueError("块记录缺少 chunk_id")


def _metadata_of(record: Any) -> Mapping[str, Any]:
    """从块对象／映射里取元数据。"""
    value = getattr(record, "metadata", None)
    if isinstance(value, Mapping):
        return value
    if isinstance(record, Mapping):
        raw = record.get("metadata")
        if isinstance(raw, Mapping):
            return raw
    return {}


def _text_of(record: Any) -> str:
    """从块对象／映射里取正文。"""
    value = getattr(record, "text", None)
    if isinstance(value, str):
        return value
    if isinstance(record, Mapping):
        raw = record.get("text")
        if isinstance(raw, str):
            return raw
    return ""


class BM25Index:
    """内存 BM25 索引，条目与向量库的 chunk 一一对应。

    规模参考：全量 18262 块、平均 164 字，构建约 1 分半（含 jieba 分词），
    常驻内存约 300MB（主要是 ``rank_bm25`` 保存的每块词频 Counter）。

    :param records: 块记录（``Chunk`` 对象、映射或 :class:`rag.indexer.ChunkRecord`）
    :param tokenizer: 分词器，默认用政策术语词典构造
    :param require_retrievable: 是否在此层就排除审计公告（红线④）
    """

    def __init__(
        self,
        records: Iterable[Any],
        tokenizer: PolicyTokenizer | None = None,
        *,
        require_retrievable: bool = True,
    ) -> None:
        import numpy as np
        from rank_bm25 import BM25Okapi

        self.tokenizer = tokenizer or PolicyTokenizer()
        self._texts: list[str] = []
        self._metas: list[Mapping[str, Any]] = []
        self._ids: list[str] = []
        self._position: dict[str, int] = {}

        corpus: list[list[str]] = []
        for record in records:
            chunk_id = _chunk_id_of(record)
            if chunk_id in self._position:
                continue  # 同一块重复出现只保留第一份，避免分数重复计入
            text = _text_of(record)
            metadata = _metadata_of(record)
            if require_retrievable and not is_retrievable(metadata):
                continue
            self._position[chunk_id] = len(self._ids)
            self._ids.append(chunk_id)
            self._texts.append(text)
            self._metas.append(metadata)
            tokens = self.tokenizer.cut(text)
            if not tokens:
                # 空正文（或全为停用词）会让 BM25Okapi 的 avgdl 计算除零，
                # 且这种块本来就检索不到，直接不入索引
                logger.warning("块 %s 分词结果为空，跳过（不参与 BM25）", chunk_id)
                self._position.pop(chunk_id, None)
                self._ids.pop()
                self._texts.pop()
                self._metas.pop()
                continue
            corpus.append(tokens)

        if not corpus:
            raise ValueError("BM25 索引为空：没有可用的块记录")
        self._bm25 = BM25Okapi(corpus)
        # 预先算好两个掩码，search 时按配置组合，避免每个查询都重新遍历元数据
        self._retrievable = np.array(
            [is_retrievable(meta) for meta in self._metas], dtype=bool
        )
        self._current = np.array(
            [meta.get(CURRENT_FIELD) is True for meta in self._metas], dtype=bool
        )
        logger.info(
            "BM25 索引就绪：%d 块（现行 %d 块），术语 %d 条",
            len(self._ids),
            int(self._current.sum()),
            len(self.tokenizer.terms),
        )

    def __len__(self) -> int:
        """索引内的块数。"""
        return len(self._ids)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """全部块 id（按索引顺序）。"""
        return tuple(self._ids)

    def record(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None:
        """按 id 取 ``(正文, 元数据)``，融合时需要补全只在稀疏侧命中的块。"""
        position = self._position.get(chunk_id)
        if position is None:
            return None
        return self._texts[position], self._metas[position]

    def search(
        self,
        query: str,
        k: int = 40,
        *,
        require_current: bool = True,
        where: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 检索。

        顺序是**全量打分 → mask → 截断**：先对每个块算分，再把不合格的块分数
        归零并剔除，最后才取前 k。若写成"先取前 k 再筛"，失效块会占掉名额，
        与稠密侧的后置过滤是同一个坑。

        :param query: 查询原文
        :param k: 返回条数
        :param require_current: 是否只保留现行有效块（A6）
        :param where: 追加的等值过滤条件，形如 ``{"doc_no": "粤农农规〔2023〕6号"}``
        :return: ``[(chunk_id, 分数)]``，分数降序
        """
        import numpy as np

        tokens = self.tokenizer.cut(query)
        if not tokens or k <= 0:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=float)

        mask = self._retrievable.copy()
        if require_current:
            mask &= self._current
        if where:
            mask &= self._mask_for(where)
        scores = np.where(mask, scores, 0.0)

        # 命中判定用 ``!= 0`` 而不是 ``> 0``：小语料（按文号过滤后只剩几块）或
        # 词典里出现"每个文档都有"的词时，BM25Okapi 的 IDF 会变成负数，命中块的
        # 分数是负的。用 ``> 0`` 会把它们全判成"没命中"，检索直接空结果——
        # 这正是本模块最初写错的地方。未命中块的分数精确为 0，用 ``!= 0`` 能区分。
        candidates = np.flatnonzero(scores != 0)
        if candidates.size == 0:
            return []
        if candidates.size > k:
            top = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        else:
            top = candidates
        ordered = top[np.argsort(-scores[top])]
        return [(self._ids[index], float(scores[index])) for index in ordered]

    def _mask_for(self, where: Mapping[str, Any]) -> Any:
        """把等值 where 条件翻译成布尔掩码。

        只支持等值（含 ``is_current`` 布尔）——混合检索里需要限定的场景就是
        "定文号""定类别"，用不上 ``$ne``／范围比较；遇到不支持的条件直接报错，
        不静默忽略（静默忽略会让调用方以为过滤生效）。
        """
        import numpy as np

        mask = np.ones(len(self._ids), dtype=bool)
        for key, value in where.items():
            if key.startswith("$"):
                raise NotImplementedError(f"BM25 侧不支持复合条件：{key}")
            if isinstance(value, Mapping):
                # Chroma 算子（如 {"category": {"$ne": "审计公告"}}）在这里会被
                # 当成"与一个 dict 相等"来比较，结果恒为 False，整路静默返回空。
                # 必须显式报错——静默失败比不支持更危险。
                raise NotImplementedError(
                    f"BM25 侧不支持算子条件：{key} = {dict(value)}；请传等值条件"
                )
            if key == CURRENT_FIELD:
                expected = value is True
                mask &= self._current if expected else ~self._current
                continue
            mask &= np.array(
                [meta.get(key) == value for meta in self._metas], dtype=bool
            )
        return mask


# ================================================================ RRF 融合


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：``score(d) = Σ w_i / (k + rank_i(d))``。

    只用排名、不用原始分数，因此不需要在不同量纲之间做归一化——BM25 分数无上界、
    随查询长度漂移，余弦相似度固定在 ``[-1, 1]``，两者加权求和的权重是拍脑袋的；
    排名则天然可比。

    :param rankings: 多路有序 id 列表（每路按相关性降序，rank 从 1 起）
    :param rrf_k: 平滑常数，手册 A5 取 60；越大则排名靠前的优势越平缓
    :param weights: 每路权重，默认等权；长度必须与 ``rankings`` 一致
    :param limit: 只返回前若干条
    :return: ``[(id, 融合分)]``，分数降序；同分保持首次出现的顺序
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须为正数")
    if weights is None:
        weights = [1.0 / len(rankings)] * len(rankings) if rankings else []
    if len(weights) != len(rankings):
        raise ValueError("weights 长度必须与 rankings 一致")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for route, (ids, weight) in enumerate(zip(rankings, weights, strict=False)):
        if weight < 0:
            raise ValueError("weight 不能为负")
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
            first_seen.setdefault(chunk_id, route * 10_000_000 + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return ordered[:limit] if limit is not None else ordered


# ================================================================ 配置与结构


@dataclass(frozen=True)
class HybridConfig:
    """混合检索参数，对应 ``config.json`` 的 ``hybrid`` 段。

    :param top_k: 最终返回条数
    :param dense_k: 稠密一路取多少条进融合
    :param bm25_k: 稀疏一路取多少条进融合
    :param rrf_k: RRF 平滑常数，手册 A5 起步值 60，属 T2 待评测定稿项
    :param dense_weight: 稠密一路权重，起步 0.5
    :param sparse_weight: 稀疏一路权重，起步 0.5
    :param enable_temporal_filter: 时效硬过滤开关（A6）。**默认开**；关掉仅用于消融，
        关掉后两路都不再按 ``is_current`` 过滤，结果会含失效条款
    """

    top_k: int = 10
    dense_k: int = 40
    bm25_k: int = 40
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    enable_temporal_filter: bool = True

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> HybridConfig:
        """从整份配置构造；``hybrid`` 段缺字段回落到默认值。"""
        section: Mapping[str, Any] = {}
        if isinstance(raw, Mapping):
            candidate = raw.get("hybrid")
            if isinstance(candidate, Mapping):
                section = candidate
        defaults = cls()
        if not section:
            return defaults
        return cls(
            top_k=int(section.get("top_k") or defaults.top_k),
            dense_k=int(section.get("dense_k") or defaults.dense_k),
            bm25_k=int(section.get("bm25_k") or defaults.bm25_k),
            rrf_k=int(section.get("rrf_k") or defaults.rrf_k),
            dense_weight=float(section.get("dense_weight", defaults.dense_weight)),
            sparse_weight=float(section.get("sparse_weight", defaults.sparse_weight)),
            enable_temporal_filter=bool(
                section.get("enable_temporal_filter", defaults.enable_temporal_filter)
            ),
        )


@dataclass(frozen=True)
class FusedHit:
    """融合后的一条命中：原始命中＋两路各自的排名与分数。

    :param hit: 原始 :class:`~rag.retriever.Hit`（正文与元数据都来自块记录）
    :param fusion_score: RRF 融合分，**排序只认它**
    :param dense_rank: 稠密一路的排名（1 起）；未命中为 ``None``
    :param dense_score: 稠密余弦相似度；未命中为 ``None``
    :param sparse_rank: 稀疏一路的排名；未命中为 ``None``
    :param bm25_score: BM25 原始分数；未命中为 ``None``
    """

    hit: Hit
    fusion_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    bm25_score: float | None = None

    @property
    def source(self) -> str:
        """命中来源：``both`` / ``dense`` / ``sparse``。

        只看 ``dense`` 与 ``sparse`` 两个字段容易被误读成"排名"，
        这个字段用于统计"混合到底补了多少只有一路能召回的块"。
        """
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "both"
        return "dense" if self.dense_rank is not None else "sparse"

    @property
    def chunk_id(self) -> str:
        """块 id。"""
        return self.hit.chunk_id

    @property
    def text(self) -> str:
        """块正文。"""
        return self.hit.text

    @property
    def metadata(self) -> Mapping[str, Any]:
        """块元数据。"""
        return self.hit.metadata

    @property
    def citation(self) -> str:
        """严格引用串（透传 :attr:`~rag.retriever.Hit.citation`）。"""
        return self.hit.citation

    @property
    def reference(self) -> str:
        """展示用引用（透传 :attr:`~rag.retriever.Hit.reference`）。"""
        return self.hit.reference

    @property
    def doc_no(self) -> str:
        """发文字号。"""
        return self.hit.doc_no

    @property
    def clause(self) -> str:
        """条款号。"""
        return self.hit.clause

    @property
    def is_current(self) -> bool:
        """是否现行有效。"""
        return self.hit.is_current

    @property
    def title(self) -> str:
        """文件标题（供无文号时展示）。"""
        return str(getattr(self.hit, "title", "") or "")

    @property
    def has_doc_no(self) -> bool:
        """是否有可回溯的**真**发文字号（占位串不算，见 ``Hit.has_doc_no``）。"""
        return bool(getattr(self.hit, "has_doc_no", False))

    @property
    def validity(self) -> str:
        """效力状态原文，报告里要展示判断依据。"""
        return str(getattr(self.hit, "validity", "") or "")

    @property
    def publish_date(self) -> str:
        """发布日期，A6 的"申报窗口"展示要用。"""
        return str(getattr(self.hit, "publish_date", "") or "")

    @property
    def serial(self) -> int | None:
        """数据仓序号，用于回查原文。"""
        value = getattr(self.hit, "serial", None)
        return value if isinstance(value, int) else None

    def as_dict(self) -> dict[str, Any]:
        """可序列化形式，供评测脚本落 JSONL。

        **只有稀疏侧命中时，``score``／``distance`` 置 ``None``**：那种块没有余弦分，
        若填 0.0 会让下游按 ``score > 0`` 过滤时把混合救回来的块整批丢掉。
        """
        payload = self.hit.as_dict()
        payload.update(
            {
                "fusion_score": self.fusion_score,
                "dense_rank": self.dense_rank,
                "dense_score": self.dense_score,
                "sparse_rank": self.sparse_rank,
                "bm25_score": self.bm25_score,
                "source": self.source,
            }
        )
        if self.dense_rank is None:
            payload["score"] = None
            payload["distance"] = None
        return payload


@dataclass
class HybridResult:
    """一次混合检索的结果。

    :param query: 原始查询
    :param hits: 融合后命中，按 ``fusion_score`` 降序
    :param where: 实际下发的附加条件（审计／调试用）
    :param dense_count: 稠密一路返回条数
    :param sparse_count: 稀疏一路返回条数
    :param overlap: 两路共同命中的块数
    """

    query: str
    hits: list[FusedHit] = field(default_factory=list)
    where: dict[str, Any] = field(default_factory=dict)
    dense_count: int = 0
    sparse_count: int = 0
    overlap: int = 0
    temporal_filtered: bool = True

    def summary(self) -> str:
        """一行摘要，含两路互补性统计。"""
        both = sum(1 for hit in self.hits if hit.source == "both")
        dense_only = sum(1 for hit in self.hits if hit.source == "dense")
        sparse_only = sum(1 for hit in self.hits if hit.source == "sparse")
        state = "已开" if self.temporal_filtered else "已关（消融）"
        return (
            f"「{self.query}」→ 融合 {len(self.hits)} 块（时效过滤{state}）｜"
            f"稠密 {self.dense_count} / 稀疏 {self.sparse_count} / 重叠 {self.overlap}｜"
            f"结果来源 both {both} / dense {dense_only} / sparse {sparse_only}"
        )

    def citations(self) -> list[str]:
        """命中里的严格引用串（去重保序），空串已剔除。"""
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.citation:
                seen.setdefault(hit.citation, None)
        return list(seen)

    def references(self) -> list[str]:
        """命中里的展示用引用（去重保序）。"""
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.reference:
                seen.setdefault(hit.reference, None)
        return list(seen)

    def doc_nos(self) -> list[str]:
        """命中涉及的文号（去重保序）。"""
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.doc_no:
                seen.setdefault(hit.doc_no, None)
        return list(seen)

    def source_counts(self) -> dict[str, int]:
        """结果来源分布，用于评测里量化"混合补了多少只有一路能召回的块"。"""
        counts = {"both": 0, "dense": 0, "sparse": 0}
        for hit in self.hits:
            counts[hit.source] += 1
        return counts


# ================================================================ 检索器


class DenseSearcher(Protocol):
    """稠密一路的协议（:class:`rag.retriever.DenseRetriever` 满足它）。"""

    def retrieve(
        self,
        query: str,
        top_k: int | None = ...,  # noqa: ANN401 - 协议声明
        where: Mapping[str, Any] | None = ...,
        count_candidates: bool = ...,
    ) -> Any:
        """返回带 ``hits`` 的结果对象。"""
        ...


class HybridRetriever:
    """稠密＋稀疏双路召回，RRF 融合。

    :param dense: 稠密检索器（PR#4 的 :class:`~rag.retriever.DenseRetriever`）
    :param sparse: BM25 索引
    :param config: 融合参数
    """

    def __init__(
        self,
        dense: DenseSearcher,
        sparse: BM25Index,
        config: HybridConfig | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.config = config or HybridConfig()

    def _sync_dense_temporal(self) -> None:
        """把混合配置的时效开关同步给稠密一路。

        为什么必须同步：混合检索的时效开关在 :class:`HybridConfig` 里，而稠密侧
        用的是它自己的 :class:`~rag.retriever.RetrievalConfig`。两者一旦不一致，
        消融实验就会出现"稀疏侧放开了、稠密侧还锁着"的假象——跑出来的"关掉过滤
        的对照组"其实只关了一半，结论直接作废。这种双开关不一致是混合架构里最
        隐蔽的错误，所以在这里强制对齐并在日志里留痕。
        """
        dense_config = getattr(self.dense, "config", None)
        if not isinstance(dense_config, RetrievalConfig):
            return
        if dense_config.enable_temporal_filter == self.config.enable_temporal_filter:
            return
        self.dense.config = replace(
            dense_config, enable_temporal_filter=self.config.enable_temporal_filter
        )
        logger.info(
            "稠密侧时效开关已同步为 %s（与混合配置对齐）",
            self.config.enable_temporal_filter,
        )

    @contextmanager
    def _aligned_dense(self) -> Iterator[None]:
        """在对齐时效口径的前提下跑稠密召回，**结束后恢复原状**。

        为什么要恢复：``DenseRetriever`` 可能被多个调用方共享。若这里把它的
        ``config`` 永久改掉，一次消融实验（关掉过滤）会让后续所有使用该实例的
        检索都失去时效过滤——错得很隐蔽，且只在"先跑消融、再跑正式"的顺序下暴露。
        所以对齐是**临时的**：进作用域改、出作用域还原（异常路径也还原）。
        """
        original = getattr(self.dense, "config", None)
        try:
            self._sync_dense_temporal()
            yield
        finally:
            if original is not None and isinstance(original, RetrievalConfig):
                self.dense.config = original

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> HybridResult:
        """检索并融合。

        :param query: 用户问题原文
        :param top_k: 覆盖配置里的最终条数
        :param where: 追加等值条件（如限定文号／类别），两路都会应用
        :return: 融合结果；查询为空时返回空 ``hits``，不抛异常
        """
        cfg = self.config
        text = (query or "").strip()
        if not text:
            logger.warning("查询为空，返回空结果")
            return HybridResult(
                query="",
                where=dict(where or {}),
                temporal_filtered=cfg.enable_temporal_filter,
            )

        limit = cfg.top_k if top_k is None else int(top_k)

        # 稠密一路：where 由 DenseRetriever 负责下发到向量库（时效条件前置）。
        # 用上下文管理器临时对齐时效口径，退出即还原（见 _aligned_dense 的说明）。
        with self._aligned_dense():
            dense_result = self.dense.retrieve(
                text,
                top_k=cfg.dense_k,
                where=dict(where or {}) or None,
                count_candidates=False,
            )
        dense_hits: list[Hit] = list(getattr(dense_result, "hits", []))

        # 稀疏一路：BM25 没有 where，同样把时效与类别掩码打在打分之后、截断之前
        sparse_pairs = self.sparse.search(
            text,
            k=cfg.bm25_k,
            require_current=cfg.enable_temporal_filter,
            where=where,
        )

        dense_ids = [hit.chunk_id for hit in dense_hits]
        sparse_ids = [chunk_id for chunk_id, _ in sparse_pairs]
        dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, 1)}
        sparse_rank = {chunk_id: rank for rank, chunk_id in enumerate(sparse_ids, 1)}
        dense_score = {hit.chunk_id: hit.score for hit in dense_hits}
        bm25_score = dict(sparse_pairs)

        # 只有一路有结果时权重归一为 1.0（两侧对称：任一路空都归 1）：若仍按
        # dense_weight/sparse_weight 打折，融合分会整体缩水一半，跨查询不可比
        has_dense = bool(dense_ids)
        has_sparse = bool(sparse_ids)
        rankings = [dense_ids, sparse_ids] if has_dense and has_sparse else (
            [dense_ids] if has_dense else [sparse_ids]
        )
        fused = rrf_fuse(
            rankings,
            rrf_k=cfg.rrf_k,
            weights=(
                [cfg.dense_weight, cfg.sparse_weight]
                if has_dense and has_sparse
                else [1.0]
            ),
            limit=limit,
        )

        by_id = {hit.chunk_id: hit for hit in dense_hits}
        hits: list[FusedHit] = []
        for chunk_id, fusion_score in fused:
            hit = by_id.get(chunk_id)
            if hit is None:
                # 只在稀疏侧命中的块：正文与元数据从 BM25 索引里取
                found = self.sparse.record(chunk_id)
                if found is None:  # pragma: no cover - 索引自洽时不发生
                    logger.warning("融合命中的块 %s 在稀疏索引里找不到，跳过", chunk_id)
                    continue
                body, metadata = found
                hit = Hit(
                    chunk_id=chunk_id,
                    text=body,
                    score=0.0,  # 无稠密分数，排序只认 fusion_score
                    distance=1.0,
                    metadata=metadata,
                )
            # 融合侧再校验一次红线④：两路都过滤过，但这里是最后一道
            if not is_retrievable(hit.metadata):
                logger.warning("块 %s 类别不允许进入结果，融合阶段丢弃（红线④）", chunk_id)
                continue
            hits.append(
                FusedHit(
                    hit=hit,
                    fusion_score=round(fusion_score, 8),
                    dense_rank=dense_rank.get(chunk_id),
                    dense_score=dense_score.get(chunk_id),
                    sparse_rank=sparse_rank.get(chunk_id),
                    bm25_score=bm25_score.get(chunk_id),
                )
            )

        result = HybridResult(
            query=text,
            hits=hits,
            where=dict(where or {}),
            dense_count=len(dense_ids),
            sparse_count=len(sparse_ids),
            overlap=len(set(dense_ids) & set(sparse_ids)),
            temporal_filtered=cfg.enable_temporal_filter,
        )
        logger.info("%s", result.summary())
        return result


# ================================================================ 装配与命令行


def build_hybrid_retriever(
    config: Mapping[str, Any] | None = None,
    *,
    records: Iterable[Any] | None = None,
    tokenizer: PolicyTokenizer | None = None,
) -> HybridRetriever:
    """按本机配置装配真实混合检索器（局部 import，避免测试被迫装 torch／chromadb）。

    :param config: 整份配置
    :param records: 块记录；``None`` 时用 loader＋chunker 现场从语料生成
    :param tokenizer: 复用已有分词器（避免重复加载 jieba 词典）
    :return: 可直接使用的 :class:`HybridRetriever`
    """
    from rag.indexer import chunks_from_corpus
    from rag.retriever import build_retriever

    if records is None:
        records = chunks_from_corpus(config)
    dense = build_retriever(config)
    sparse = BM25Index(records, tokenizer=tokenizer)
    return HybridRetriever(dense, sparse, HybridConfig.from_config(config))


def _load_config(path: Path | str) -> dict[str, Any]:
    """读取本机配置；文件缺失时返回空配置（各段回落默认值）。"""
    target = Path(path)
    if not target.is_file():
        logger.warning("配置文件不存在，使用默认参数：%s", target)
        return {}
    return json.loads(target.read_text(encoding="utf-8-sig"))


def _format_hit(rank: int, hit: FusedHit) -> str:
    """一行命中摘要。"""
    marks = []
    if hit.dense_rank is not None:
        marks.append(f"d{hit.dense_rank}")
    if hit.sparse_rank is not None:
        marks.append(f"s{hit.sparse_rank}")
    head = hit.citation or f"[无文号] {hit.reference}" or hit.chunk_id
    return f"{rank:>2}. [{hit.fusion_score:.5f}] {'+'.join(marks):>7}  {head}"


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：``python -m rag.hybrid "查询" --config config.json``。"""
    parser = argparse.ArgumentParser(description="粤策哨混合检索（BM25＋稠密，RRF 融合）")
    parser.add_argument("query", nargs="*", help="查询文本（多个词会自动拼接）")
    parser.add_argument("--config", default="config.json", help="本机配置文件路径")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数，默认取配置")
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="关闭时效硬过滤（仅供消融对照，两路都会放开）",
    )
    parser.add_argument("--doc-no", default=None, help="只检索指定文号的块")
    parser.add_argument("--show-text", action="store_true", help="打印命中块正文")
    parser.add_argument(
        "--chunks",
        default=None,
        help="从块 JSONL 建 BM25 索引（省去重新分块）；默认现场从语料分块",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = " ".join(args.query).strip()
    if not query:
        parser.error("请提供查询文本")

    raw_config = _load_config(args.config)
    config = HybridConfig.from_config(raw_config)
    if args.no_temporal:
        config = replace(config, enable_temporal_filter=False)

    records = None
    if args.chunks:
        from rag.indexer import load_chunks

        records = load_chunks(args.chunks)
    retriever = build_hybrid_retriever(raw_config, records=records)
    retriever.config = config

    where = {"doc_no": args.doc_no} if args.doc_no else None
    result = retriever.retrieve(query, top_k=args.top_k, where=where)

    print(result.summary())
    for rank, hit in enumerate(result.hits, start=1):
        print(_format_hit(rank, hit))
        if args.show_text:
            print(f"    {hit.text[:120].replace(chr(10), ' ')}")
    if not result.hits:
        print("（无命中）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
