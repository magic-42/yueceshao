"""粤策哨 · 稠密检索与时效硬过滤。

职责
----
把一段用户问题变成**有序的条款命中列表**，每条命中带齐文号／条款号／效力状态，
供下游生成层做标记式引用（A8）。对应《粤策哨开发手册 v1.0》§3.1 PR-4：

- 稠密检索：查询经 Qwen3-Embedding 编码（查询侧带任务指令）后在 Chroma 上取 top-k；
- 时效硬过滤（A6）：``is_current`` **前置为元数据条件**，由向量库先筛后排序；
- 红线④兜底：检索端再排除一次 ``category == 审计公告``。

为什么时效过滤必须前置、不能取回后再筛
--------------------------------------
后置过滤只写了半句对的：先取 top-10，再与现行清单求交。问题在**召回塌陷**——
若 top-10 里恰好 8 条来自已失效文件，后置筛完只剩 2 条，而且这 2 条并不是"现行
有效中最相关的前 2 条"，只是"碰巧挤进前十的前 2 条"。用户看到"只找到 2 条依据"，
真实原因却是失效文件占满了名额。前置过滤让候选池一开始就只含现行有效块，
top-k 名额全部花在能用的条款上。这也是 A6 写成"前置为元数据过滤"的原因。

同理，后置过滤还会**破坏 k 的语义**：下游 rerank 按 k40→top8 设计（PR#6），
拿到的 40 条里混着失效条款，等于重排预算被废掉一半。

为什么这里不 import ``rag.indexer``
------------------------------------
本模块只依赖两样东西：能编码查询的 ``Embedder``、能按向量检索的 ``SearchBackend``。
两者都是协议（鸭子类型），测试用内存实现即可，CI 不必装 chromadb／torch。
命令行入口里才局部 import ``rag.indexer`` 去装配真实组件——与 indexer 的
「不 import chunker」是同一套解耦口径。

关于相似度数值
--------------
集合按 ``cosine`` 建（见 indexer 的 ``IndexConfig.distance``），Chroma 返回的是
**距离**：``distance = 1 - cos_sim``，取值 0（完全相同）到 2（完全相反）。本模块对
外统一暴露 ``score = 1 - distance``（越大越相关），``Hit.distance`` 同时保留原始值，
方便调阈值时对照。别把两者混着用——这是最容易写反的地方。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

AUDIT_CATEGORY = "审计公告"
"""红线④：审计公告是评测出卷集，进检索结果等于答案泄露。

indexer 已在入库端拦一次并会清出遗留块；这里在检索端再拦一次——集合可能来自旧版本、
手工导入或他人机器，检索端不能假定它是干净的。
"""

AUDIT_CATEGORIES: frozenset[str] = frozenset({AUDIT_CATEGORY})
"""禁止出现在检索结果里的类别集合。将来若有多类"内部评测用"文件，加在这里。"""

CURRENT_FIELD = "is_current"
"""时效字段。A6 的硬过滤开关只切这一个字段，消融实验也只看它。"""

PLACEHOLDER_DOC_NO = re.compile(r"^无[（(]|未载|未标注|仅正文引用|不计")
"""占位文号的形态：registry 里未载文号的文件，``文号`` 字段写成 ``无（…）``。

这类字符串非空但与"文号"无关，混进引用位等同伪造（红线①）。loader 已用
``〔年份〕序号号`` 正则算好 ``has_doc_no`` 字段；这里是**兜底**，用于字段缺失的
旧索引或手工导入数据。
"""

# (chunk_id, 正文, 元数据, distance)
RawMatch = tuple[str, str, dict[str, Any], float]


# ================================================================ 配置


@dataclass(frozen=True)
class RetrievalConfig:
    """检索参数，对应 ``config.json`` 的 ``retrieval`` 段。

    :param top_k: 返回条数。给重排用时可调大（手册 PR#6 是 k40→top8）
    :param enable_temporal_filter: 时效硬过滤开关。**默认开**；关掉仅用于消融对照，
        关掉后结果里会混入已失效条款，对外输出前必须重新打开。
        消融实验统一由 ``eval/`` 的配置布尔位驱动（README 要求消融开关集中在 eval/），
        本字段是它读取的下游参数；命令行 ``--no-temporal`` 只用于手工对照，不参与
        正式跑批——两者都在这里落到同一个字段，不会被绕过。
    :param exclude_categories: 检索端排除的类别，默认排除审计公告（红线④）
    :param exclude_preamble: 是否排除前言块（"为贯彻落实……"），默认保留——
        申报条件有时只在前言里写，是否排除交给评测决定
    :param exclude_appendix: 是否排除附录块（附表／附件），默认保留，同上
    """

    top_k: int = 10
    enable_temporal_filter: bool = True
    exclude_categories: tuple[str, ...] = (AUDIT_CATEGORY,)
    exclude_preamble: bool = False
    exclude_appendix: bool = False

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> RetrievalConfig:
        """从整份配置构造；``retrieval`` 段缺字段回落到默认值。"""
        section = _section(raw, "retrieval")
        categories = section.get("exclude_categories")
        return cls(
            top_k=int(section.get("top_k") or 10),
            enable_temporal_filter=bool(section.get("enable_temporal_filter", True)),
            exclude_categories=(
                tuple(str(item) for item in categories)
                if isinstance(categories, (list, tuple))
                else (AUDIT_CATEGORY,)
            ),
            exclude_preamble=bool(section.get("exclude_preamble", False)),
            exclude_appendix=bool(section.get("exclude_appendix", False)),
        )


def _section(raw: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    """取配置子段；缺失或类型不对时返回空映射。"""
    if not isinstance(raw, Mapping):
        return {}
    section = raw.get(key)
    return section if isinstance(section, Mapping) else {}


# ================================================================ where 构造


def build_where(
    config: RetrievalConfig | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """把检索配置拼成传给向量库的 ``where`` 条件（A6 前置过滤的落点）。

    多个条件用 ``$and`` 连接：Chroma 的 ``where`` 一次只能表达一个键，
    多键并列必须显式包一层。

    :param config: 检索参数；``None`` 用默认值（时效过滤开启）
    :param overrides: 调用方临时追加的等值条件，如 ``{"doc_no": "粤财农〔2020〕106号"}``
    :return: Chroma ``where`` 字典；无条件时返回 ``{}``（表示不过滤）
    """
    cfg = config or RetrievalConfig()
    clauses: list[dict[str, Any]] = []

    if cfg.enable_temporal_filter:
        clauses.append({CURRENT_FIELD: True})
    for category in cfg.exclude_categories:
        clauses.append({"category": {"$ne": category}})
    if cfg.exclude_preamble:
        clauses.append({"is_preamble": False})
    if cfg.exclude_appendix:
        clauses.append({"is_appendix": False})
    for key, value in (overrides or {}).items():
        clauses.append({key: value})

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def is_retrievable(metadata: Mapping[str, Any]) -> bool:
    """该块的类别是否允许出现在检索结果里（红线④，**fail-closed**）。

    为什么需要这个函数、而不是只靠 where 里的 ``$ne``
    ------------------------------------------------
    Chroma 对 ``{"category": {"$ne": "审计公告"}}`` 的语义是"字段值不等于它"，
    而**缺少该字段或字段为空串的块会被判为满足条件**（实测 chromadb 1.5.9）。
    也就是说 ``$ne`` 挡不住"类别未知"的块。叠加 ``scalarize_metadata`` 会把
    ``None`` 收敛成 ``""``，一旦集合来自旧版本、手工导入或他人机器，红线④就会从
    这个缝里漏过去——而审计公告进检索结果等于评测答案泄露，这条缝不能留。

    所以判定写成**白名单式**：类别必须是非空字符串且不在禁用集合里，否则一律丢弃。
    宁可错杀几块类别不明的块（当前全库 18262 块都有合法类别，实测空值 0 块，
    代价为零），也不放一条审计公告进来。

    :param metadata: 块的扁平元数据
    :return: ``True`` 表示可进入结果
    """
    category = metadata.get("category")
    if not isinstance(category, str) or not category.strip():
        return False
    return category not in AUDIT_CATEGORIES


# ================================================================ 数据结构


@dataclass(frozen=True)
class Hit:
    """一条命中。

    :param chunk_id: 块 id，``{序号}-{顺序号}``，可回溯到原文
    :param text: 块正文（已含层级头注入，可直接作为证据引用）
    :param score: 余弦相似度，``1 - distance``，越大越相关
    :param distance: 向量库原始距离，便于与阈值对照
    :param metadata: 该块的扁平元数据
    """

    chunk_id: str
    text: str
    score: float
    distance: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def _meta(self, key: str) -> str:
        """取一个元数据字符串字段；缺失或非字符串时返回空串。"""
        value = self.metadata.get(key)
        return value if isinstance(value, str) else ""

    @property
    def doc_no(self) -> str:
        """发文字号；来源是 registry 元数据，缺失时为空串。"""
        return self._meta("doc_no")

    @property
    def title(self) -> str:
        """文件标题。"""
        return self._meta("title")

    @property
    def clause(self) -> str:
        """条款号，如"第十二条"；无条款结构的文件为空串。"""
        return self._meta("clause")

    @property
    def serial(self) -> int | None:
        """数据仓序号，用于回查原文。"""
        value = self.metadata.get("serial")
        return value if isinstance(value, int) else None

    @property
    def is_current(self) -> bool:
        """是否现行有效（A6 过滤依据）。

        用 ``is True`` 而不是 ``bool(...)``：元数据里若出现字符串 ``"false"``
        （手工导入、其他后端写入），``bool("false")`` 为真，会把已失效条款当现行条款。
        """
        return self.metadata.get(CURRENT_FIELD) is True

    @property
    def publish_date(self) -> str:
        """发布日期（``YYYY-MM-DD``）；A6 的"申报窗口（展示）"要用。"""
        return self._meta("publish_date")

    @property
    def validity(self) -> str:
        """效力状态原文（如"有效"／"已失效"），用于报告里展示判断依据。"""
        return self._meta("validity")

    @property
    def has_doc_no(self) -> bool:
        """是否有可回溯的**真**发文字号。

        **优先读元数据里的 ``has_doc_no`` 字段**——它是 loader 用 ``〔年份〕序号号``
        正则判定出来的权威结论；只有该字段缺失（旧索引、手工导入）时才回退到
        "非空即真"。

        为什么不能用 ``bool(doc_no)`` 判定
        ----------------------------------
        语料里有 6267 块（34%）来自未载文号的文件，registry 的 ``文号`` 字段写的是
        占位串，例如 ``无（申报通知未载文号）``、``无（页面未载/仅正文引用，不计）``。
        这些字符串非空，用 ``bool()`` 会全部判成"有文号"，最终报告里就会出现

            无（页面未载/仅正文引用，不计） 第四条

        这种既不是文号、也查不动的引用——红线①要求文号可核验，占位串混进引用位
        等同于伪造。实测 ``has_doc_no=True`` 的 11995 块 100% 含真文号格式，
        所以以该字段为准是可靠的。
        """
        flag = self.metadata.get("has_doc_no")
        if isinstance(flag, bool):
            return flag
        if not self.doc_no:
            return False
        return PLACEHOLDER_DOC_NO.search(self.doc_no) is None

    @property
    def citation(self) -> str:
        """**严格**引用串：``文号 条款号``；没有真文号时返回空串。

        **只用元数据里已有的字段拼接，任何一段缺失都不补、不猜**。下游渲染层
        依赖这个约定来保证"零手打文号"（红线②）：拿到的要么是 registry 里的
        真实文号，要么是空——不会出现模型编的字符串，也不会出现占位串。

        为什么文号缺失时给空串而不是退回"只有第十二条"
        ----------------------------------------------
        校验器（PR#3 gen/validator）要靠文号去 registry 反查，只有条款号的引用串
        无法校验，等于把一条查不动的依据混进报告。这类块占全库 34%，不是少数，
        所以必须让下游"看得见它没文号"，而不是拿到一个看起来能用的半截引用。
        需要展示时用 :attr:`reference`。
        """
        if not self.has_doc_no:
            return ""
        return f"{self.doc_no} {self.clause}".strip()

    @property
    def reference(self) -> str:
        """**展示**用引用：无真文号时退回标题，保证读者仍能定位到文件。

        与 :attr:`citation` 的分工：``citation`` 给校验器和最终报告（可能为空），
        ``reference`` 给人看（尽量不为空）。占位文号（``无（…）``）也视为"没有真文号"，
        展示时同样退回标题——把占位串摆在引用位上会让人误以为有出处。
        """
        head = self.doc_no if self.has_doc_no else self.title
        return f"{head} {self.clause}".strip()

    def as_dict(self) -> dict[str, Any]:
        """可序列化形式，供评测脚本落 JSONL。"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "distance": self.distance,
            "citation": self.citation,
            "reference": self.reference,
            "has_doc_no": self.has_doc_no,
            "metadata": dict(self.metadata),
        }


@dataclass
class RetrievalResult:
    """一次检索的完整结果，含"过滤器到底生效没有"的证据。

    :param query: 原始查询
    :param hits: 有序命中，相似度降序
    :param where: 实际下发的 where 条件（审计／调试用）
    :param candidates: 该 where 条件下的候选块总数；``None`` 表示未统计
    """

    query: str
    hits: list[Hit] = field(default_factory=list)
    where: dict[str, Any] = field(default_factory=dict)
    candidates: int | None = None

    @property
    def temporal_filtered(self) -> bool:
        """本次是否施加了时效硬过滤。"""
        return _has_clause(self.where, CURRENT_FIELD)

    def citations(self) -> list[str]:
        """命中里的严格引用串（去重保序）。

        只含**文号可回溯**的那些；无文号的块不出现（它们由 :meth:`references` 覆盖）。
        """
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.citation:
                seen.setdefault(hit.citation, None)
        return list(seen)

    def references(self) -> list[str]:
        """命中里的展示引用（去重保序），无文号者退回标题，通常不为空。"""
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.reference:
                seen.setdefault(hit.reference, None)
        return list(seen)

    def no_doc_no_count(self) -> int:
        """命中里没有文号的块数——报告里要据此提示"此依据无发文字号"。"""
        return sum(1 for hit in self.hits if not hit.has_doc_no)

    def doc_nos(self) -> list[str]:
        """命中涉及的发文字号（去重保序）。"""
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.doc_no:
                seen.setdefault(hit.doc_no, None)
        return list(seen)

    def summary(self) -> str:
        """一行摘要，供命令行与 PR 输出样例使用。"""
        ageing = "已开" if self.temporal_filtered else "已关（消融）"
        pool = f"候选池 {self.candidates} 块" if self.candidates is not None else "候选池未统计"
        docs = len(self.doc_nos())
        return (
            f"「{self.query}」→ 命中 {len(self.hits)} 块 / {docs} 个文件"
            f"（时效过滤{ageing}，{pool}）"
        )


def _has_clause(where: Mapping[str, Any], key: str) -> bool:
    """``where`` 里是否含某个字段的条件（支持 ``$and`` 嵌套一层）。"""
    if not where:
        return False
    if key in where:
        return True
    nested = where.get("$and")
    if isinstance(nested, Sequence) and not isinstance(nested, str):
        return any(isinstance(item, Mapping) and key in item for item in nested)
    return False


# ================================================================ 后端协议


@runtime_checkable
class SearchBackend(Protocol):
    """向量检索后端协议；测试用内存实现替换，不必装 chromadb。"""

    def search(
        self, vector: Sequence[float], top_k: int, where: Mapping[str, Any]
    ) -> list[RawMatch]:
        """按向量取 top_k，返回 ``(chunk_id, text, metadata, distance)`` 列表。"""
        ...

    def count(self, where: Mapping[str, Any] | None = None) -> int:
        """该条件下的候选块总数。"""
        ...


@runtime_checkable
class Embedder(Protocol):
    """查询编码器协议（indexer 的 ``Qwen3Embedder`` 满足）。"""

    def encode_query(self, text: str) -> list[float]:
        """编码单条查询。"""
        ...


class ChromaBackend:
    """Chroma 集合的检索封装。

    只做「把向量查询翻译成 collection.query」这一件事，不涉及业务过滤逻辑——
    过滤条件由 :func:`build_where` 统一生成，避免两处各写一套。
    """

    def __init__(self, collection: Any) -> None:
        """:param collection: ``chromadb`` 的 Collection 对象。"""
        self._collection = collection

    def search(
        self, vector: Sequence[float], top_k: int, where: Mapping[str, Any]
    ) -> list[RawMatch]:
        """取 top-k 命中；空集合时返回空列表。"""
        if top_k <= 0:
            return []
        response = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=top_k,
            where=dict(where) or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        matches: list[RawMatch] = []
        for index, chunk_id in enumerate(ids):
            matches.append(
                (
                    str(chunk_id),
                    str(documents[index] or ""),
                    dict(metadatas[index] or {}),
                    float(distances[index]),
                )
            )
        return matches

    def count(self, where: Mapping[str, Any] | None = None) -> int:
        """候选块总数。

        Chroma 的 ``count()`` 不接受 ``where``，故用 ``get(include=[])`` 只取 id。
        这是**诊断用**接口（回答"前置过滤后还剩多少候选"），不在检索主路径上。
        """
        if not where:
            return int(self._collection.count())
        page = self._collection.get(where=dict(where), include=[])
        return len(page.get("ids") or [])


# ================================================================ 检索器


class DenseRetriever:
    """稠密检索器：查询编码 → where 前置过滤 → top-k → 距离转相似度。

    典型用法::

        retriever = DenseRetriever(embedder, ChromaBackend(store.collection))
        result = retriever.retrieve("产业园申报需要什么条件", top_k=8)
    """

    def __init__(
        self,
        embedder: Embedder,
        backend: SearchBackend,
        config: RetrievalConfig | None = None,
    ) -> None:
        """:param embedder: 查询编码器；:param backend: 检索后端。"""
        self.embedder = embedder
        self.backend = backend
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
        count_candidates: bool = False,
    ) -> RetrievalResult:
        """检索一段查询。

        :param query: 用户问题原文
        :param top_k: 覆盖配置里的条数；``None`` 用配置值
        :param where: 追加的等值条件（如限定某个文号／类别），与时效条件取交集
        :param count_candidates: 是否统计候选池大小。**默认关**——它是诊断接口，
            要走一次 ``get(include=[])`` 全量取 id，实测 961 块 16ms、按 18262 块
            外推约 300ms，是 query 本身（13ms）的 20 倍；A4 条目分解后查询会成倍
            增加，放在主路径上会白白拖慢每一次检索。只有排查"过滤是不是过严"时开
        :return: 检索结果；查询为空或无命中时返回空 ``hits``，不抛异常
        """
        text = (query or "").strip()
        if not text:
            logger.warning("查询为空，返回空结果")
            # 仍返回真实 where，否则 summary() 会把"没查"误报成"时效过滤已关（消融）"
            return RetrievalResult(query="", where=build_where(self.config, where), candidates=0)

        condition = build_where(self.config, where)
        limit = self.config.top_k if top_k is None else int(top_k)
        vector = self.embedder.encode_query(text)
        raw = self.backend.search(vector, limit, condition)

        hits = [
            Hit(
                chunk_id=chunk_id,
                text=body,
                score=round(1.0 - distance, 6),
                distance=distance,
                metadata=metadata,
            )
            for chunk_id, body, metadata, distance in raw
            # 红线④兜底：走到这一步不该再有审计公告，但检索端不假定集合是干净的。
            # 用 fail-closed 判定（类别缺失也丢弃），原因见 is_retrievable 的说明。
            if is_retrievable(metadata)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)

        candidates = self.backend.count(condition) if count_candidates else None
        result = RetrievalResult(query=text, hits=hits, where=condition, candidates=candidates)
        logger.info("检索完成 → %s", result.summary())
        return result

    def retrieve_many(
        self, queries: Sequence[str], **kwargs: Any
    ) -> list[RetrievalResult]:
        """批量检索（评测跑批用，逐条调用以免批次语义掩盖单条异常）。"""
        return [self.retrieve(query, **kwargs) for query in queries]


# ================================================================ 命令行


def _load_config(path: Path | str) -> dict[str, Any]:
    """读本机配置；文件不存在时返回空字典（全部走默认值）。"""
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("配置文件不存在：%s，使用默认参数", config_path)
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_retriever(config: Mapping[str, Any] | None = None) -> DenseRetriever:
    """按本机配置装配真实检索器（局部 import，避免测试被迫装 torch／chromadb）。"""
    from rag.indexer import ChromaStore, IndexConfig, Qwen3Embedder

    index_config = IndexConfig.from_config(config)
    store = ChromaStore(index_config)
    embedder = Qwen3Embedder(index_config.embedding)
    return DenseRetriever(
        embedder, ChromaBackend(store.collection),
        RetrievalConfig.from_config(config))


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：``python -m rag.retriever "查询" --config config.json``。"""
    parser = argparse.ArgumentParser(description="粤策哨稠密检索（含 A6 时效硬过滤）")
    parser.add_argument("query", nargs="*", help="查询文本（多个词会自动拼接）")
    parser.add_argument("--config", default="config.json", help="本机配置文件路径")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数，默认取配置")
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="关闭时效硬过滤（仅供消融对照，结果会含已失效条款）",
    )
    parser.add_argument("--doc-no", default=None, help="只检索指定文号的块")
    parser.add_argument(
        "--count-candidates",
        action="store_true",
        help="统计候选池大小（诊断用，多一次全量 get，默认关）",
    )
    parser.add_argument("--show-text", action="store_true", help="打印命中块正文")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = " ".join(args.query).strip()
    if not query:
        parser.error("请提供查询文本")

    raw_config = _load_config(args.config)
    config = RetrievalConfig.from_config(raw_config)
    if args.no_temporal:
        config = RetrievalConfig(
            top_k=config.top_k,
            enable_temporal_filter=False,
            exclude_categories=config.exclude_categories,
            exclude_preamble=config.exclude_preamble,
            exclude_appendix=config.exclude_appendix,
        )
    retriever = build_retriever(raw_config)
    retriever.config = config

    overrides = {"doc_no": args.doc_no} if args.doc_no else None
    result = retriever.retrieve(
        query,
        top_k=args.top_k,
        where=overrides,
        count_candidates=args.count_candidates,
    )

    lines = [result.summary(), f"where = {json.dumps(result.where, ensure_ascii=False)}"]
    for rank, hit in enumerate(result.hits, start=1):
        # 展示用 reference（无文号会退回标题），并明确标出"无文号"——
        # 这类块占全库 34%，报告里不能让人误以为它有正式发文字号
        flag = "" if hit.has_doc_no else "  [无文号]"
        lines.append(f"{rank:>2}. [{hit.score:.4f}] {hit.reference or hit.chunk_id}{flag}")
        if args.show_text:
            lines.append(f"    {hit.text[:120].replace(chr(10), ' ')}")
    if not result.hits:
        lines.append("（无命中）")
    # 统一走 logging（README 编码规范），整块输出只带一个前缀
    logger.info("检索输出\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover - 入口
    raise SystemExit(main())
