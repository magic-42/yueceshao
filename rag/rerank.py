"""Cross-Encoder 重排与检索质量门（《开发手册 v1.0》A7／A11，PR#6）。

职责
----
把召回阶段的候选（默认 40 条）用 Cross-Encoder 逐对打分重排，截到 ``top_k_out``
（默认 8），并把重排分**兼作质量门信号**：最高分不过阈值 τ 时不硬答，走降级重试，
仍不过门就诚实输出"知识库未覆盖"。

为什么需要这一层
----------------
召回（向量／BM25）是"两段编码后比对向量"，快但粗——它衡量的是表述相似度，无法
判断"这条条款到底能不能回答这个问题"。Cross-Encoder 把 query 与 passage 拼起来
一起过模型，能捕捉否定、条件、主体错配这类细粒度差别。代价是慢：40 条候选 ×
每对一次前向，所以它只能做重排，不能做召回。

三个必须说明的设计决策
----------------------
1. **重排分归一化到 [0, 1] 再谈阈值**。bge-reranker 输出的是单 logit（无界、可为负），
   直接拿它定 τ 会随查询长度漂移。统一 sigmoid 到概率尺度后，"0.3" 才有跨查询的
   可比性，标定出来的阈值也才可迁移。
2. **质量门不能只判"空结果"**。检索永远会返回 top-k（向量库不会说"没有"），
   所以"没检索到"这件事只能靠分数表达。只看 hits 是否为空，等于永不触发门控。
3. **不过门的唯一正确动作是诚实降级，不是硬答**。红线③禁止"合规/可以申报"类结论，
   放宽到质量门这一层就是：宁可告诉用户"知识库未覆盖"，也不能把 0.15 分的条款
   包装成依据。降级路径本身也受时效过滤约束——**不允许为了凑分数而放开失效条款**。
4. **所有"判不了"的分支都必须落到"不通过"**。质量门是唯一挡在硬答前面的闸，任何
   让闸失效的路径（模型吐 ``NaN``、阈值被配成 0、候选为空）都按不通过处理：``NaN``
   比较在 IEEE 语义下恒为 False，若不显式收口，``nan < tau`` 会判成"通过"，
   门就在不报任何错的情况下失效了。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "RerankerConfig",
    "PairScorer",
    "CrossEncoderScorer",
    "RerankedHit",
    "RerankResult",
    "GateDecision",
    "Reranker",
    "QualityGate",
    "GateRunner",
    "sigmoid",
    "resolve_model_source",
    "build_reranker",
]


HONEST_EMPTY = "知识库未检索到与该问题相关的政策条款。"
"""完全无候选时的诚实输出。"""

HONEST_LOW_SCORE = (
    "检索到的最相关条款与问题的匹配度不足（最高分 {score:.2f}，阈值 {tau:.2f}），"
    "知识库可能未覆盖该问题。建议补充更具体的项目要素后重试，或核对是否属于本省政策范围。"
)
"""有候选但不过门时的诚实输出。

注意措辞：只陈述"未覆盖 / 匹配度不足"，**不得**出现"合规""可以申报""建议申报"等
结论性判断（红线③）。
"""

NON_FINITE_LOGIT = -50.0
"""非有限分数（``NaN``／``±inf``）的替代值。

模型异常时（半精度溢出、空 passage）可能吐出非有限 logit。直接参与比较会让
``nan < tau`` 恒为 False，把质量门判成"通过"；直接透传给下游又会让 JSON 落盘
产生非法的 ``NaN``／``Infinity``。统一压到这个下界：``sigmoid(-50) ≈ 1.9e-22``，
既能被截断为 0.0、又保证排到任何真实分数之后，同时保持数值有限。
真正发生替换时一定会写一条 WARNING，不会静默。
"""

PLACEHOLDER_DOC_NO = re.compile(r"^无[（(]|未载|未标注|仅正文引用|不计")
"""占位文号的特征（``无（申报通知未载文号）`` 这类）。

全库约 1/3 的块是这种串，它们**不是文号**，混进引用位等于伪造（红线①）。
与 ``rag/retriever.py`` 的同名判定保持一致：这里再写一份是为了让本模块
在没有上游对象的场合也能 fail-closed，不依赖调用方记得先判。
"""


def sigmoid(value: float) -> float:
    """把 logit 压到 ``(0, 1)``。

    手写而不是用 ``numpy``／``scipy``：这里只处理几十个标量，引入依赖不值当；
    同时处理 ``|value|`` 很大时的溢出（``exp(-x)`` 上溢到 inf 会让结果变 0/1，
    对门控判断无害，但会产生 inf 告警）。

    :param value: 必须是有限值；非有限值请先经 :data:`NON_FINITE_LOGIT` 收口，
        本函数对 ``NaN`` 会原样返回 ``NaN``
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


# ================================================================ 配置


@dataclass(frozen=True)
class RerankerConfig:
    """重排与质量门参数，对应 ``config.json`` 的 ``reranker`` 段。

    :param model_name: Hub 上的模型名（``model_dir`` 缺失时兜底）
    :param model_dir: 本地权重目录（团队约定：权重放本地 ``models/`` 且不入库）
    :param device: ``cpu`` / ``cuda``
    :param top_k_in: 进入重排的候选数，手册 A7 取 40
    :param top_k_out: 重排后保留条数，手册 A7 取 8
    :param max_length: 单对 (query, passage) 的最大 token 数；政策条款较长，默认 512
    :param tau: 质量门阈值，作用在**归一化后的最高分**上。默认 0.55 由 40 条自建查询
        标定得出（见 ``docs/rag-rerank-标定.md``）：域内查询最低 0.605、完全域外最高
        0.507，取 0.55 落在两者之间的零误拦平台区。改阈值前请先复跑标定脚本
    :param enable_gate: 是否启用质量门。**默认开**；关掉仅用于消融对照，
        关掉后不过门的结果也会照常返回
    :param retry_factor: 降级重试时候选池的放大倍数；小于 2 等于重试拿同一批候选，
        所以配置层直接拒绝
    :param max_attempts: 最多尝试次数（1 = 不重试）
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_dir: str = "models/bge-reranker-v2-m3"
    device: str = "cpu"
    top_k_in: int = 40
    top_k_out: int = 8
    max_length: int = 512
    tau: float = 0.55
    enable_gate: bool = True
    retry_factor: int = 2
    max_attempts: int = 2

    def __post_init__(self) -> None:
        """校验参数区间。

        这些数字每一个都能把质量门"静默关掉"，所以宁可在构造时响亮地报错：
        ``tau<=0`` 让门恒过、``tau>=1`` 让门恒不过、``top_k_out<=0`` 让任何查询
        都输出"知识库未覆盖"、``retry_factor<2`` 让重试形同虚设。这些配置写在
        ``config.json`` 里，出问题不会有任何异常，只会悄悄产出错误结论。
        """
        if not isinstance(self.enable_gate, bool):
            raise ValueError(f"enable_gate 必须是 bool，收到 {self.enable_gate!r}")
        if not 0.0 < self.tau < 1.0:
            raise ValueError(f"tau 必须落在 (0, 1) 之间，收到 {self.tau!r}（0 会让门恒过）")
        if self.top_k_in < 1:
            raise ValueError(f"top_k_in 必须 ≥ 1，收到 {self.top_k_in!r}")
        if self.top_k_out < 1:
            raise ValueError(
                f"top_k_out 必须 ≥ 1，收到 {self.top_k_out!r}（0 会让每次查询都报未覆盖）"
            )
        if self.max_length < 1:
            raise ValueError(f"max_length 必须 ≥ 1，收到 {self.max_length!r}")
        if self.retry_factor < 2:
            raise ValueError(
                f"retry_factor 必须 ≥ 2，收到 {self.retry_factor!r}（1 表示重试不放大候选池）"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts 必须 ≥ 1，收到 {self.max_attempts!r}")

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> RerankerConfig:
        """从整份配置构造；``reranker`` 段缺字段回落到默认值。

        **只认顶层的 ``reranker`` 段**：直接传子段（``{"tau": 0.4}``）会被忽略并回落
        默认值。显式写出的数字一律交给 :meth:`__post_init__` 校验，不会被"默认值兜底"
        悄悄吞掉——除 ``max_attempts=0``，它的语义是"不重试"，收敛到 1。
        """
        section: Mapping[str, Any] = {}
        if isinstance(raw, Mapping):
            candidate = raw.get("reranker")
            if isinstance(candidate, Mapping):
                section = candidate
        defaults = cls()
        if not section:
            return defaults

        def pick(key: str, fallback: Any) -> Any:
            """取配置值：只有"没写"才回落默认，写 0 / 空串都不算没写。"""
            value = section.get(key)
            return fallback if value is None else value

        raw_gate = section.get("enable_gate")
        if raw_gate is None:
            enable_gate = defaults.enable_gate
        elif isinstance(raw_gate, bool):
            enable_gate = raw_gate
        else:
            # 不用 bool()：bool("false") 是 True，会让"关掉消融"写成字符串时反而开着
            raise ValueError(f"reranker.enable_gate 只接受 true/false，收到 {raw_gate!r}")

        raw_attempts = section.get("max_attempts")
        # 显式写 0 的意图是"不重试"，收敛到 1；没写才用默认值
        max_attempts = (
            max(1, int(raw_attempts)) if raw_attempts is not None else defaults.max_attempts
        )
        return cls(
            model_name=str(pick("model_name", defaults.model_name) or defaults.model_name),
            model_dir=str(pick("model_dir", defaults.model_dir)),
            device=str(pick("device", defaults.device) or defaults.device),
            top_k_in=int(pick("top_k_in", defaults.top_k_in)),
            top_k_out=int(pick("top_k_out", defaults.top_k_out)),
            max_length=int(pick("max_length", defaults.max_length)),
            tau=float(pick("tau", defaults.tau)),
            enable_gate=enable_gate,
            retry_factor=int(pick("retry_factor", defaults.retry_factor)),
            max_attempts=max_attempts,
        )


def resolve_model_source(config: RerankerConfig) -> str:
    """决定重排权重的来源：本地目录优先，留空才走 Hub。

    **不允许静默退回 Hub**——``model_dir`` 写了但目录不存在时，原来的实现会当作
    "没配"去 ``HuggingFace`` 下载 2.2GB 权重，代理不通时表现为
    ``ProxyError: 502 Bad Gateway``，根因被这层兜底完全掩盖。这里直接报错，
    把选择权交回给配置。

    两个打分器（:class:`CrossEncoderScorer`／:class:`FlagEmbeddingScorer`）共用本函数，
    避免"同一份配置在两个实现下行为不同"。

    :param config: 重排参数
    :return: 可直接交给模型构造器的路径或 Hub 模型名
    :raises FileNotFoundError: ``model_dir`` 非空但目录不存在
    """
    configured = str(config.model_dir or "").strip()
    if not configured:
        return config.model_name
    if not Path(configured).is_dir():
        raise FileNotFoundError(
            f"重排权重目录不存在：{configured}。团队约定权重放本地 models/ 且不入库，"
            f"请先下载 bge-reranker-v2-m3 并把 config.json 的 reranker.model_dir "
            f"指向绝对路径；若确实要走 Hub，请把 model_dir 留空只用 model_name。"
        )
    return configured


# ================================================================ 打分器


class PairScorer(Protocol):
    """逐对打分器协议：测试用假实现替换，不必装 torch／模型权重。"""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """返回每段的原始 logit（未归一化）。"""
        ...


class CrossEncoderScorer:
    """用 sentence-transformers 的 ``CrossEncoder`` 加载 bge-reranker-v2-m3。

    :param config: 重排参数
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model: Any = None

    def _load(self) -> Any:
        """懒加载模型（首次打分时才读权重）。"""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            source = self._resolve_source()
            logger.info("加载重排模型：%s（device=%s）", source, self.config.device)
            self._model = CrossEncoder(
                source,
                device=self.config.device,
                max_length=self.config.max_length,
            )
        return self._model

    def _resolve_source(self) -> str:
        """确定从哪加载权重；规则见模块级 :func:`resolve_model_source`。"""
        return resolve_model_source(self.config)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """逐对打分，返回 logit。"""
        if not passages:
            return []
        model = self._load()
        pairs = [(query, passage) for passage in passages]
        raw = model.predict(pairs, batch_size=8, show_progress_bar=False)
        return [float(value) for value in raw]


class FlagEmbeddingScorer:
    """用 FlagEmbedding 的 ``FlagReranker`` 加载（手册技术栈指定的加载方式）。

    与 :class:`CrossEncoderScorer` 等价，区别是依赖更重（会拉 datasets／peft）。
    在要求严格的部署环境里优先用它；两者分数口径一致，都是原始 logit。

    :param config: 重排参数
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model: Any = None

    def _load(self) -> Any:
        """懒加载模型。"""
        if self._model is None:
            from FlagEmbedding import FlagReranker

            source = resolve_model_source(self.config)
            logger.info("加载重排模型（FlagEmbedding）：%s", source)
            self._model = FlagReranker(
                source, use_fp16=self.config.device.startswith("cuda")
            )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """逐对打分，返回 logit。"""
        if not passages:
            return []
        model = self._load()
        raw = model.compute_score(
            [[query, passage] for passage in passages], normalize=False
        )
        if isinstance(raw, (int, float)):  # 只有一对时 FlagEmbedding 返回标量
            return [float(raw)]
        return [float(value) for value in raw]


# ================================================================ 数据结构


@dataclass(frozen=True)
class RerankedHit:
    """重排后的一条命中。

    :param hit: 原始命中（:class:`~rag.retriever.Hit` 或 :class:`~rag.hybrid.FusedHit`）
    :param rerank_score: 归一化后的重排分 ``sigmoid(logit)``，在 ``(0, 1)``
    :param logit: 模型原始输出，便于与阈值／论文数据对照；模型吐非有限值时
        统一记成 :data:`NON_FINITE_LOGIT`（同时写 WARNING），不会有 ``NaN`` 落进报告
    :param rank: 重排后的排名（1 起）
    :param source_rank: 重排前的排名（1 起），用来观察重排改动了多少
    """

    hit: Any
    rerank_score: float
    logit: float
    rank: int
    source_rank: int

    @property
    def chunk_id(self) -> str:
        """块 id。"""
        return str(getattr(self.hit, "chunk_id", ""))

    @property
    def text(self) -> str:
        """块正文。"""
        return str(getattr(self.hit, "text", ""))

    @property
    def metadata(self) -> Mapping[str, Any]:
        """块元数据。"""
        value = getattr(self.hit, "metadata", None)
        return value if isinstance(value, Mapping) else {}

    @property
    def citation(self) -> str:
        """严格引用串（透传原始命中）。

        没有真文号时给空串：占位文号拼进引用位就是伪造（红线①），报告层只要
        见到非空引用就当成"有文号可回溯"。
        """
        if not self.has_doc_no:
            return ""
        return str(getattr(self.hit, "citation", "") or "")

    @property
    def reference(self) -> str:
        """展示用引用（透传原始命中）。"""
        return str(getattr(self.hit, "reference", "") or "")

    @property
    def doc_no(self) -> str:
        """发文字号。"""
        return str(getattr(self.hit, "doc_no", "") or "")

    @property
    def clause(self) -> str:
        """条款号。"""
        return str(getattr(self.hit, "clause", "") or "")

    @property
    def has_doc_no(self) -> bool:
        """是否真的有文号。

        占位串（``无（申报通知未载文号）`` 之类）不算文号——全库约 1/3 的块是这种。
        以原始命中的判断为准（``Hit.has_doc_no`` 用的是 loader 正则结论 + metadata 字段）；
        没有该属性的对象退回本模块自己的占位判定，**不退回"非空即真"**——那正是
        让占位串混进引用位的那个 bug（红线①）。
        """
        flag = getattr(self.hit, "has_doc_no", None)
        if isinstance(flag, bool):
            return flag
        if not self.doc_no:
            return False
        return PLACEHOLDER_DOC_NO.search(self.doc_no) is None

    @property
    def is_current(self) -> bool:
        """是否现行有效（红线③：失效政策不得作为依据）。

        转发而不是让下游自己读 ``metadata``：正确判法是 ``metadata["is_current"] is True``，
        用 ``bool()`` 会把字符串 ``"false"`` 当成现行有效。
        """
        flag = getattr(self.hit, "is_current", None)
        if isinstance(flag, bool):
            return flag
        return self.metadata.get("is_current") is True

    @property
    def title(self) -> str:
        """文件标题。"""
        return str(getattr(self.hit, "title", "") or "")

    @property
    def validity(self) -> str:
        """时效字段原文，供报告层展示。"""
        return str(getattr(self.hit, "validity", "") or "")

    @property
    def publish_date(self) -> str:
        """发布日期。"""
        return str(getattr(self.hit, "publish_date", "") or "")

    @property
    def rank_delta(self) -> int:
        """排名变化：正数表示重排把它往前提了。"""
        return self.source_rank - self.rank

    def as_dict(self) -> dict[str, Any]:
        """可序列化形式，供评测脚本落 JSONL。"""
        base: dict[str, Any]
        if hasattr(self.hit, "as_dict"):
            base = dict(self.hit.as_dict())
        else:  # pragma: no cover - 协议兜底
            base = {"chunk_id": self.chunk_id, "text": self.text}
        base.update(
            {
                "rerank_score": self.rerank_score,
                "rerank_logit": self.logit,
                "rank": self.rank,
                "source_rank": self.source_rank,
                "rank_delta": self.rank_delta,
            }
        )
        return base


@dataclass(frozen=True)
class GateDecision:
    """质量门判定结果。

    :param passed: 是否过门
    :param reason: ``pass`` / ``below_tau`` / ``empty_query`` / ``empty_candidates`` /
        ``gate_disabled``（消融）／``ungated``（只重排、没走门控）
    :param top_score: 归一化后的最高分；无候选时为 0
    :param tau: 本次使用的阈值
    :param message: 给用户看的说明；过门时为空串
    """

    passed: bool
    reason: str
    top_score: float
    tau: float
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        """可序列化形式。"""
        return {
            "passed": self.passed,
            "reason": self.reason,
            "top_score": round(self.top_score, 6),
            "tau": self.tau,
        }


@dataclass
class RerankResult:
    """一次重排（含门控尝试）的结果。

    :param query: 原始查询
    :param hits: 重排后命中
    :param gate: 门控判定。**默认是"未过门"**——手写构造出来的结果不会天然带上
        "已通过体检"的姿态，必须经 :class:`QualityGate` 判过才算数
    :param attempts: 实际尝试次数（1 = 未触发重试）
    :param candidate_count: 最后一次尝试进入重排的候选数
    """

    query: str
    hits: list[RerankedHit] = field(default_factory=list)
    gate: GateDecision = field(
        default_factory=lambda: GateDecision(False, "ungated", 0.0, 0.0)
    )
    attempts: int = 1
    candidate_count: int = 0

    def summary(self) -> str:
        """一行摘要，含门控结论。

        门控没真正跑过时**不许打出 ``≥ τ`` 的不等式**：消融组日志要是也印
        "最高分 0.002 ≥ τ 0.99"，对照实验就分不清哪组是哪组了。
        """
        head = f"「{self.query}」→ 重排 {len(self.hits)} 块"
        if self.gate.passed and self.gate.reason == "pass":
            return (
                f"{head}（最高分 {self.gate.top_score:.3f} ≥ τ {self.gate.tau:.2f}，"
                f"尝试 {self.attempts} 次）"
            )
        if self.gate.reason == "gate_disabled":
            return (
                f"{head}（门控已关／消融对照，最高分 {self.gate.top_score:.3f}，"
                f"τ {self.gate.tau:.2f} 仅供参考，尝试 {self.attempts} 次）"
            )
        if self.gate.reason == "ungated":
            return f"{head}（仅重排，未施加质量门）"
        return (
            f"{head}，**未过质量门**（{self.gate.reason}，最高分 "
            f"{self.gate.top_score:.3f} < τ {self.gate.tau:.2f}，尝试 {self.attempts} 次）"
        )

    def citations(self) -> list[str]:
        """命中里的严格引用串（去重保序），空串已剔除。

        **未过门时返回空列表**：门控没过就说明这些条款不足以作为依据（红线③），
        把它当引用交给报告层等于绕过质量门。要诊断"为什么没命中"请直接读 ``hits``。
        """
        if not self.gate.passed:
            return []
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.citation:
                seen.setdefault(hit.citation, None)
        return list(seen)

    def doc_nos(self) -> list[str]:
        """命中涉及的文号（去重保序）。

        只收**真有文号**的块：占位串（``无（申报通知未载文号）``）不算，见红线①。
        未过门时同样返回空列表。
        """
        if not self.gate.passed:
            return []
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.has_doc_no and hit.doc_no:
                seen.setdefault(hit.doc_no, None)
        return list(seen)


# ================================================================ 重排与门控


class Reranker:
    """纯重排：打分、排序、截断。不做门控判断。

    :param scorer: 逐对打分器
    :param config: 重排参数
    """

    def __init__(self, scorer: PairScorer, config: RerankerConfig | None = None) -> None:
        self.scorer = scorer
        self.config = config or RerankerConfig()

    def rerank(
        self,
        query: str,
        hits: Sequence[Any],
        top_k: int | None = None,
        top_k_in: int | None = None,
    ) -> RerankResult:
        """对候选重排。

        :param query: 用户问题原文
        :param hits: 候选命中（任何带 ``text`` 属性的对象）
        :param top_k: 覆盖配置里的 ``top_k_out``
        :param top_k_in: 覆盖配置里的 ``top_k_in``。**降级重试时必须传**——
            否则调用方辛苦放大的候选池会在这里被切回原长度，重试等于没做
        :return: 重排结果；``gate`` 字段此处只反映"有无候选"，门控判定交给
            :class:`QualityGate`／:class:`GateRunner`
        """
        cfg = self.config
        text = (query or "").strip()
        limit_in = cfg.top_k_in if top_k_in is None else int(top_k_in)
        candidates = list(hits)[:limit_in]
        limit = cfg.top_k_out if top_k is None else int(top_k)

        if not text or not candidates:
            # 两种空要分开报：空 query 是上游传错，空候选才是"真的没检索到"。
            # 混成一句会让排查的人去翻索引，而问题其实在调用方。
            reason = "empty_query" if not text else "empty_candidates"
            return RerankResult(
                query=text,
                hits=[],
                gate=GateDecision(
                    passed=False,
                    reason="gate_disabled" if not cfg.enable_gate else reason,
                    top_score=0.0,
                    tau=cfg.tau,
                    message="" if not cfg.enable_gate else HONEST_EMPTY,
                ),
                candidate_count=len(candidates),
            )

        passages = [str(getattr(hit, "text", "")) for hit in candidates]
        logits = [float(value) for value in self.scorer.score(text, passages)]
        if len(logits) != len(candidates):
            raise RuntimeError(
                f"打分器返回 {len(logits)} 个分数，但候选有 {len(candidates)} 条"
            )
        invalid = sum(1 for value in logits if not math.isfinite(value))
        if invalid:
            # fail-closed：压到下界而不是丢弃或放行。若全部无效，最高分会停在 0.0，
            # 门自然判不过——比"nan 比较为 False 于是通过"安全得多。
            logger.warning(
                "打分器返回 %d/%d 个非有限分数（NaN/Inf），已压到 %.0f 处理；"
                "若持续出现请检查模型精度与空 passage",
                invalid,
                len(logits),
                NON_FINITE_LOGIT,
            )
            logits = [
                value if math.isfinite(value) else NON_FINITE_LOGIT for value in logits
            ]

        # 带上重排前的位置再排序：不能用 id(hit) 做键，同一个对象出现两次时
        # 两条会拿到同一位置，rank_delta 跟着算错
        scored = sorted(
            zip(candidates, logits, range(1, len(candidates) + 1), strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked = [
            RerankedHit(
                hit=hit,
                rerank_score=round(sigmoid(logit), 6),
                logit=round(logit, 6),
                rank=rank,
                source_rank=source_rank,
            )
            for rank, (hit, logit, source_rank) in enumerate(scored[:limit], start=1)
        ]

        return RerankResult(
            query=text,
            hits=ranked,
            # reason="ungated"：这里只做了重排，还没过门。写成 "pass" 会让
            # summary() 打出"≥ τ"的假不等式，也让手写构造的结果显得已体检
            gate=GateDecision(
                False, "ungated", ranked[0].rerank_score if ranked else 0.0, cfg.tau
            ),
            candidate_count=len(candidates),
        )


class QualityGate:
    """质量门：只判"最高分够不够"，不做重试。

    :param config: 重排参数（读 ``tau`` 与 ``enable_gate``）
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()

    def evaluate(self, result: RerankResult) -> GateDecision:
        """判定一次重排结果是否过门。

        :param result: 重排结果
        :return: 门控判定；``enable_gate=False`` 时恒为通过（消融对照用），
            但 ``reason`` 标成 ``gate_disabled``，好让日志区分"真通过"与"没开门"
        """
        cfg = self.config
        if not cfg.enable_gate:
            return GateDecision(True, "gate_disabled", result.gate.top_score, cfg.tau)

        if not result.hits:
            return GateDecision(False, "empty_candidates", 0.0, cfg.tau, HONEST_EMPTY)

        # 取最高分而不取 hits[0]：门控不依赖入参已按分数降序（手写乱序也能判对）
        top = max(hit.rerank_score for hit in result.hits)
        if not math.isfinite(top):
            # 纵深防御：非有限分数在 rerank() 里已被压到 0，但手写构造的
            # RerankResult 可能绕过那一层。NaN < tau 恒为 False，必须先收口
            logger.warning("最高分不是有限值（%r），按不过门处理", top)
            top = 0.0
        if top < cfg.tau:
            return GateDecision(
                False,
                "below_tau",
                top,
                cfg.tau,
                HONEST_LOW_SCORE.format(score=top, tau=cfg.tau),
            )
        return GateDecision(True, "pass", top, cfg.tau)


class GateRunner:
    """重排＋门控＋降级重试的完整流程。

    降级路径（A11）：不过门 → 放大候选池重试 → 仍不过门 → 诚实输出"未覆盖"。

    为什么重试只放大候选池、不放宽时效过滤
    ------------------------------------
    放大候选是"再多找一些依据"；而放开时效过滤会把已失效的条款拉回结果里，
    那是**用错误换分数**。所以重试的动作严格限定在"取更多候选"，
    时效条件与审计公告排除一律保持不动。

    但要说清代价：``max(sigmoid)`` 随候选池单调不减，同一个 τ 在更大的池子上
    必然更容易过。τ=0.55 是在 **top-40** 上标定的，所以第二轮（默认取 80 条）
    的真实漏放率高于 ``docs/rag-rerank-标定.md`` 里那组数字。若要做到口径一致，
    标定时必须按 ``max_attempts`` 复现同样的池子大小。

    :param reranker: 重排器
    :param gate: 质量门
    :param config: 重排参数（读重试倍数与次数）
    """

    def __init__(
        self,
        reranker: Reranker,
        gate: QualityGate | None = None,
        config: RerankerConfig | None = None,
    ) -> None:
        self.reranker = reranker
        self.config = config or reranker.config
        self.gate = gate or QualityGate(self.config)

    def run(
        self,
        query: str,
        fetch: Callable[[int], Sequence[Any]],
        top_k: int | None = None,
    ) -> RerankResult:
        """跑完整流程。

        :param query: 用户问题原文
        :param fetch: 取候选的回调，参数是"要多少条"；调用方负责时效过滤与红线④
        :param top_k: 覆盖 ``top_k_out``
        :return: 最终结果；``gate.passed=False`` 时 ``hits`` 仍保留（供诊断与
            报告里"为什么没命中"的说明），但下游**不得**把它当作依据输出
        """
        cfg = self.config
        attempts = max(1, cfg.max_attempts)
        attempt = 0
        result: RerankResult | None = None
        previous_pool = -1

        for attempt in range(1, attempts + 1):
            size = cfg.top_k_in * (cfg.retry_factor ** (attempt - 1))
            candidates = list(fetch(size))
            # 必须把放大后的 size 传给 rerank：否则内部按 top_k_in 又切回原长度，
            # 重试就成了摆设（这个 bug 是 test_gate_retries_with_larger_pool 抓到的）
            result = self.reranker.rerank(query, candidates, top_k=top_k, top_k_in=size)
            result.attempts = attempt
            decision = self.gate.evaluate(result)
            result.gate = decision
            if decision.passed:
                logger.info("%s", result.summary())
                return result
            if attempt < attempts:
                # 报实际拿到的候选数而不是请求的 size：语料池比 size 小时
                # "候选 40 → 80" 是假的，重试其实在重复喂同一批
                if len(candidates) <= previous_pool:
                    logger.info(
                        "候选池已饱和（%d 条），继续放大不再增益，停止重试", len(candidates)
                    )
                    break
                logger.info(
                    "未过质量门（%s，最高分 %.3f < τ %.2f），降级重试：候选 %d → 请求 %d",
                    decision.reason,
                    decision.top_score,
                    decision.tau,
                    len(candidates),
                    size * cfg.retry_factor,
                )
                previous_pool = len(candidates)

        assert result is not None  # attempts >= 1 保证至少跑一次
        logger.warning("%s", result.summary())
        return result


# ================================================================ 装配与命令行


def build_reranker(
    config: Mapping[str, Any] | None = None,
    *,
    scorer: PairScorer | None = None,
    prefer_flag_embedding: bool = False,
) -> Reranker:
    """按本机配置装配重排器。

    :param config: 整份配置
    :param scorer: 指定打分器（测试／替换模型时用）
    :param prefer_flag_embedding: ``True`` 时优先用 FlagEmbedding 加载（手册技术栈），
        否则用 sentence-transformers 的 CrossEncoder；两者分数口径一致
    :return: 可直接使用的 :class:`Reranker`
    """
    cfg = RerankerConfig.from_config(config)
    if scorer is None:
        scorer = FlagEmbeddingScorer(cfg) if prefer_flag_embedding else CrossEncoderScorer(cfg)
    return Reranker(scorer, cfg)


def _load_config(path: Path | str) -> dict[str, Any]:
    """读取本机配置；文件缺失时返回空配置（各段回落默认值）。"""
    target = Path(path)
    if not target.is_file():
        logger.warning("配置文件不存在，使用默认参数：%s", target)
        return {}
    return json.loads(target.read_text(encoding="utf-8-sig"))


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：``python -m rag.rerank "查询" --config config.json``。

    检索用 PR#4 的稠密检索器（已含 A6 时效硬过滤）；本模块只负责重排与门控。
    """
    parser = argparse.ArgumentParser(description="粤策哨 Cross-Encoder 重排与质量门")
    parser.add_argument("query", nargs="*", help="查询文本（多个词会自动拼接）")
    parser.add_argument("--config", default="config.json", help="本机配置文件路径")
    parser.add_argument("--top-k", type=int, default=None, help="重排后保留条数")
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="关闭质量门（仅供消融对照；关掉后不过门的结果也会照常输出）",
    )
    parser.add_argument("--show-text", action="store_true", help="打印命中块正文")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = " ".join(args.query).strip()
    if not query:
        parser.error("请提供查询文本")

    raw_config = _load_config(args.config)
    config = RerankerConfig.from_config(raw_config)
    if args.no_gate:
        config = replace(config, enable_gate=False)

    from rag.retriever import build_retriever

    retriever = build_retriever(raw_config)
    # 用最终生效的 config 装配，而不是"先按 raw_config 装好再改属性"：后者会让
    # reranker.config 与 scorer 手里的 config 变成两份，下次加字段时只改到一份
    reranker = build_reranker({"reranker": asdict(config)})
    runner = GateRunner(reranker, QualityGate(config), config)

    retrievals: list[Any] = []

    def fetch(size: int) -> Sequence[Any]:
        """取候选：时效过滤由检索器下发到向量库（A6）。

        直接按 ``size`` 取，**不要**再截到 ``top_k_in``——降级重试靠的就是"取更多"，
        在取数这一层截断会让重试机制彻底失效。顺手留下 ``RetrievalResult``，
        好在下面核对时效过滤到底有没有生效。
        """
        retrieval = retriever.retrieve(query, top_k=size, count_candidates=False)
        retrievals.append(retrieval)
        return retrieval.hits

    result = runner.run(query, fetch, top_k=args.top_k)

    if retrievals and not retrievals[-1].temporal_filtered:
        # 红线③：失效政策不得进入答案。这条只能靠配置保证，而配置可能被消融改写，
        # 所以至少要在输出里留痕——否则"重排把失效条款排进 top-8 且门判通过"
        # 会一路无声地走进报告
        logger.warning(
            "本次检索**没有**施加时效过滤（where=%s），结果可能含已失效条款；"
            "仅 A6 消融对照允许如此，正式产物需打开 retrieval.enable_temporal_filter",
            retrievals[-1].where,
        )

    print(result.summary())
    for hit in result.hits:
        print(
            f"{hit.rank:>2}. [{hit.rerank_score:.4f}] "
            f"(原第 {hit.source_rank}) {hit.reference or hit.chunk_id}"
        )
        if args.show_text:
            print(f"    {hit.text[:120].replace(chr(10), ' ')}")
    if not result.gate.passed:
        print()
        print(result.gate.message)
    return 0 if result.gate.passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
