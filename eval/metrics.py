"""三项评测指标的统计口径（3.4 评测与消融 · PR2 · feat/eval-runner）。

手册 3.4 给的验收线是 **匹配 ≥85% / 召回 ≥80% / 引用错误 ≤5%**。本模块把三个词
落成可复算的确定定义，避免"指标好看但口径说不清"：

1. **匹配率（match_rate）**——判定一致率。
   每条案例系统给出 ``verdict``（``hit`` 命中负面清单 / ``miss`` 未命中），与标注的
   ``expected_verdict`` 比对。分母只计**已标注判定**的案例；未标注的案例进明细并计入
   ``verdict_unlabeled``，不参与该指标。

2. **召回率（recall）**——正例条款召回。
   只在标注为 ``hit`` 且给了 ``expected_refs`` 的案例上计算（micro 口径）：
   ``Σ|预测 ∩ 标注| / Σ|标注|``。只标了 ``hit`` 没给具体条款的案例，退化为
   "整案命中即算 1 条标注"，并在明细 ``recall_mode`` 里标 ``case-level``。
   没有可计算样本时返回 ``None``（报告里显示 N/A），绝不拿 0 或 1 冒充。

3. **引用错误率（citation_error_rate）**——引用可信度。
   统计系统输出里的每条引用（文号/条款标识）是否能在 registry（文号注册表元数据）中
   核验到；未提供 registry 时该指标为 ``None``（N/A），因为无法判断真假时不能默认它是对的。
   另单独统计 ``forbidden_phrasing``：输出里出现"合规/可以申报"类姿态禁语即计一次违规。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from eval.dataset import EvalCase

__all__ = [
    "CasePrediction",
    "CaseOutcome",
    "EvalThresholds",
    "MetricsSummary",
    "FORBIDDEN_PHRASES",
    "REPORT_DISCLAIMER",
    "DISCLAIMER_LINE_HINTS",
    "strip_disclaimer",
    "evaluate_case",
    "summarize",
]

logger = logging.getLogger(__name__)

#: 姿态禁语（红线第 3 条）：输出中出现即视为违规，不论上下文语气。
#: 既包括「合规 / 可以申报」这类正面放行结论，也包括「违规 / 不能申报」这类反向判定
#: ——两者都是系统在替主管部门下结论，同属红线禁止的姿态。
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "合规",
    "可以申报",
    "建议申报",
    "符合申报条件",
    "准予申报",
    "违规",
    "不能申报",
)

#: 渲染层强制注入的免责声明（红线第 3 条：姿态结论由渲染层兜底，不靠 LLM 自觉）。
#: 它本身含「合规」二字，扫描姿态禁语前必须先剔除，否则接上真实管线后每一份报告
#: 都会因这句免责声明被误判为姿态违规。
REPORT_DISCLAIMER: str = (
    "本系统仅为申报前自查辅助参考，不构成合规判定，请以政策法规原文及主管部门意见为准。"
)

#: 免责声明可能的不同措辞片段；整行命中任一片段即整行剔除，不参与禁语扫描。
#: 用「行」为粒度是因为免责声明在报告里总是独立成段，按行剔除不会误伤正文。
DISCLAIMER_LINE_HINTS: tuple[str, ...] = (
    "不构成合规判定",
    "不构成政策判断",
    "仅为申报前自查辅助参考",
    "以政策法规原文及主管部门意见为准",
)


def strip_disclaimer(text: str) -> str:
    """剔除渲染层强制的免责声明，返回可参与姿态扫描的正文。

    两步：先删掉与 :data:`REPORT_DISCLAIMER` 完全相同的片段，再按行丢弃含
    :data:`DISCLAIMER_LINE_HINTS` 的免责声明行。渲染层改措辞时只要往 hints 里补一条即可。

    :param text: 报告全文
    :return: 去掉免责声明后的文本（无免责声明时原样返回）
    """
    if not text:
        return ""
    stripped = text.replace(REPORT_DISCLAIMER, "")
    kept_lines = [
        line
        for line in stripped.splitlines()
        if not any(hint in line for hint in DISCLAIMER_LINE_HINTS)
    ]
    return "\n".join(kept_lines)



@dataclass(frozen=True)
class EvalThresholds:
    """验收线，默认取手册 3.4 的三个数字。"""

    match_rate: float = 0.85
    recall: float = 0.80
    citation_error_rate: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return {
            "match_rate": self.match_rate,
            "recall": self.recall,
            "citation_error_rate": self.citation_error_rate,
        }


@dataclass(frozen=True)
class CasePrediction:
    """系统对一条案例的输出（由被测管线产生，评测侧只读）。

    :param verdict: ``hit`` / ``miss``
    :param refs: 命中的条款标识集合（应为 registry 里的已核验标识）
    :param citations: 报告中出现的引用标识（用于引用错误率）
    :param text: 报告全文（仅用于姿态禁语扫描，可选）。照实传全文即可——渲染层强制的
        免责声明会在扫描前由 :func:`strip_disclaimer` 剔除，不需要调用方自己裁剪。
    """

    verdict: str
    refs: tuple[str, ...] = field(default_factory=tuple)
    citations: tuple[str, ...] = field(default_factory=tuple)
    text: str = ""


@dataclass
class CaseOutcome:
    """单条案例的评测明细（逐案例输出即为此结构的序列化）。"""

    case_id: str
    expected_verdict: str | None
    predicted_verdict: str
    verdict_match: bool | None
    expected_refs: tuple[str, ...]
    predicted_refs: tuple[str, ...]
    refs_hit: tuple[str, ...]
    recall_mode: str
    citations_total: int
    citations_bad: int
    forbidden_hits: tuple[str, ...]
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        """扁平化，便于写 CSV。"""
        return {
            "case_id": self.case_id,
            "expected_verdict": self.expected_verdict or "",
            "predicted_verdict": self.predicted_verdict,
            "verdict_match": "" if self.verdict_match is None else int(self.verdict_match),
            "expected_refs": ";".join(self.expected_refs),
            "predicted_refs": ";".join(self.predicted_refs),
            "refs_hit": ";".join(self.refs_hit),
            "recall_mode": self.recall_mode,
            "citations_total": self.citations_total,
            "citations_bad": self.citations_bad,
            "forbidden_hits": ";".join(self.forbidden_hits),
            "notes": self.notes,
        }


def _scan_forbidden(text: str) -> tuple[str, ...]:
    """扫描姿态禁语，返回命中词元组。

    先剔除渲染层免责声明（它含「合规」二字）再扫，否则真实管线的每份报告都会被误判。
    """
    body = strip_disclaimer(text)
    # 条款原文段豁免：逐字引用的政策原文可能天然含「合规发票」「违规举债」等字样，
    # 那是语料本身而非系统措辞——姿态禁语只约束系统自己的结论（预审与真数据首跑发现）。
    segments = re.split(r"条款原文[：:]", body)
    outside_clauses = "".join(segments[::2])  # 奇数段为原文引文，扫描偶数段（系统措辞）
    return tuple(phrase for phrase in FORBIDDEN_PHRASES if phrase in outside_clauses)


def evaluate_case(
    case: EvalCase,
    prediction: CasePrediction,
    registry: Mapping[str, Any] | None = None,
) -> CaseOutcome:
    """评测单条案例。

    :param case: 标注案例
    :param prediction: 系统输出
    :param registry: 文号注册表（标识 → 元数据）；``None`` 表示无法核验引用
    """
    verdict_match: bool | None = None
    if case.expected_verdict is not None:
        verdict_match = prediction.verdict == case.expected_verdict

    predicted = set(prediction.refs)
    expected = set(case.expected_refs)
    refs_hit = tuple(ref for ref in case.expected_refs if ref in predicted)

    if case.expected_verdict == "hit" and expected:
        recall_mode = "ref-level"
    elif case.expected_verdict == "hit":
        recall_mode = "case-level"
    else:
        recall_mode = "n/a"

    citations_total = len(prediction.citations)
    if registry is None:
        citations_bad = 0
    else:
        citations_bad = sum(1 for ref in prediction.citations if ref not in registry)

    return CaseOutcome(
        case_id=case.case_id,
        expected_verdict=case.expected_verdict,
        predicted_verdict=prediction.verdict,
        verdict_match=verdict_match,
        expected_refs=case.expected_refs,
        predicted_refs=prediction.refs,
        refs_hit=refs_hit,
        recall_mode=recall_mode,
        citations_total=citations_total,
        citations_bad=citations_bad,
        forbidden_hits=_scan_forbidden(prediction.text),
        notes=case.notes,
    )


@dataclass
class MetricsSummary:
    """三项指标的汇总结果。

    ``None`` 一律表示"该指标在本次数据上不可计算"，报告须显示 N/A 而不是 0。
    """

    case_count: int
    labeled_verdict_count: int
    verdict_unlabeled: int
    match_rate: float | None
    recall: float | None
    recall_denominator: int
    citation_total: int
    citation_bad: int
    citation_error_rate: float | None
    forbidden_violations: int
    thresholds: EvalThresholds

    def passes(self) -> dict[str, bool | None]:
        """逐项与验收线比对；不可计算的指标记 ``None``。"""
        return {
            "match_rate": (
                None if self.match_rate is None
                else self.match_rate >= self.thresholds.match_rate),
            "recall": None if self.recall is None else self.recall >= self.thresholds.recall,
            "citation_error_rate": (
                None
                if self.citation_error_rate is None
                else self.citation_error_rate <= self.thresholds.citation_error_rate
            ),
        }

    def all_pass(self) -> bool:
        """三项指标是否全部达标；有 N/A 项视为未达标（不能靠缺失项过关）。"""
        verdicts = self.passes().values()
        return all(flag is True for flag in verdicts)

    def to_dict(self) -> dict[str, Any]:
        """序列化为报告 JSON。"""
        return {
            "case_count": self.case_count,
            "labeled_verdict_count": self.labeled_verdict_count,
            "verdict_unlabeled": self.verdict_unlabeled,
            "metrics": {
                "match_rate": self.match_rate,
                "recall": self.recall,
                "citation_error_rate": self.citation_error_rate,
                "recall_denominator": self.recall_denominator,
                "citation_total": self.citation_total,
                "citation_bad": self.citation_bad,
                "forbidden_violations": self.forbidden_violations,
            },
            "thresholds": self.thresholds.to_dict(),
            "passes": self.passes(),
            "all_pass": self.all_pass(),
        }


def summarize(
    outcomes: Sequence[CaseOutcome],
    thresholds: EvalThresholds,
    registry_provided: bool,
) -> MetricsSummary:
    """把逐案例明细汇总成三项指标。

    :param outcomes: 逐案例结果
    :param thresholds: 验收线
    :param registry_provided: 是否提供了 registry（决定引用错误率能否计算）
    """
    labeled = [item for item in outcomes if item.verdict_match is not None]
    unlabeled = len(outcomes) - len(labeled)
    match_rate = (
        sum(1 for item in labeled if item.verdict_match) / len(labeled) if labeled else None
    )

    recall_denominator = 0
    recall_numerator = 0
    for item in outcomes:
        if item.recall_mode == "ref-level":
            recall_denominator += len(item.expected_refs)
            recall_numerator += len(item.refs_hit)
        elif item.recall_mode == "case-level":
            recall_denominator += 1
            recall_numerator += 1 if item.predicted_verdict == "hit" else 0
    recall = recall_numerator / recall_denominator if recall_denominator else None

    citation_total = sum(item.citations_total for item in outcomes)
    citation_bad = sum(item.citations_bad for item in outcomes)
    citation_error_rate = (
        citation_bad / citation_total if registry_provided and citation_total else None
    )

    return MetricsSummary(
        case_count=len(outcomes),
        labeled_verdict_count=len(labeled),
        verdict_unlabeled=unlabeled,
        match_rate=match_rate,
        recall=recall,
        recall_denominator=recall_denominator,
        citation_total=citation_total,
        citation_bad=citation_bad,
        citation_error_rate=citation_error_rate,
        forbidden_violations=sum(1 for item in outcomes if item.forbidden_hits),
        thresholds=thresholds,
    )
