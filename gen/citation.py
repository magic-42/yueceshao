"""gen/citation.py — 标记式引用渲染（A8）：[Cn] → 文号/条款号/生效日期由元数据注入。

职责（手册 §3.3 PR2）：
1. ChunkMeta：检索块元数据的渲染侧视图（标记 → 文号/条款号/生效日期/效力状态），
   ``from_hit`` 以鸭子类型适配检索侧命中对象（任何带 ``chunk_id``/``metadata`` 的
   对象，如 rag.retriever.Hit／rag.rerank 的命中），``from_mapping`` 适配裸字典；
2. build_report：把 LLM 输出的 GeneratedSections＋②段 NegativeHit 组装为
   CheckupReport——引用逐条查表解析，**置信标签在此由确定性信号填充**（§4.8：
   时效状态＋命中类型，LLM 不自评）；
3. render_report：CheckupReport → 四段式文本（供终端／demo／app 使用），
   **免责声明由本层强制附加**（渲染层唯一出口，LLM 不生成）。

红线落地（README 结构性判断之二「LLM 永远不手打文号」）：
- 文号/条款号/生效日期只从元数据表读取；引用解析失败时输出诚实的
  「依据标注缺失」占位（不猜、不补、不编）；
- 无真文号的块按检索侧同款 ``has_doc_no`` 口径退化为标题引用（registry 权威
  字段，非编造）；占位串（「无（页面未载…）」）永不出现成品文本；
- 入参的 LLM 文本重过 schema 的禁语/文号防线（防绕过 parse_generated 直接构造）。

置信口径（确定性信号 → §4.8 三档）：
- 明确适用：现行有效条款；
- 建议确认：待人工核验条款，或时效状态不明；
- 未覆盖：该段在生成侧已声明 uncovered（继承，不渲染条目）。
已失效条款不进①段（检索侧已过滤，此处为渲染层双保险）：剔除并记 warning，
剔除后该段无条目则转「未覆盖」。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from gen.schema import (
    CONFIDENCE_EXPLICIT,
    CONFIDENCE_SUGGEST,
    DISCLAIMER_TEXT,
    CheckupReport,
    GeneratedSections,
    MissingItem,
    NegativeHit,
    PolicyItem,
    SuggestionItem,
    check_llm_text,
)

logger = logging.getLogger(__name__)


class ChunkMetaImportError(ValueError):
    """检索命中对象缺少渲染所需的字段（chunk_id/metadata）。"""

# ---- 占位文号形态（与 loader/retriever 口径一致：这些字符串非空但不是文号）----
_PLACEHOLDER_DOC_NO = re.compile(r"^无（|^（页面未载")
# ---- 真文号模式（回退判据：metadata 缺 has_doc_no 字段时用）----
_DOC_NO_PATTERN = re.compile(r"〔\d{4}〕\s*\d+\s*号")

# 效力状态取值（与数据仓「所在文件效力状态」及检索元数据一致）
_VALID_CURRENT = frozenset({"有效", "现行有效"})
_VALID_PENDING = frozenset({"待人工核验", "待核验"})
_VALID_INVALID = frozenset({"已失效"})


class CitationError(ValueError):
    """引用渲染失败（查表/组装阶段的结构性问题）。"""


# ---------------------------------------------------------------------------
# 元数据视图
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChunkMeta:
    """单条检索块的渲染侧元数据（字段与检索侧 metadata 命名对齐）。"""

    citation_id: str          # [Cn] 标记本身，如 "[C1]"
    doc_no: str               # registry 文号（可能是占位串）
    clause_no: str            # 条款号（可为空串）
    effective_date: str       # 生效/发布日期（渲染用）
    validity: str             # 效力状态：有效/待人工核验/已失效
    level: str                # 层级：国家/省/市/县…
    title: str                # registry 标题（无真文号时的退化引用源）
    is_current: bool          # 时效过滤布尔（loader 写入；缺失时按 validity 推断）

    # ---- has_doc_no 判据（与 retriever 同款三层：布尔字段 > 正则 > 占位拦截）----
    @property
    def has_doc_no(self) -> bool:
        if self.doc_no and _DOC_NO_PATTERN.search(self.doc_no):
            return not _PLACEHOLDER_DOC_NO.match(self.doc_no.strip())
        return False

    @classmethod
    def from_hit(cls, hit: Any) -> ChunkMeta:
        """从检索侧命中对象构造（鸭子类型：需 ``chunk_id`` 与 ``metadata`` 映射）。"""
        chunk_id = getattr(hit, "chunk_id", None)
        metadata = getattr(hit, "metadata", None)
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ChunkMetaImportError(f"命中对象缺少 chunk_id：{hit!r}")
        if not isinstance(metadata, Mapping):
            raise ChunkMetaImportError(f"命中对象 metadata 不是映射：{hit!r}")
        return cls.from_mapping(metadata, citation_id=chunk_id)

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, Any], *, citation_id: str) -> ChunkMeta:
        """从元数据字典构造；缺失字段按诚实口径补空（渲染时降级）。"""
        doc_no = str(metadata.get("doc_no") or metadata.get("文号") or "")
        validity = str(metadata.get("validity") or metadata.get("效力状态") or "")
        is_current = metadata.get("is_current")
        if isinstance(is_current, bool):
            current = is_current
        else:
            current = validity in _VALID_CURRENT  # 缺失时按效力推断（保守）
        return cls(
            citation_id=citation_id,
            doc_no=doc_no,
            clause_no=str(metadata.get("clause_no") or metadata.get("条款号") or ""),
            effective_date=str(
                metadata.get("effective_date") or metadata.get("生效日期")
                or metadata.get("发布日期") or ""),
            validity=validity,
            level=str(metadata.get("level") or metadata.get("层级") or ""),
            title=str(metadata.get("title") or metadata.get("标题") or ""),
            is_current=current,
        )


def build_meta_table(hits: Iterable[Any]) -> dict[str, ChunkMeta]:
    """从检索命中序列构建标记→元数据表；同标记后到者覆盖并记 warning。"""
    table: dict[str, ChunkMeta] = {}
    for hit in hits:
        if isinstance(hit, Mapping):
            cid = str(hit.get("chunk_id") or hit.get("citation_id") or "")
            if not cid:
                raise ChunkMetaImportError(f"元数据字典缺少 chunk_id/citation_id：{hit!r}")
            # 兼容两种 dict 形状：检索命中（chunk_id+text+嵌套 metadata）与裸 metadata
            inner = hit.get("metadata")
            meta = ChunkMeta.from_mapping(
                inner if isinstance(inner, Mapping) else hit, citation_id=cid)
        else:
            meta = ChunkMeta.from_hit(hit)
        if meta.citation_id in table:
            logger.warning("标记 %s 重复出现，后者覆盖", meta.citation_id)
        table[meta.citation_id] = meta
    return table


# ---------------------------------------------------------------------------
# 引用解析与格式化
# ---------------------------------------------------------------------------

def format_reference(meta: ChunkMeta) -> str:
    """引用行（人类可读）：有真文号用文号，无则退化标题——两者都是 registry 权威字段。"""
    if meta.has_doc_no:
        head = meta.doc_no + (f" {meta.clause_no}" if meta.clause_no else "")
    else:
        head = (
            (f"《{meta.title}》" + (f" {meta.clause_no}" if meta.clause_no else ""))
            if meta.title else "（来源未载文号与标题）")
    bits = [head]
    if meta.level:
        bits.append(meta.level)
    bits.append(meta.validity or "时效不明")
    return "（".join([bits[0], "，".join(bits[1:]) + "）"]) if len(bits) > 1 else bits[0]


def resolve_citation(citation: str, table: Mapping[str, ChunkMeta]) -> ChunkMeta | None:
    """查表解析标记；查不到返回 None（调用方输出诚实降级，不猜不补）。"""
    return table.get(citation)


# ---------------------------------------------------------------------------
# 置信填充（确定性信号 → §4.8 三档）
# ---------------------------------------------------------------------------

def _confidence_for(meta: ChunkMeta) -> str:
    if meta.validity in _VALID_PENDING:
        return CONFIDENCE_SUGGEST
    if meta.validity not in _VALID_CURRENT:
        return CONFIDENCE_SUGGEST  # 时效不明 → 建议确认
    return CONFIDENCE_EXPLICIT


# ---------------------------------------------------------------------------
# 报告组装（GeneratedSections + NegativeHit + 元数据表 → CheckupReport）
# ---------------------------------------------------------------------------

def build_report(
    project_summary: str,
    sections: GeneratedSections,
    negative_hits: tuple[NegativeHit, ...],
    table: Mapping[str, ChunkMeta],
) -> CheckupReport:
    """组装四段式报告：查表注入引用、填充置信、剔除失效、诚实降级。

    红线：LLM 文本重过禁语/文号防线（防绕过 parse_generated 直接构造）；
    ①段已失效条目剔除（检索侧过滤的渲染层双保险），剔空转未覆盖。
    """
    # ——①段——
    policy_items: list[PolicyItem] = []
    dropped_invalid = 0
    uncovered = list(sections.uncovered_sections)
    for item in sections.policy_items:
        check_llm_text(item.summary, f"①段 {item.citation} summary")  # 防线（含文号模式）
        meta = resolve_citation(item.citation, table)
        if meta is None:
            logger.warning("①段标记 %s 不在元数据表中，条目丢弃（依据标注缺失）", item.citation)
            continue
        if not meta.is_current or meta.validity in _VALID_INVALID:
            dropped_invalid += 1
            logger.warning("①段标记 %s 指向已失效条款，已剔除", item.citation)
            continue
        policy_items.append(PolicyItem(
            citation=item.citation,
            summary=check_llm_text(item.summary, f"①段 {item.citation}"),
            confidence=_confidence_for(meta),
        ))
    if dropped_invalid and not policy_items and "①适用政策" not in uncovered:
        uncovered.append("①适用政策")

    # ——③段——（同防线；失效指南条目同样剔除）
    missing_items: list[MissingItem] = []
    for item in sections.missing_items:
        check_llm_text(item.note, f"③段 {item.citation} note")
        check_llm_text(item.name, f"③段 {item.citation} name")
        meta = resolve_citation(item.citation, table)
        if meta is None or not meta.is_current or meta.validity in _VALID_INVALID:
            logger.warning("③段标记 %s 无法解析或已失效，条目丢弃", item.citation)
            continue
        missing_items.append(item)

    # ——④段——
    suggestions = tuple(
        SuggestionItem(text=check_llm_text(s.text, "④段 text")) for s in sections.suggestions
    )

    # project_summary 是用户自述（非 LLM 生成）：只做非空校验，不套 LLM 文本防线——
    # 申报人描述里引用真实文号或提到「合规」是合法输入，防线只约束 LLM 产物（预审 A2）。
    summary_clean = project_summary.strip()
    if not summary_clean:
        raise CitationError("project_summary 不能为空")
    report = CheckupReport(
        project_summary=summary_clean,
        policy_items=tuple(policy_items),
        negative_hits=tuple(negative_hits),
        missing_items=tuple(missing_items),
        suggestions=suggestions,
        uncovered_sections=tuple(dict.fromkeys(uncovered)),  # 去重保序
    )
    # 结构校验不传标记白名单：白名单口径（LLM 标记须来自上下文）由 parse_generated
    # 在解析口把关；渲染层是防御纵深——真遇表外标记走「依据标注缺失」降级，不崩。
    report.validate()
    return report


# ---------------------------------------------------------------------------
# 文本渲染（渲染层唯一出口；免责声明强制附加）
# ---------------------------------------------------------------------------

def _ref_or_missing(citation: str, table: Mapping[str, ChunkMeta]) -> str:
    meta = resolve_citation(citation, table)
    if meta is None:
        return f"〔依据标注缺失：{citation} 未在检索上下文中解析〕"
    return format_reference(meta)


def render_hit_line(hit: NegativeHit, table: Mapping[str, ChunkMeta]) -> str:
    """②段命中行：命中类型（规则/疑似）＋引用行＋命中词对照。"""
    ref = _ref_or_missing(hit.citation, table)
    words = "、".join(f"「{a}」↔「{b}」" for a, b in hit.matched_words) or "（蕴含复核命中）"
    return f"〔{hit.hit_type}〕{ref}｜命中词：{words}"


def render_report(
    report: CheckupReport,
    table: Mapping[str, ChunkMeta],
    degraded_sections: tuple[str, ...] = (),
) -> str:
    """CheckupReport → 四段式文本。免责声明在本函数强制附加（唯一出口）。

    degraded_sections：校验器（gen.validator）重试耗尽后显式降级的段位清单——
    对应段标题附加 DEGRADED_NOTICE 标注（A9「降级显式标注」的渲染落点）。
    """
    lines: list[str] = [f"体检输入：{report.project_summary}", ""]

    lines.append("① 适用政策清单"
                 + ("〔已降级：未通过完整性校验，仅供参考〕"
                    if "①适用政策" in degraded_sections else ""))
    if report.policy_items:
        for i, item in enumerate(report.policy_items, 1):
            lines.append(f"{i}. {_ref_or_missing(item.citation, table)}〔{item.confidence}〕")
            lines.append(f"   {item.summary}")
    else:
        lines.append("（未在演示语料库中识别到相关政策，不代表项目本身有问题）")
    lines.append("")

    lines.append("② 负面清单命中提示（建议对照原文，非判定）")
    if report.negative_hits:
        lines.extend(f"· {render_hit_line(h, table)}" for h in report.negative_hits)
    else:
        lines.append("· 未命中已收录的禁止性条款")
    lines.append("")

    lines.append("③ 材料缺项清单（建议核对，非正式申报清单）"
                 + ("〔已降级：未通过完整性校验，仅供参考〕"
                    if "③材料缺项" in degraded_sections else ""))
    if report.missing_items:
        lines.extend(
            f"· {m.name}——{m.note}（依据：{_ref_or_missing(m.citation, table)}）"
            for m in report.missing_items)
    else:
        lines.append("· 未识别到可列出的材料缺项")
    lines.append("")

    lines.append("④ 修改建议"
                 + ("〔已降级：未通过完整性校验，仅供参考〕"
                    if "④修改建议" in degraded_sections else ""))
    if report.suggestions:
        lines.extend(f"· {s.text}" for s in report.suggestions)
    else:
        lines.append("· 暂无")

    if report.uncovered_sections:
        lines.append(f"诚实声明——以下段未能可靠作答：{'、'.join(report.uncovered_sections)}")

    lines.append("")
    lines.append(f"免责声明：{DISCLAIMER_TEXT}")  # 渲染层强制注入（唯一出口）
    return "\n".join(lines)
