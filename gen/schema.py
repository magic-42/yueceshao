"""gen/schema.py — 四段式体检报告与三段 LLM 输出的结构定义。

职责：
1. CheckupReport：四段式报告最终结构（①适用政策清单/②负面清单命中提示/③材料
   缺项清单/④修改建议），②段由规则命中与蕴含复核结果模板渲染，不经 LLM；
2. LLM 输出结构：QueryExtraction（查询结构化抽取，A3）、EntailmentVerdict
   （负面条款蕴含复核，A10）、GeneratedSections（①③④段单次生成，A8）；
3. 结构级校验（枚举/标记格式/置信纪律/禁语文号防线）＋ parse_* JSON 解析。
   完整的四类确定性校验器与重试降级在 feat/gen-validator（手册 §3.3 PR3）。

红线（README 开发红线，本模块以结构手段落地）：
- LLM 不手打文号：LLM 结构的文本字段出现文号模式即校验失败（文号由 chunk
  元数据在渲染层注入，feat/gen-citation PR2）；
- 姿态禁语：LLM 文本字段禁「合规/可以申报/违规/不合规」类结论性措辞；
- 置信标签由确定性信号驱动（规则命中/重排分/时效状态），LLM 不自评置信度
  （手册 §4.8）——GeneratedSections 里的置信字段必须为 None，系统侧填充；
- 免责声明由渲染层强制注入（DISCLAIMER_TEXT 锚点），LLM 不生成。
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

# ---- 置信分级（手册 §4.8 口径，由确定性信号驱动，LLM 不自评）----
CONFIDENCE_EXPLICIT = "明确适用"
CONFIDENCE_SUGGEST = "建议确认"
CONFIDENCE_UNCOVERED = "未覆盖"
VALID_CONFIDENCE = frozenset({CONFIDENCE_EXPLICIT, CONFIDENCE_SUGGEST, CONFIDENCE_UNCOVERED})

# ---- ②段命中类型（与 rules/engine render_hits 口径一致）----
HIT_RULE = "规则命中"
HIT_SUSPECT = "疑似命中（建议对照原文）"
VALID_HIT_TYPES = frozenset({HIT_RULE, HIT_SUSPECT})

# ---- 蕴含复核结论（A10：规则引擎 ∪ LLM 蕴含复核，取并集）----
ENTAILMENT_YES = "蕴含"
ENTAILMENT_NO = "不蕴含"
ENTAILMENT_UNSURE = "不确定"
VALID_ENTAILMENTS = frozenset({ENTAILMENT_YES, ENTAILMENT_NO, ENTAILMENT_UNSURE})

# ---- 主体类型（A3：只作重排信号，不用于过滤）----
VALID_SUBJECT_TYPES = frozenset({"村集体经济组织", "农民专业合作社", "强村公司", "其他"})

# ---- LLM 文本字段的姿态禁语（结论性措辞）----
FORBIDDEN_PHRASES: tuple[str, ...] = ("合规", "可以申报", "违规", "不合规")

# ---- 文号模式（LLM 输出出现即失败：文号只能由元数据注入，不许可述）----
_DOC_NO_PATTERN = re.compile(r"〔\d{4}〕\s*\d+\s*号")

# ---- 标记式引用格式（A8）：[C1]、[C12] 等 ----
CITATION_PATTERN = re.compile(r"^\[C(\d+)\]$")

# ---- 渲染层强制注入的免责声明（README 产品级原文锚点，feat/gen-citation 使用）----
DISCLAIMER_TEXT = (
    "本报告仅为申报前自查辅助参考，不构成合规判定，"
    "请以政策法规原文及主管部门意见为准。"
)


class SchemaError(ValueError):
    """结构校验失败（带字段定位信息）。"""


# ---------------------------------------------------------------------------
# 校验原语
# ---------------------------------------------------------------------------

def check_llm_text(text: str, where: str) -> str:
    """LLM 文本字段防线：姿态禁语＋文号模式（完整校验器在 feat/gen-validator）。

    渲染层注入的系统文本（免责声明、元数据行）不走本函数——禁语与文号约束
    只针对 LLM 生成内容。
    """
    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            raise SchemaError(f"{where}：出现姿态禁语「{phrase}」（建议口吻，禁结论性措辞）")
    if _DOC_NO_PATTERN.search(text):
        raise SchemaError(f"{where}：出现文号字样（引用只用 [Cn] 标记，文号由系统注入）")
    return text


def check_citation(citation: str, allowed: Collection[str], where: str) -> str:
    """标记式引用校验：格式必须为 [Cn]，且 n 必须在本次提供的上下文标记集合内。"""
    if not CITATION_PATTERN.match(citation or ""):
        raise SchemaError(f"{where}：引用标记格式非法（应为 [C1] 形式），收到 {citation!r}")
    if allowed and citation not in allowed:
        raise SchemaError(f"{where}：引用标记 {citation} 不在本次提供的上下文标记集合内")
    return citation


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{where}：需要非空字符串，收到 {value!r}")
    return value


# ---------------------------------------------------------------------------
# ① 抽取结构（A3/A4）
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QueryExtraction:
    """查询结构化抽取输出：主体类型＋建设内容/用途条目分解＋术语归一建议。

    construction_items / use_items 供检索按条目分解（A4）；subject_type 只作
    重排信号（A3），不用于过滤。
    """

    subject_type: str                    # 主体类型（枚举）
    construction_items: tuple[str, ...]  # 建设内容条目
    use_items: tuple[str, ...]           # 资金用途条目
    terms: tuple[str, ...]               # 术语归一建议（如「建厕所」→「农村人居环境整治」）

    def validate(self) -> QueryExtraction:
        if self.subject_type not in VALID_SUBJECT_TYPES:
            raise SchemaError(
                f"主体类型非法：{self.subject_type!r}"
                f"（枚举：{sorted(VALID_SUBJECT_TYPES)}）")
        if not self.construction_items and not self.use_items:
            raise SchemaError("建设内容与资金用途条目不能同时为空（检索分解无从下手）")
        for label, items in (
            ("建设内容条目", self.construction_items),
            ("资金用途条目", self.use_items),
            ("术语归一建议", self.terms),
        ):
            for i, item in enumerate(items, 1):
                check_llm_text(_require_str(item, f"{label}[{i}]"), f"{label}[{i}]")
        return self


# ---------------------------------------------------------------------------
# ② 蕴含复核结构（A10）
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EntailmentVerdict:
    """单条负面条款的蕴含复核结论（与规则命中取并集，命中类型按 Hierarchy 归并）。"""

    citation: str   # 负面条款的 [Cn] 标记
    verdict: str    # 蕴含 / 不蕴含 / 不确定
    reason: str     # 判断依据（引 chunk 原文措辞；禁文号、禁结论性措辞）

    def validate(self, allowed: Collection[str] = ()) -> EntailmentVerdict:
        check_citation(self.citation, allowed, "EntailmentVerdict.citation")
        if self.verdict not in VALID_ENTAILMENTS:
            raise SchemaError(
                f"蕴含结论非法：{self.verdict!r}（枚举：{sorted(VALID_ENTAILMENTS)}）")
        check_llm_text(
            _require_str(self.reason, "EntailmentVerdict.reason"), "EntailmentVerdict.reason")
        return self


# ---------------------------------------------------------------------------
# ③ 生成结构（①③④段，A8 标记式引用）
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PolicyItem:
    """①段条目：适用政策（LLM 只写适用说明；置信与时效由系统侧确定性信号填充）。"""

    citation: str           # [Cn] 标记
    summary: str            # 适用说明（不含文号/条款号/生效日期——渲染层注入）
    confidence: str | None  # 置信标签：LLM 输出必须为 None（手册 §4.8：LLM 不自评置信度）

    def validate(self, allowed: Collection[str] = ()) -> PolicyItem:
        check_citation(self.citation, allowed, "PolicyItem.citation")
        check_llm_text(_require_str(self.summary, "PolicyItem.summary"), "PolicyItem.summary")
        if self.confidence is not None and self.confidence not in VALID_CONFIDENCE:
            raise SchemaError(
                f"置信标签非法：{self.confidence!r}（且 LLM 输出必须为 None，系统侧填充）")
        return self


@dataclass(frozen=True, slots=True)
class MissingItem:
    """③段条目：材料缺项（按①命中的现行申报指南列出，提示性质）。"""

    citation: str   # [Cn] 标记（指向申报指南条款）
    name: str       # 材料名称
    note: str       # 依据说明（建议核对口径，非正式申报清单）

    def validate(self, allowed: Collection[str] = ()) -> MissingItem:
        check_citation(self.citation, allowed, "MissingItem.citation")
        check_llm_text(_require_str(self.name, "MissingItem.name"), "MissingItem.name")
        check_llm_text(_require_str(self.note, "MissingItem.note"), "MissingItem.note")
        return self


@dataclass(frozen=True, slots=True)
class SuggestionItem:
    """④段条目：申报表述层面的修改建议（建议口吻，非判定）。"""

    text: str

    def validate(self) -> SuggestionItem:
        check_llm_text(_require_str(self.text, "SuggestionItem.text"), "SuggestionItem.text")
        return self


@dataclass(frozen=True, slots=True)
class GeneratedSections:
    """LLM 单次生成的 ①③④ 段（README 架构：②段由规则命中模板渲染，不经 LLM）。

    uncovered_sections：无法从给定上下文可靠回答的段（"①适用政策"/"③材料缺项"/
    "④修改建议"），必须如实声明——检索质量门不过时走诚实输出（A11）。
    """

    policy_items: tuple[PolicyItem, ...] = field(default=())
    missing_items: tuple[MissingItem, ...] = field(default=())
    suggestions: tuple[SuggestionItem, ...] = field(default=())
    uncovered_sections: tuple[str, ...] = field(default=())

    VALID_SECTION_NAMES = frozenset({"①适用政策", "③材料缺项", "④修改建议"})

    def validate(self, allowed: Collection[str] = ()) -> GeneratedSections:
        for _i, item in enumerate(self.policy_items, 1):
            item.validate(allowed)
        for _i, item in enumerate(self.missing_items, 1):
            item.validate(allowed)
        for _i, item in enumerate(self.suggestions, 1):
            item.validate()
        for name in self.uncovered_sections:
            if name not in self.VALID_SECTION_NAMES:
                raise SchemaError(
                    f"未覆盖段声明非法：{name!r}（枚举：{sorted(self.VALID_SECTION_NAMES)}）")
        # 未覆盖声明与条目互斥：声明了未覆盖的段不得同时给出条目
        if "①适用政策" in self.uncovered_sections and self.policy_items:
            raise SchemaError("①段已声明未覆盖，不得同时给出适用政策条目")
        if "③材料缺项" in self.uncovered_sections and self.missing_items:
            raise SchemaError("③段已声明未覆盖，不得同时给出材料缺项条目")
        if not self.policy_items and "①适用政策" not in self.uncovered_sections:
            raise SchemaError("①段既无条目也未声明未覆盖（A11：不知道要说不知道）")
        if not self.missing_items and "③材料缺项" not in self.uncovered_sections:
            raise SchemaError("③段既无条目也未声明未覆盖（A11：不知道要说不知道）")
        return self


# ---------------------------------------------------------------------------
# ④ 四段式报告最终结构（渲染层数据）
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NegativeHit:
    """②段条目：负面清单命中（规则引擎/蕴含复核并集，模板渲染，不经 LLM）。"""

    citation: str              # [Cn] 标记
    hit_type: str              # 规则命中 / 疑似命中（建议对照原文）
    matched_words: tuple[tuple[str, str], ...]  # (描述侧词, 条款侧词) 对

    def validate(self, allowed: Collection[str] = ()) -> NegativeHit:
        check_citation(self.citation, allowed, "NegativeHit.citation")
        if self.hit_type not in VALID_HIT_TYPES:
            raise SchemaError(f"命中类型非法：{self.hit_type!r}（枚举：{sorted(VALID_HIT_TYPES)}）")
        return self


@dataclass(frozen=True, slots=True)
class CheckupReport:
    """四段式体检报告（渲染层组装后的最终结构；免责声明由渲染层强制附加）。"""

    project_summary: str                  # 项目描述摘要（抽取产物）
    policy_items: tuple[PolicyItem, ...]  # ①段（置信已由系统侧填充）
    negative_hits: tuple[NegativeHit, ...]  # ②段（规则/蕴含并集）
    missing_items: tuple[MissingItem, ...]  # ③段
    suggestions: tuple[SuggestionItem, ...]  # ④段
    uncovered_sections: tuple[str, ...] = field(default=())

    def validate(self, allowed: Collection[str] = ()) -> CheckupReport:
        _require_str(self.project_summary, "CheckupReport.project_summary")
        for item in self.policy_items:
            item.validate(allowed)
            if item.confidence not in VALID_CONFIDENCE:
                raise SchemaError(
                    f"最终报告的①段条目必须带系统侧置信标签，"
                    f"{item.citation} 收到 {item.confidence!r}")
        for item in self.negative_hits:
            item.validate(allowed)
        for item in self.missing_items:
            item.validate(allowed)
        for item in self.suggestions:
            item.validate()
        return self


# ---------------------------------------------------------------------------
# JSON 解析（LLM 返回 → 结构）
# ---------------------------------------------------------------------------

def _strip_json_fences(raw: str) -> str:
    """剥掉 markdown 代码围栏与前后杂讯（qwen 常见输出习惯）。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_object(raw: str, where: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{where}：JSON 解析失败——{exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{where}：顶层必须是 JSON 对象，收到 {type(data).__name__}")
    return data


def _as_str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaError(f"{where}：需要字符串数组，收到 {type(value).__name__}")
    return tuple(_require_str(v, f"{where}[{i}]") for i, v in enumerate(value, 1))


def parse_extraction(raw: str) -> QueryExtraction:
    """解析抽取段 LLM 输出（JSON → QueryExtraction，含结构校验）。"""
    data = _parse_json_object(raw, "parse_extraction")
    try:
        return QueryExtraction(
            subject_type=str(data.get("subject_type") or ""),
            construction_items=_as_str_tuple(
                data.get("construction_items") or [], "construction_items"),
            use_items=_as_str_tuple(data.get("use_items") or [], "use_items"),
            terms=_as_str_tuple(data.get("terms") or [], "terms"),
        ).validate()
    except SchemaError as exc:
        raise SchemaError(f"抽取输出结构不符：{exc}") from exc


def parse_entailments(raw: str, allowed: Collection[str] = ()) -> tuple[EntailmentVerdict, ...]:
    """解析负面复核段 LLM 输出（JSON → EntailmentVerdict 元组，逐条校验）。"""
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"parse_entailments：JSON 解析失败——{exc}") from exc
    if not isinstance(data, list):
        raise SchemaError(f"parse_entailments：顶层需要 JSON 数组，收到 {type(data).__name__}")
    verdicts = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise SchemaError(f"parse_entailments[{i}]：需要 JSON 对象")
        verdicts.append(EntailmentVerdict(
            citation=str(item.get("citation") or ""),
            verdict=str(item.get("verdict") or ""),
            reason=str(item.get("reason") or ""),
        ).validate(allowed))
    return tuple(verdicts)


def _reject_llm_confidence(value: Any, index: int) -> None:
    """置信纪律闸（手册 §4.8）：LLM 输出里的置信字段必须缺省或为 null。

    置信标签由确定性信号（规则命中/重排分/时效状态）在系统侧填充——LLM 自评
    置信度在解析口直接拦下，不进结构。
    """
    if value is not None:
        raise SchemaError(
            f"生成输出 policy_items[{index}]：置信字段必须缺省/null（LLM 不自评置信度，"
            f"由系统侧确定性信号填充），收到 {value!r}"
        )


def parse_generated(raw: str, allowed: Collection[str] = ()) -> GeneratedSections:
    """解析生成段 LLM 输出（JSON → GeneratedSections，含置信纪律与禁语校验）。"""
    data = _parse_json_object(raw, "parse_generated")
    try:
        raw_policy = [it for it in (data.get("policy_items") or []) if isinstance(it, dict)]
        policy_items = []
        for index, it in enumerate(raw_policy, 1):
            _reject_llm_confidence(it.get("confidence"), index)
            policy_items.append(PolicyItem(
                citation=str(it.get("citation") or ""),
                summary=str(it.get("summary") or ""),
                confidence=None,
            ).validate(allowed))
        missing_items = tuple(
            MissingItem(
                citation=str(it.get("citation") or ""),
                name=str(it.get("name") or ""),
                note=str(it.get("note") or ""),
            ).validate(allowed)
            for it in (data.get("missing_items") or [])
            if isinstance(it, dict)
        )
        suggestions = tuple(
            SuggestionItem(text=str(it.get("text") or "")).validate()
            for it in (data.get("suggestions") or [])
            if isinstance(it, dict)
        )
        uncovered = _as_str_tuple(data.get("uncovered_sections") or [], "uncovered_sections")
        return GeneratedSections(
            tuple(policy_items), missing_items, suggestions, uncovered
        ).validate(allowed)
    except SchemaError as exc:
        raise SchemaError(f"生成输出结构不符：{exc}") from exc
