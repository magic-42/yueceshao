"""gen/validator.py — 确定性校验器＋重试降级（A9）：LLM 幻觉的最后防线。

职责（手册 §3.3 PR3；四个「先红后绿」关键路径之一）：
1. 四类确定性检查（全部不依赖 LLM 判断，逐条给出 ValidationIssue）：
   - 标记存在性：LLM 引用的 [Cn] 必须在本次上下文标记集合内（不信任上游，
     schema 层已查过仍再查一遍——兜底防线定义如此）；
   - 包含性：LLM 文本中 ≥8 字的「」成句引语必须逐字出现在所引 chunk 原文中
     （防编造引语／引错条款；短于 8 字视为普通强调不核验，防误报）；
   - 姿态禁语：FORBIDDEN_PHRASES 独立重扫（合规/可以申报/违规/不合规）；
   - 免责声明：渲染成品必须含 DISCLAIMER_TEXT（渲染层唯一出口的复核）。
2. validate_all：四类汇总 → ValidationReport（passed＋逐条 issue，issue 带段位）。
   标记存在性与姿态禁语在正常管线中会被 parse_generated 解析口前置拦截
   （schema 快速失败），独立保留属纵深防御——覆盖绕过解析直接构造的场景。
3. run_with_retry（A9）：generate 回调首次不带反馈、失败重试携带格式化失败原因；
   首次＋重试 ≤2（共至多 3 次）仍失败 → apply_degradation 显式降级：失败段条目
   清空＋声明未覆盖（①③）或清空（④），可安全进渲染——报告照给，但绝不含
   未过校验的内容。JSON 解析失败同样计入重试（校验器不信任任何环节）。

分层（承接 #33 的结论）：本模块是防线终点，不与渲染层耦合——render_report 的
降级标注（DEGRADED_NOTICE）由调用方把 failed_sections 传入渲染完成。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from gen.schema import (
    DISCLAIMER_TEXT,
    FORBIDDEN_PHRASES,
    GeneratedSections,
    SchemaError,
    parse_generated,
)

# ---- 常量 ----
QUOTE_PATTERN = re.compile(r"「([^」]+)」")
QUOTE_MIN_LEN = 8          # ≥8 字的「」片段才视为条款引语（防普通强调误报）
MAX_RETRIES = 2            # A9：重试 ≤2（首次之外）
SECTION_FIRST = "①适用政策"
SECTION_MISSING = "③材料缺项"
SECTION_SUGGEST = "④修改建议"
DEGRADED_NOTICE = "本段未通过完整性校验，已降级处理，仅供参考；请人工对照政策法规原文。"


# ---------------------------------------------------------------------------
# Issue 与报告
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """单条校验失败（check＝四类之一；section＝所属段位）。"""

    check: str
    section: str
    message: str

    def format(self) -> str:
        return f"[{self.check}｜{self.section}] {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """validate_all 的汇总结果。"""

    passed: bool
    issues: tuple[ValidationIssue, ...] = field(default=())

    def feedback(self) -> str:
        """把失败原因格式化为重试反馈（喂回给 LLM 的修正指令）。"""
        return "上一版输出未通过完整性校验，请逐条修正后重新输出完整 JSON：\n" + "\n".join(
            f"- {i.format()}" for i in self.issues
        )


@dataclass(frozen=True, slots=True)
class RetryOutcome:
    """run_with_retry 的结果：成功给 sections；降级给清理后的 sections＋失败段清单。"""

    delivered: GeneratedSections        # 最终交付（成功＝通过版；降级＝清理版）
    degraded: bool                      # True＝重试耗尽后降级
    attempts: int                       # 实际生成次数（1~3）
    issues: tuple[ValidationIssue, ...] # 全部历史 issue（含成功前各轮）
    failed_sections: tuple[str, ...] = field(default=())


# ---------------------------------------------------------------------------
# 四类检查
# ---------------------------------------------------------------------------

def _iter_llm_fields(sections: GeneratedSections):
    """产出 (段位, 字段名, 文本) 三元组——四类检查共用遍历。"""
    for item in sections.policy_items:
        yield SECTION_FIRST, "summary", item.summary
    for item in sections.missing_items:
        yield SECTION_MISSING, "name", item.name
        yield SECTION_MISSING, "note", item.note
    for item in sections.suggestions:
        yield SECTION_SUGGEST, "text", item.text


def check_citation_existence(
    sections: GeneratedSections, allowed: Collection[str]
) -> tuple[ValidationIssue, ...]:
    """检查一：所有 [Cn] 引用必须在上下文标记集合内（格式＋存在双查）。"""
    issues: list[ValidationIssue] = []
    for item in sections.policy_items:
        if item.citation not in allowed:
            issues.append(ValidationIssue("标记存在性", SECTION_FIRST,
                                          f"引用 {item.citation} 不在本次上下文标记集合内"))
    for item in sections.missing_items:
        if item.citation not in allowed:
            issues.append(ValidationIssue("标记存在性", SECTION_MISSING,
                                          f"引用 {item.citation} 不在本次上下文标记集合内"))
    return tuple(issues)


def check_containment(
    sections: GeneratedSections, chunk_texts: Mapping[str, str]
) -> tuple[ValidationIssue, ...]:
    """检查二：≥8 字「」成句引语必须逐字出现在所引 chunk 原文中。"""
    issues: list[ValidationIssue] = []
    pairs: list[tuple[str, str, str]] = []
    for item in sections.policy_items:
        pairs.append((SECTION_FIRST, item.citation, item.summary))
    for item in sections.missing_items:
        pairs.append((SECTION_MISSING, item.citation, item.note))
    for section, citation, text in pairs:
        quotes = [q for q in QUOTE_PATTERN.findall(text) if len(q) >= QUOTE_MIN_LEN]
        if not quotes:
            continue
        source = chunk_texts.get(citation)
        if source is None:
            issues.append(ValidationIssue("包含性", section,
                                          f"引语所引 {citation} 无原文可核验（宁严勿漏）"))
            continue
        for q in quotes:
            if q not in source:
                issues.append(ValidationIssue(
                    "包含性", section,
                    f"引语「{q[:20]}…」未逐字出现在 {citation} 原文中（疑似编造或改写）"))
    return tuple(issues)


def check_posture(sections: GeneratedSections) -> tuple[ValidationIssue, ...]:
    """检查三：姿态禁语独立重扫（不信任 schema 层已查过）。"""
    issues: list[ValidationIssue] = []
    for section, field_name, text in _iter_llm_fields(sections):
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                issues.append(ValidationIssue("姿态禁语", section,
                                              f"{field_name} 出现结论性措辞「{phrase}」"))
                break  # 同字段一条足够
    return tuple(issues)


def check_disclaimer(rendered_text: str) -> tuple[ValidationIssue, ...]:
    """检查四：渲染成品必须携带免责声明（渲染层唯一出口的复核）。"""
    if DISCLAIMER_TEXT in rendered_text:
        return ()
    return (ValidationIssue("免责声明", "渲染成品",
                            "成品文本缺少免责声明（渲染层强制项）"),)


def validate_all(
    sections: GeneratedSections,
    allowed: Collection[str],
    chunk_texts: Mapping[str, str],
) -> ValidationReport:
    """四类检查汇总（免责声明项针对生成产物恒不适用——它在渲染成品上检查）。"""
    issues = (
        check_citation_existence(sections, allowed)
        + check_containment(sections, chunk_texts)
        + check_posture(sections)
    )
    return ValidationReport(passed=not issues, issues=issues)


# ---------------------------------------------------------------------------
# 降级（A9：报告照给，但绝不含未过校验的内容）
# ---------------------------------------------------------------------------

def apply_degradation(
    sections: GeneratedSections, failed_sections: Collection[str]
) -> GeneratedSections:
    """把失败段降级：条目清空＋声明未覆盖（①③）；④清空（无未覆盖枚举值）。"""
    policy = () if SECTION_FIRST in failed_sections else sections.policy_items
    missing = () if SECTION_MISSING in failed_sections else sections.missing_items
    suggestions = () if SECTION_SUGGEST in failed_sections else sections.suggestions
    uncovered = list(sections.uncovered_sections)
    if SECTION_FIRST in failed_sections and SECTION_FIRST not in uncovered:
        uncovered.append(SECTION_FIRST)
    if SECTION_MISSING in failed_sections and SECTION_MISSING not in uncovered:
        uncovered.append(SECTION_MISSING)
    # 完备性兜底：清空后的段若未声明未覆盖，补声明——保证降级输出永远合法
    # （即使输入本身残缺，如直接构造的 sections），降级永不产生非法结构。
    if not policy and SECTION_FIRST not in uncovered:
        uncovered.append(SECTION_FIRST)
    if not missing and SECTION_MISSING not in uncovered:
        uncovered.append(SECTION_MISSING)
    degraded = GeneratedSections(
        policy_items=policy,
        missing_items=missing,
        suggestions=suggestions,
        uncovered_sections=tuple(uncovered),
    )
    degraded.validate()
    return degraded


def _failed_sections_of(issues: Collection[ValidationIssue]) -> tuple[str, ...]:
    """从 issue 集合提取失败段（去重保序）。"""
    seen: dict[str, None] = {}
    for issue in issues:
        seen.setdefault(issue.section)
    return tuple(seen)


# ---------------------------------------------------------------------------
# 重试主循环（A9）
# ---------------------------------------------------------------------------

def run_with_retry(
    generate_fn: Callable[[str | None], str],
    allowed: Collection[str],
    chunk_texts: Mapping[str, str],
    parse_fn: Callable[[str, Collection[str]], GeneratedSections] = parse_generated,
    max_retries: int = MAX_RETRIES,
) -> RetryOutcome:
    """生成→校验→（带反馈）重试 → 耗尽降级。

    generate_fn(feedback)：首次 feedback=None，重试时携带格式化失败原因；
    parse_fn 可注入替换（测试假回调／未来接真实 LLM 的适配层）。
    """
    issues_all: list[ValidationIssue] = []
    last_round_issues: tuple[ValidationIssue, ...] = ()
    last_sections: GeneratedSections | None = None
    last_feedback: str | None = None
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        raw = generate_fn(last_feedback if attempt else None)
        try:
            sections = parse_fn(raw, allowed)
        except SchemaError as exc:
            issue = ValidationIssue("结构解析", "整体", f"输出不符合 JSON 结构约定：{exc}")
            issues_all.append(issue)
            last_round_issues = (issue,)
            last_feedback = issue.format()
            continue
        last_sections = sections
        report = validate_all(sections, allowed, chunk_texts)
        issues_all.extend(report.issues)
        if report.passed:
            return RetryOutcome(delivered=sections, degraded=False,
                                attempts=attempts, issues=tuple(issues_all))
        last_round_issues = report.issues
        last_feedback = report.feedback()

    # 重试耗尽 → 显式降级。降级段只取**最后一轮**的失败（预审 A1 修复：
    # 用累计历史会把跨轮已修复的段一并误清——某段第 2 轮修好、第 3 轮败在
    # 另一段时，它不应被牵连清空）；issues_all 全量保留供审计。
    failed = _failed_sections_of(last_round_issues)
    delivered = apply_degradation(last_sections, failed) if last_sections else GeneratedSections(
        policy_items=(), missing_items=(), suggestions=(),
        uncovered_sections=(SECTION_FIRST, SECTION_MISSING),
    )
    return RetryOutcome(delivered=delivered, degraded=True,
                        attempts=attempts, issues=tuple(issues_all),
                        failed_sections=failed)
