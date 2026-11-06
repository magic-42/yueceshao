"""app/graph.py — LangGraph 编排（A12 定稿 + 方向4：真 LLM 蕴含复核 + 图级重试边）。

编排拓扑（与 README 架构图一致）：

    parse → [retrieve ‖ rule_scan]（并行）→ entail → generate → validate ─┐
                                              ┌────────────────────────────┘
                                              └─（校验失败且 attempts≤2 回 generate）
                                                 （通过 / 耗尽 → render → END）

- 节点＝已合并模块的真实调用（gen.schema 解析防线／gen.citation 查表与置信／
  gen.validator 四类校验＋降级＋render 免责声明——**无一是 mock**）；
  外部依赖（检索命中／LLM 输出／规则命中）经 :class:`OrchestrationComponents`
  回调注入——骨架阶段为夹具，真实适配器实现同一组回调即完成替换；
- 方向4（feat/langgraph-entail-retry）两处升级：
  1. entail 节点从拓扑占位改为真节点——调 ``entail_review`` 得 LLM 蕴含 JSON，
     经 ``parse_entailments`` 解析后把「蕴含」项转成 :class:`NegativeHit`（疑似命中），
     与规则命中取并集回写 ``negative_hits``（②段 = 规则 ∪ LLM，README 架构）；
  2. A9 重试从 generate 节点内部抽到**图级条件边**：generate 只做单次生成，
     validate 做单次解析＋四类校验，``route_after_validate`` 按「是否通过／
     是否耗尽」路由回 generate（带 feedback）或进 render——校验/降级纯函数仍复用
     gen.validator（validate_all / apply_degradation / RetryOutcome），只把循环
     提到图级，便于 LangGraph 追踪每次尝试；
- entail 调真实 LLM 失败 → ②段退化为仅规则命中 + 渲染层显式标注（决策4），不阻断管线。

Mock 夹具全部〔2099〕人造文号＋自造条款原文（红线⑤⑥）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from gen.citation import build_meta_table, build_report, render_report
from gen.schema import (
    ENTAILMENT_YES,
    HIT_SUSPECT,
    CheckupReport,
    GeneratedSections,
    NegativeHit,
    QueryExtraction,
    SchemaError,
    parse_entailments,
    parse_extraction,
    parse_generated,
)
from gen.validator import (
    MAX_RETRIES,
    SECTION_FIRST,
    SECTION_MISSING,
    RetryOutcome,
    ValidationIssue,
    ValidationReport,
    _failed_sections_of,
    apply_degradation,
    check_disclaimer,
    validate_all,
)

logger = logging.getLogger(__name__)

# entail 复核失败时在②段渲染的显式标注（决策4：宁可少报不可错报，不阻断管线）
ENTAIL_FAILURE_NOTE = "②段：LLM 蕴含复核未完成，仅显示规则引擎命中结果。"


class CheckupState(TypedDict, total=False):
    """图状态：节点间传递的全部产物（内存编译不序列化，允许承载 dataclass）。"""

    description: str
    extraction: QueryExtraction
    extraction_json: str
    hits: tuple[dict, ...]
    meta_table: dict[str, Any]
    chunk_texts: dict[str, str]
    negative_hits: tuple
    outcome: RetryOutcome
    report: CheckupReport
    rendered_text: str
    degraded: bool
    attempts: int
    failed_sections: tuple[str, ...]
    # 方向4：图级重试边与 entail 降级
    raw_generated: str
    gen_feedback: str | None
    gen_attempts: int
    issues_accum: tuple
    entail_failed: bool


@dataclass(frozen=True)
class OrchestrationComponents:
    """编排的外部依赖回调集（骨架＝Mock 夹具；真实适配器逐一替换）。

    签名约定（与 app.pipeline.MockCheckupPipeline 的同名私有方法一致）：
    - ``extract(description) -> LLM JSON 字符串``
    - ``retrieve(description, extraction) -> 命中 dict 元组（chunk_id/text/metadata）``
    - ``rule_scan(description) -> NegativeHit 元组（②段，确定性模板渲染）``
    - ``entail_review(description, hits, feedback) -> 蕴含复核 LLM JSON 字符串``
    - ``generate(description, extraction_json, context_blocks, feedback) -> 生成 LLM JSON``
    """

    extract: Callable[[str], str]
    retrieve: Callable[[str, Any], tuple[dict, ...]]
    rule_scan: Callable[[str], tuple]
    entail_review: Callable[[str, Any, str | None], str]
    generate: Callable[[str, str, str, str | None], str]


def _merge_negative_hits(
    rule_hits: tuple, llm_hits: tuple[NegativeHit, ...]
) -> tuple[NegativeHit, ...]:
    """②段并集：规则命中优先，LLM「蕴含」命中按 citation 去重补充。"""
    merged: dict[str, NegativeHit] = {h.citation: h for h in rule_hits}
    for hit in llm_hits:
        merged.setdefault(hit.citation, hit)
    return tuple(merged.values())


def _retry_feedback(issues: tuple[ValidationIssue, ...]) -> str:
    """把本轮失败原因格式化为重试反馈（SchemaError 单条 / 校验失败汇总）。"""
    if len(issues) == 1 and issues[0].check == "结构解析":
        return issues[0].format()
    return ValidationReport(passed=False, issues=issues).feedback()


def build_checkup_graph(components: OrchestrationComponents) -> Any:
    """编译体检编排图（返回 CompiledGraph；``invoke({"description": ...})`` 触发）。"""

    def parse_node(state: CheckupState) -> dict:
        extraction_json = components.extract(state["description"])
        return {"extraction": parse_extraction(extraction_json), "extraction_json": extraction_json}

    def retrieve_node(state: CheckupState) -> dict:
        hits = components.retrieve(state["description"], state["extraction"])
        table = build_meta_table(hits)
        return {
            "hits": hits,
            "meta_table": table,
            "chunk_texts": {h["chunk_id"]: h["text"] for h in hits},
        }

    def rule_scan_node(state: CheckupState) -> dict:
        return {"negative_hits": components.rule_scan(state["description"])}

    def entail_node(state: CheckupState) -> dict:
        """真实 LLM 蕴含复核：解析「蕴含」项并并进②段（规则 ∪ LLM）。

        解析失败（SchemaError，如姿态禁语/引用越界/结构错）→ 带反馈重试 ≤2（A9 同款语义），
        仍失败或非 SchemaError（网络/key 异常）→ 降级为仅规则命中，不阻断管线。
        """
        allowed = set(state["meta_table"].keys())
        feedback: str | None = None
        verdicts: tuple | None = None
        for _attempt in range(MAX_RETRIES + 1):
            try:
                raw = components.entail_review(state["description"], state["hits"], feedback)
                verdicts = parse_entailments(raw, allowed=allowed)
                break
            except SchemaError as exc:
                feedback = (
                    "上一版蕴含复核输出未通过校验，请按以下反馈修正后重新输出完整 JSON 数组：\n"
                    f"- {exc}"
                )
            except Exception:  # noqa: BLE001 - 网络/key 异常不重试，退化为仅规则命中
                logger.warning("LLM 蕴含复核失败，②段退化为仅规则命中", exc_info=True)
                return {"entail_failed": True}
        if verdicts is None:
            logger.warning("LLM 蕴含复核重试耗尽，②段退化为仅规则命中")
            return {"entail_failed": True}
        llm_hits = tuple(
            NegativeHit(citation=v.citation, hit_type=HIT_SUSPECT, matched_words=())
            for v in verdicts if v.verdict == ENTAILMENT_YES
        )
        return {"negative_hits": _merge_negative_hits(state["negative_hits"], llm_hits)}

    def generate_node(state: CheckupState) -> dict:
        """单次生成（重试由图级条件边驱动，feedback 从状态携带）。"""
        raw = components.generate(
            state["description"], state["extraction_json"], "", state.get("gen_feedback")
        )
        return {"raw_generated": raw}

    def validate_node(state: CheckupState) -> dict:
        """单次解析＋四类校验；通过/耗尽给 outcome，否则给 feedback 触发重试。"""
        raw = state["raw_generated"]
        attempts = state.get("gen_attempts", 0) + 1
        allowed = set(state["meta_table"].keys())
        chunk_texts = state["chunk_texts"]
        issues_accum = list(state.get("issues_accum", ()))
        try:
            sections = parse_generated(raw, allowed)
        except SchemaError as exc:
            issue = ValidationIssue("结构解析", "整体", f"输出不符合 JSON 结构约定：{exc}")
            issues_accum.append(issue)
            round_issues = (issue,)
            sections = None
        else:
            report = validate_all(sections, allowed, chunk_texts)
            issues_accum.extend(report.issues)
            if report.passed:
                return {
                    "outcome": RetryOutcome(delivered=sections, degraded=False,
                                            attempts=attempts, issues=tuple(issues_accum)),
                    "gen_attempts": attempts,
                }
            round_issues = report.issues

        # 校验失败：未耗尽 → 带反馈重试；耗尽 → 显式降级
        if attempts <= MAX_RETRIES:
            return {
                "gen_feedback": _retry_feedback(round_issues),
                "gen_attempts": attempts,
                "issues_accum": tuple(issues_accum),
            }
        failed = _failed_sections_of(round_issues)
        if sections is None:
            delivered = GeneratedSections(
                policy_items=(), missing_items=(), suggestions=(),
                uncovered_sections=(SECTION_FIRST, SECTION_MISSING),
            )
        else:
            delivered = apply_degradation(sections, failed)
        return {
            "outcome": RetryOutcome(delivered=delivered, degraded=True,
                                    attempts=attempts, issues=tuple(issues_accum),
                                    failed_sections=failed),
            "gen_attempts": attempts,
            "issues_accum": tuple(issues_accum),
        }

    def route_after_validate(state: CheckupState) -> str:
        """条件边路由：有 outcome（通过/耗尽）进 render，否则回 generate 重试。"""
        return "render" if "outcome" in state else "generate"

    def render_node(state: CheckupState) -> dict:
        outcome = state["outcome"]
        report = build_report(
            state["description"],
            outcome.delivered,
            negative_hits=state["negative_hits"],
            table=state["meta_table"],
        )
        degraded_sections = outcome.failed_sections if outcome.degraded else ()
        text = render_report(report, state["meta_table"], degraded_sections=degraded_sections)
        if state.get("entail_failed"):
            text = _insert_entail_note(text)
        if check_disclaimer(text):  # 免责声明最后防线复核（渲染层保证，此处兜底）
            logger.error("渲染成品缺少免责声明（不应发生）")
        return {
            "report": report,
            "rendered_text": text,
            "degraded": outcome.degraded,
            "attempts": outcome.attempts,
            "failed_sections": outcome.failed_sections,
        }

    builder = StateGraph(CheckupState)
    builder.add_node("parse", parse_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rule_scan", rule_scan_node)
    builder.add_node("entail", entail_node)
    builder.add_node("generate", generate_node)
    builder.add_node("validate", validate_node)
    builder.add_node("render", render_node)

    builder.add_edge(START, "parse")
    builder.add_edge("parse", "retrieve")     # 并行分支一：检索
    builder.add_edge("parse", "rule_scan")    # 并行分支二：规则（README 架构：两路并行）
    builder.add_edge("retrieve", "entail")    # 汇合后负面复核
    builder.add_edge("rule_scan", "entail")
    builder.add_edge("entail", "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges(
        "validate", route_after_validate,
        {"generate": "generate", "render": "render"},
    )
    builder.add_edge("render", END)
    return builder.compile()


def _insert_entail_note(text: str) -> str:
    """把 entail 失败标注插到免责声明之前（免责声明保持末行，姿态红线）。"""
    marker = "免责声明："
    if marker in text:
        index = text.rindex(marker)
        return text[:index] + ENTAIL_FAILURE_NOTE + "\n\n" + text[index:]
    return text + "\n\n" + ENTAIL_FAILURE_NOTE
