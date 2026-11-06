"""rules/engine.py — 确定性负面清单匹配引擎（自建禁止用途词典词表共现匹配，无 LLM、无分词依赖）。

职责：
- 输入项目描述（申报场景文本），输出命中列表：规则文号/条款号（带层级/效力标注）
  /条款原文/命中词/等价组对照；
- 匹配口径：描述侧术语与条款侧术语（词典词表子串匹配，输出确定性）落在同一概念组
  即命中，可解释（输出两侧命中词与概念组）；
- 匹配范围：现行规则集（剔除已失效）中再剔除「结构标注」组（总括句等结构噪音）；
- 姿态红线：引擎是提示器不是判定器——文案只写「命中提示/建议对照原文」，
  不得出现「违规/不合规/可以申报/合规」类措辞；
- 展示优先级：等价组内现行有效优先，同有效时层级更具体优先（县>市>省>国家）。

红线遵守：本模块与测试均不含真实政策语料原文（语料不入代码仓）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rules.lexicon import FORBIDDEN_USE_CONCEPTS
from rules.schema import (
    GROUP_TYPE_STRUCTURE,
    RuleRecord,
    RuleStore,
    display_priority_key,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleHit:
    """单条规则命中（可解释：两侧命中词 + 共现概念组 + 等价组对照）。"""

    rule: RuleRecord
    matched_terms: tuple[str, ...]              # 条款侧参与共现的术语
    matched_description_terms: tuple[str, ...]  # 描述侧触发术语
    concept: str                                # 共现的概念组（多个用／连接）
    group_peers: tuple[RuleRecord, ...]         # 等价组其他条款（对照展示）


@dataclass(frozen=True)
class MatchResult:
    """一次匹配的结果：命中列表（按展示优先级排序）+ 描述侧识别术语。"""

    hits: tuple[RuleHit, ...]
    description_terms: tuple[str, ...]

    def groups(self) -> frozenset[str]:
        """命中规则所属等价组集合（评测用）。"""
        return frozenset(hit.rule.group_id for hit in self.hits)


class NegRuleMatcher:
    """确定性匹配器：规则库（剔除已失效）× 项目描述 → 命中列表。"""

    def __init__(
        self,
        store: RuleStore,
        concepts: Mapping[str, Sequence[str]] = FORBIDDEN_USE_CONCEPTS,
    ) -> None:
        """构建匹配器：预计算现行规则（剔除已失效与结构标注组）的概念组。"""
        self.store = store
        self._ordered_terms: tuple[str, ...] = tuple(
            t for terms in concepts.values() for t in terms)
        self._term_concept: dict[str, str] = {t: c for c, terms in concepts.items() for t in terms}
        self._rule_concepts: list[tuple[RuleRecord, frozenset[str]]] = [
            (rule, self._concepts_in(rule.clause_text))
            for rule in store.active_records()
            if rule.group_type != GROUP_TYPE_STRUCTURE
        ]
        logger.info(
            "匹配器构建完成：词典 %d 词 / %d 概念组，参与匹配规则 %d 条",
            len(self._ordered_terms), len(concepts), len(self._rule_concepts),
        )

    def _terms_in(self, text: str) -> tuple[str, ...]:
        """文本中出现的词典术语（词表子串匹配，按词典顺序输出，确定性）。"""
        return tuple(term for term in self._ordered_terms if term in text)

    def _concepts_in(self, text: str) -> frozenset[str]:
        """文本中出现的术语所属概念组集合。"""
        return frozenset(self._term_concept[term] for term in self._terms_in(text))

    def match(self, description: str) -> MatchResult:
        """匹配项目描述：与描述术语同概念组的规则即命中，按展示优先级排序。"""
        if not description:
            return MatchResult(hits=(), description_terms=())
        desc_terms = self._terms_in(description)
        desc_concepts = frozenset(self._term_concept[t] for t in desc_terms)
        if not desc_concepts:
            return MatchResult(hits=(), description_terms=())
        hits: list[RuleHit] = []
        for rule, rule_concepts in self._rule_concepts:
            shared = rule_concepts & desc_concepts
            if not shared:
                continue
            rule_terms = tuple(
                t for t in self._ordered_terms
                if self._term_concept[t] in shared and t in rule.clause_text
            )
            desc_hits = tuple(t for t in desc_terms if self._term_concept[t] in shared)
            hits.append(
                RuleHit(
                    rule=rule,
                    matched_terms=rule_terms,
                    matched_description_terms=desc_hits,
                    concept="／".join(sorted(shared)),
                    group_peers=tuple(self.store.expand_equivalents(rule)),
                )
            )
        hits.sort(key=lambda hit: display_priority_key(hit.rule))
        logger.debug("描述术语 %s，命中 %d 条", desc_terms, len(hits))
        return MatchResult(hits=tuple(hits), description_terms=desc_terms)


def render_hits(result: MatchResult) -> list[str]:
    """渲染命中提示文案（只作提示，不判定——姿态红线）。

    每条命中输出：文号+条款号（层级/效力标注）｜命中词（描述侧↔条款侧，概念组）｜
    条款原文｜等价组对照。无命中时返回中性提示。
    """
    if not result.hits:
        return ["未识别到禁止用途词，暂无命中提示。"]
    lines: list[str] = []
    for hit in result.hits:
        rule = hit.rule
        peers = "；".join(
            f"{p.doc_no} {p.clause_no}（{p.level}，{p.effect_status}）" for p in hit.group_peers
        ) or "无"
        lines.append(
            f"命中提示（建议对照原文）｜{rule.doc_no}{rule.clause_no}（{rule.level}，{rule.effect_status}）｜"
            f"命中词：描述侧「{'、'.join(hit.matched_description_terms)}」"
            f"↔ 条款侧「{'、'.join(hit.matched_terms)}」（概念：{hit.concept}）｜"
            f"条款原文：{rule.clause_text}｜"
            f"等价组：{rule.group_id} {rule.group_name}（同组其他层级：{peers}）"
        )
    return lines
