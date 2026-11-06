"""gen/ — 生成、标记式引用渲染、确定性校验器。

模块（手册 §3.3）：
- schema.py：四段式报告与三段 LLM 输出结构＋校验＋JSON 解析（PR1）；
- prompts.py：抽取/负面复核/生成三段提示词，temperature=0（PR1）；
- 引用渲染（feat/gen-citation）与校验器（feat/gen-validator）后续 PR 落地。
"""

from gen.schema import (
    CONFIDENCE_EXPLICIT,
    CONFIDENCE_SUGGEST,
    CONFIDENCE_UNCOVERED,
    DISCLAIMER_TEXT,
    CheckupReport,
    EntailmentVerdict,
    GeneratedSections,
    MissingItem,
    NegativeHit,
    PolicyItem,
    QueryExtraction,
    SchemaError,
    SuggestionItem,
    parse_entailments,
    parse_extraction,
    parse_generated,
)

__all__ = [
    "CONFIDENCE_EXPLICIT",
    "CONFIDENCE_SUGGEST",
    "CONFIDENCE_UNCOVERED",
    "DISCLAIMER_TEXT",
    "CheckupReport",
    "EntailmentVerdict",
    "GeneratedSections",
    "MissingItem",
    "NegativeHit",
    "PolicyItem",
    "QueryExtraction",
    "SchemaError",
    "SuggestionItem",
    "parse_entailments",
    "parse_extraction",
    "parse_generated",
]
