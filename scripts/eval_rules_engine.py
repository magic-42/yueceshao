"""本地评测脚本：规则引擎 × 案例标注表（已知违规召回/误报双口径，数据不入库）。

用法（代码仓根目录）：python scripts/eval_rules_engine.py
前置：根目录 config.json 已按 config.example.json 配置数据仓、规则源与案例表路径。

口径（2026-11-02 经 AI 预审修订）：
1. 可映射案例＝历史适用条款按（文号,条款号）成对匹配规则库记录（排除「结构标注」组）且
   匹配非空的案例；
2. 期望行为按标注语义拆分：
   - 标注「不覆盖本情形」→ 期望引擎不提示（引擎触发即误报）；
   - 覆盖清单项（如 106号第十八条第（N）项）→ 期望命中映射到的等价组（条级映射 G01-G11）；
3. 判定：覆盖案例 命中组∩期望组≠∅ 即召回命中；不覆盖案例 零命中即判定正确；
4. 输出：已知违规召回率、误报率、含禁止用途词子集参考值；抽样（可映射案例按编号升序＋
   固定随机种子 seed=20261102 抽 20 条）与全量参考，可复现。
"""

from __future__ import annotations

import csv
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rules.engine import NegRuleMatcher  # noqa: E402
from rules.loader import load_config, load_store_from_config  # noqa: E402
from rules.schema import GROUP_TYPE_STRUCTURE, RuleStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval")

_DOC_PAT = re.compile(r"([\u4e00-\u9fff]{2,10}〔\d{4}〕\d+号)")
_CLAUSE_PAT = re.compile(r"第([一二三四五六七八九十百零]+)条")
_SAMPLE_SIZE = 20
_SEED = 20261102


def _doc_clause_pairs(history: str) -> set[tuple[str, str]]:
    """按文本顺序把「第X条」归到最近的前一个文号，返回（文号,条款号）成对集合。"""
    docs = [(m.start(), m.group(1)) for m in _DOC_PAT.finditer(history)]
    pairs: set[tuple[str, str]] = set()
    for index, (start, doc) in enumerate(docs):
        end = docs[index + 1][0] if index + 1 < len(docs) else len(history)
        for clause in _CLAUSE_PAT.finditer(history, start, end):
            pairs.add((doc, clause.group(1)))
    return pairs


def _mapped_groups(history: str, store: RuleStore) -> frozenset[str]:
    """历史适用条款（文号,条款号）成对匹配规则库记录（排除结构标注组）取组号集合。"""
    pairs = _doc_clause_pairs(history)
    return frozenset(
        r.group_id
        for r in store.records
        if (r.doc_no, r.clause_no) in pairs and r.group_type != GROUP_TYPE_STRUCTURE
    )


def is_mappable(history: str, store: RuleStore) -> bool:
    """历史适用条款能否映射到等价组（映射非空即可参与评测）。"""
    return bool(_mapped_groups(history, store))


def expected_groups(history: str, store: RuleStore) -> frozenset[str]:
    """期望组集合：标注「不覆盖本情形」→ 空集（正确行为＝不提示）；否则为映射组集合。"""
    if "不覆盖" in history:
        return frozenset()
    return _mapped_groups(history, store)


def judge(history: str, description: str, store: RuleStore,
          matcher: NegRuleMatcher) -> tuple[bool, bool]:
    """单案例判定：返回（判定正确?, 是否覆盖清单项）。

    覆盖案例要求命中期望组（召回）；标注「不覆盖」的案例要求零命中（触发即误报）。
    """
    covered = "不覆盖" not in history
    got = matcher.match(description or "").groups()
    if covered:
        return bool(got & expected_groups(history, store)), True
    return not got, False


def main() -> int:
    """跑抽样 20 条与全量参考，输出召回/误报/含词子集三口径。"""
    config = load_config()
    store = load_store_from_config()
    matcher = NegRuleMatcher(store)
    cases_file = Path(config["data_repo"]) / config["audit_cases_file"]
    with open(cases_file, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    mappable = sorted(
        (r for r in rows if is_mappable(r["历史适用条款_文号及条款号"] or "", store)),
        key=lambda r: r["案例编号"],
    )
    covered = [r for r in mappable if "不覆盖" not in (r["历史适用条款_文号及条款号"] or "")]
    uncovered = [r for r in mappable if "不覆盖" in (r["历史适用条款_文号及条款号"] or "")]
    logger.info("案例总数 %d，历史条款可等价组映射 %d 条（覆盖清单项 %d / 标注不覆盖 %d）",
                len(rows), len(mappable), len(covered), len(uncovered))

    sample = random.Random(_SEED).sample(mappable, _SAMPLE_SIZE)
    sample_ok = 0
    sample_covered_ok = sample_covered_n = 0
    sample_fp = sample_uncovered_n = 0
    for row in sample:
        history = row["历史适用条款_文号及条款号"] or ""
        description = row["改编后申报场景描述_体检输入"] or ""
        ok, is_covered = judge(history, description, store, matcher)
        result = matcher.match(description)
        sample_ok += ok
        if is_covered:
            sample_covered_ok += ok
            sample_covered_n += 1
        else:
            sample_fp += not ok
            sample_uncovered_n += 1
        logger.info(
            "%s %s %s 命中组=%s 命中规则=%s",
            row["案例编号"], "✓" if ok else "✗",
            "覆盖清单项" if is_covered else "标注不覆盖",
            ",".join(sorted(result.groups())) or "-",
            "；".join(f"{h.rule.doc_no}{h.rule.clause_no}" for h in result.hits[:3]) or "-",
        )
    logger.info(
        "抽样 %d 条判定正确 %d；覆盖 %d 条命中 %d（%.1f%%），标注不覆盖 %d 条误触发 %d（%.1f%%）",
        len(sample), sample_ok,
        sample_covered_n, sample_covered_ok,
        100.0 * sample_covered_ok / sample_covered_n if sample_covered_n else 0.0,
        sample_uncovered_n, sample_fp,
        100.0 * sample_fp / sample_uncovered_n if sample_uncovered_n else 0.0,
    )

    recall_ok = sum(
        judge(r["历史适用条款_文号及条款号"] or "",
              r["改编后申报场景描述_体检输入"] or "", store, matcher)[0]
        for r in covered
    )
    false_positive = sum(
        not judge(r["历史适用条款_文号及条款号"] or "",
                  r["改编后申报场景描述_体检输入"] or "", store, matcher)[0]
        for r in uncovered
    )
    logger.info("全量参考——已知违规召回率（覆盖清单项 %d 条）：%d/%d = %.1f%%",
                len(covered), recall_ok, len(covered), 100.0 * recall_ok / len(covered))
    logger.info("全量参考——误报率（标注不覆盖 %d 条中引擎触发 %d 条）= %.1f%%",
                len(uncovered), false_positive, 100.0 * false_positive / len(uncovered))

    covered_with_terms = [
        r for r in covered
        if matcher.match(r["改编后申报场景描述_体检输入"] or "").description_terms
    ]
    hits_with_terms = sum(
        judge(r["历史适用条款_文号及条款号"] or "",
              r["改编后申报场景描述_体检输入"] or "", store, matcher)[0]
        for r in covered_with_terms
    )
    logger.info(
        "含禁止用途词子集参考（覆盖清单项且描述含词 %d 条）：命中 %d/%d = %.1f%%",
        len(covered_with_terms), hits_with_terms, len(covered_with_terms),
        100.0 * hits_with_terms / len(covered_with_terms) if covered_with_terms else 0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
