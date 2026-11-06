"""评测跑批入口（3.4 评测与消融 · PR2 · feat/eval-runner）。

用法::

    # 10 条 smoke，验证全流程能出报告
    python -m eval.runner --cases ../数据仓/审计公告/案例标注表.csv --limit 10 --out runs/smoke

    # 全量
    python -m eval.runner --cases ../数据仓/审计公告/案例标注表.csv --out runs/full

    # 完整系统 + 逐模块消融（7 组对比矩阵）
    python -m eval.runner --cases ... --ablate-all --out runs/ablation

产出（``--out`` 目录下）：
* ``metrics.json`` —— 三项指标 + 达标判定 + 元数据
* ``details.csv`` —— 逐案例明细
* ``ablation_matrix.csv`` —— 多组对比（仅 ``--ablate`` / ``--ablate-all`` 时）
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.config import SWITCH_DOCS, SWITCH_NAMES, AblationSwitches
from eval.dataset import EvalCase, RegistryError, load_cases, take
from eval.gating import ablation_context
from eval.metrics import (
    CaseOutcome,
    EvalThresholds,
    MetricsSummary,
    evaluate_case,
    summarize,
)
from eval.pipeline import NullPipeline, Pipeline

__all__ = ["GroupResult", "run_group", "main"]

logger = logging.getLogger(__name__)


@dataclass
class GroupResult:
    """一个消融组的完整结果。"""

    label: str
    switches: AblationSwitches
    outcomes: list[CaseOutcome] = field(default_factory=list)
    summary: MetricsSummary | None = None
    trace: dict[str, Any] = field(default_factory=dict)


def _load_registry(path: str | Path | None) -> dict[str, Any] | None:
    """读取文号注册表导出（团队数据仓），无则返回 ``None``（引用错误率记 N/A）。

    :raises FileNotFoundError: 给了路径但文件不存在
    :raises RegistryError: 文件内容无法建立「标识 → 元数据」索引
    """
    if path is None:
        return None
    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"找不到 registry 导出：{registry_path}")
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        if not raw:
            raise RegistryError(
                f"registry 导出为空对象：{registry_path}；"
                "空索引会让所有引用都判为无法核验、引用错误率显示 100%，属误导性结果。"
            )
        return dict(raw)
    if isinstance(raw, list):
        if not raw:
            raise RegistryError(
                f"registry 导出为空列表：{registry_path}；"
                "空索引会让所有引用都判为无法核验、引用错误率显示 100%，属误导性结果。"
            )
        # 约定：列表元素含 "ref_id" 字段时按其建索引
        index: dict[str, Any] = {}
        for position, item in enumerate(raw, start=1):
            if not isinstance(item, Mapping) or "ref_id" not in item:
                raise RegistryError(
                    f"registry 导出的第 {position} 个元素缺少 ref_id 字段：{item!r}；"
                    '列表格式要求每个元素都形如 {"ref_id": "...", ...}。'
                )
            index[str(item["ref_id"])] = item
        if not index:
            raise RegistryError(
                f"registry 导出未能建出索引：{registry_path}；"
                "空索引会让所有引用都判为无法核验、引用错误率显示 100%，属误导性结果。"
            )
        return index
    raise RegistryError(f"registry 导出格式不支持：{type(raw).__name__}（应为 JSON 对象或列表）")


def run_group(
    cases: Sequence[EvalCase],
    switches: AblationSwitches,
    pipeline: Pipeline,
    thresholds: EvalThresholds,
    registry: Mapping[str, Any] | None,
) -> GroupResult:
    """在一个消融配置下跑完全部案例。"""
    result = GroupResult(label=switches.ablated_label(), switches=switches)
    with ablation_context(switches) as trace:
        for case in cases:
            prediction = pipeline.predict(case)
            result.outcomes.append(evaluate_case(case, prediction, registry))
    result.summary = summarize(result.outcomes, thresholds, registry_provided=registry is not None)
    result.trace = trace.summary()
    logger.info(
        "组 %s：%d 条，匹配=%s 召回=%s 引用错误=%s",
        result.label,
        len(result.outcomes),
        _fmt(result.summary.match_rate),
        _fmt(result.summary.recall),
        _fmt(result.summary.citation_error_rate),
    )
    return result


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _write_details(path: Path, outcomes: Iterable[CaseOutcome]) -> None:
    """逐案例明细写 CSV（UTF-8 BOM，Excel 直接打开不乱码）。"""
    rows = [item.to_row() for item in outcomes]
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix(path: Path, groups: Sequence[GroupResult]) -> None:
    """多组对比矩阵。"""
    fieldnames = [
        "group",
        "disabled_switches",
        "case_count",
        "match_rate",
        "recall",
        "citation_error_rate",
        "forbidden_violations",
        "all_pass",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            summary = group.summary
            assert summary is not None
            writer.writerow(
                {
                    "group": group.label,
                    "disabled_switches": "+".join(group.switches.disabled()),
                    "case_count": summary.case_count,
                    "match_rate": "" if summary.match_rate is None else f"{summary.match_rate:.4f}",
                    "recall": "" if summary.recall is None else f"{summary.recall:.4f}",
                    "citation_error_rate": (
                        ""
                        if summary.citation_error_rate is None
                        else f"{summary.citation_error_rate:.4f}"
                    ),
                    "forbidden_violations": summary.forbidden_violations,
                    "all_pass": int(summary.all_pass()),
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.runner",
        description="粤策哨评测跑批（团队手册 3.4）：三项指标 + 消融对比",
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="案例标注表 CSV 路径（团队数据仓，不入库）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（smoke 用）")
    parser.add_argument("--out", default="runs/latest", help="输出目录，默认 runs/latest")
    parser.add_argument(
        "--registry", default=None,
        help="文号注册表导出 JSON；不给则引用错误率记 N/A")
    parser.add_argument(
        "--ablate",
        action="append",
        default=[],
        metavar="SWITCH",
        help=f"额外跑一个消融组（移除该模块），可重复；可选：{', '.join(SWITCH_NAMES)}",
    )
    parser.add_argument(
        "--pipeline",
        choices=("null", "rules"),
        default="null",
        help="被测管线：null=占位（链路验证）；rules=规则引擎基线"
             "（需根目录 config.json 指向数据仓）",
    )
    parser.add_argument(
        "--ablate-all",
        action="store_true",
        help="完整系统 + 逐个移除六组开关，输出 7 组对比矩阵",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.85, help="匹配率验收线，默认 0.85"
    )
    parser.add_argument(
        "--recall-threshold", type=float, default=0.80, help="召回验收线，默认 0.80")
    parser.add_argument(
        "--citation-error-threshold", type=float, default=0.05, help="引用错误率上限，默认 0.05"
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser


def _build_pipeline(name: str) -> Pipeline:
    """按名构建被测管线；rules 需要根目录 config.json（数据仓路径）。"""
    if name == "null":
        return NullPipeline()
    if name == "rules":
        from eval.rules_pipeline import RulesPipeline  # 局部 import：避免 null 路径强依赖 rules

        try:
            return RulesPipeline()
        except FileNotFoundError as exc:
            raise SystemExit(
                f"--pipeline rules 需要根目录 config.json 且数据仓可访问：{exc}"
            ) from exc
    raise ValueError(f"未知管线：{name}")


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；返回进程退出码（0=全部指标达标，1=有指标未达标，2=配置错误）。"""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        cases = take(load_cases(args.cases), args.limit)
        registry = _load_registry(args.registry)
        for switch in args.ablate:
            if switch not in SWITCH_NAMES:
                raise ValueError(f"--ablate 取值非法：{switch}；可选：{', '.join(SWITCH_NAMES)}")
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    thresholds = EvalThresholds(
        match_rate=args.match_threshold,
        recall=args.recall_threshold,
        citation_error_rate=args.citation_error_threshold,
    )
    pipeline = _build_pipeline(args.pipeline)

    groups: list[GroupResult] = [
        run_group(cases, AblationSwitches.default(), pipeline, thresholds, registry)]
    ablate_targets: list[str] = []
    if args.ablate_all:
        ablate_targets = list(SWITCH_NAMES)
    else:
        ablate_targets = list(dict.fromkeys(args.ablate))
    for switch in ablate_targets:
        switches = AblationSwitches.default().with_overrides(**{switch: False})
        groups.append(run_group(cases, switches, pipeline, thresholds, registry))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = groups[0]
    assert primary.summary is not None

    metrics_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "pipeline": pipeline.name,
        "cases_file": str(Path(args.cases).resolve()),
        "case_limit": args.limit,
        "registry_provided": registry is not None,
        "switches": primary.switches.to_dict(),
        "switch_docs": {name: SWITCH_DOCS[name] for name in SWITCH_NAMES},
        "trace": primary.trace,
        **primary.summary.to_dict(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_details(out_dir / "details.csv", primary.outcomes)
    if len(groups) > 1:
        _write_matrix(out_dir / "ablation_matrix.csv", groups)

    print(
        "\n".join(
            [
                f"pipeline : {pipeline.name}（占位管线，数字无实验意义；接真实管线后重跑）",
                f"cases    : {primary.summary.case_count} 条（limit={args.limit}）",
                f"匹配率   : {_fmt(primary.summary.match_rate)}  线 {thresholds.match_rate:.0%}",
                f"召回率   : {_fmt(primary.summary.recall)}  线 {thresholds.recall:.0%}",
                f"引用错误 : {_fmt(primary.summary.citation_error_rate)}"
                f"  上限 {thresholds.citation_error_rate:.0%}",
                f"姿态违规 : {primary.summary.forbidden_violations} 处",
                f"输出目录 : {out_dir.resolve()}",
            ]
        )
    )
    return 0 if primary.summary.all_pass() else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
