"""τ 标定脚本（可复跑）。

按 ``docs/rag-rerank-标定.md`` §2 记载的协议重跑标定：

    每条查询 → 稠密检索 top-40（含 A6 时效硬过滤）→ Cross-Encoder 逐对打分 →
    sigmoid 归一化 → 取最高分。

再把最高分按 A/B/C 三组统计「漏放率 / 误拦率」，输出建议 τ 区间。

用法
----
.. code-block:: bash

    # 在本仓库根目录，且 models/ 下已放好两套权重（见下方「依赖」）
    python -m eval.calibration.calibrate_rerank --config config.json

    # 也可直接当脚本跑
    python eval/calibration/calibrate_rerank.py --config config.json

依赖（本机，均不入库）
----------------------
- 稠密检索：``models/Qwen3-Embedding-0.6B``（rag.indexer 的 Qwen3Embedder 读取）
- 重排：``models/bge-reranker-v2-m3``（rag.rerank 的 CrossEncoderScorer 读取）
- ``pip install -r requirements.txt`` 后再补装 ``sentence-transformers``

查询集
------
A/B/C 三组各一个纯文本文件（每行一条查询，``#`` 开头为注释），与本脚本同目录：

- ``queries-A组.txt``（20 条）／ ``queries-B组.txt``（10 条）／ ``queries-C组.txt``（10 条）
  三组齐备，可直接复现标定文档 §3 的分布与 τ 扫表。任一组缺失时脚本告警并跳过，
  此时算不出完整的 τ 建议区间。

三组含义（与标定文档 §2 一致）：

- A：域内、且语料确实覆盖（20 条）——「误拦」的计数来源：A 组判未覆盖 = 误拦。
- B：近域（10 条）——「漏放」的计数来源之一。**注意这组标注偏宽**：10 条被放行的
  8 条里，7 条语料确实覆盖、1 条仅「部分相关」（宅基地使用权纠纷，命中的条款答不了
  这个问题），见标定文档 §5／§6。这 7 条被放行属于正确判断而非漏放，
  详见 ``README.md`` 的「解读漏放率前必读」。
- C：完全域外（10 条）——「漏放」的计数来源之一：C 组判有依据 = 漏放。
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("calibrate_rerank")

# 以脚本方式运行时，把仓库根目录加进 sys.path，保证能 import rag.*
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.rerank import build_reranker  # noqa: E402
from rag.retriever import build_retriever  # noqa: E402

# A/B/C 三组的查询文件（与本脚本同目录）
GROUP_FILES = {
    "A": "queries-A组.txt",
    "B": "queries-B组.txt",
    "C": "queries-C组.txt",
}

# τ 扫表的取值粒度（与标定文档 §3.2 一致）
TAU_SWEEP = [round(0.500 + 0.025 * i, 3) for i in range(9)]  # 0.500 .. 0.700


def load_queries(group_file: Path, group: str) -> list[str]:
    """读一组查询（每行一条，去空行与 ``#`` 注释）。

    文件缺失时告警并返回空列表，该组不参与统计——缺 A 组算不出上沿、缺 C 组算不出下沿。
    """
    if not group_file.is_file():
        logger.warning(
            "查询集 %s 不存在（%s），该组跳过；如需完整 τ 区间，请先补上该组快照",
            group,
            group_file.name,
        )
        return []
    queries: list[str] = []
    for line in group_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            queries.append(stripped)
    return queries


def score_one_query(retriever, reranker, query: str, top_k: int) -> float:
    """跑单条查询，返回归一化后的最高重排分（sigmoid 后的 ``max``）。"""
    retrieval = retriever.retrieve(query, top_k=top_k, count_candidates=False)
    if not getattr(retrieval, "temporal_filtered", True):
        logger.warning("查询 %r 的检索未施加时效过滤，结果可能含已失效条款", query)
    result = reranker.rerank(query, retrieval.hits, top_k_in=top_k)
    # 取最高分而不是 hits[0]：不依赖入参已排序（与 rag.rerank.QualityGate 同口径）
    return max((hit.rerank_score for hit in result.hits), default=0.0)


def _fmt_table(rows: list[list[str]]) -> str:
    """把二维列表排成对齐的纯文本表。"""
    widths = [max(len(str(cell)) for cell in col) for col in zip(*rows, strict=False)]
    lines = []
    for row in rows:
        lines.append(
            "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="粤策哨重排质量门 τ 标定（复现 docs/rag-rerank-标定.md §2）"
    )
    parser.add_argument("--config", default="config.json", help="本机配置文件路径")
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="每查询检索候选数（标定文档固定 40；降级重试口径需另跑 max@80）",
    )
    parser.add_argument(
        "--queries-dir",
        default=None,
        help="查询集目录（默认与本脚本同目录）",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("配置文件不存在：%s", config_path)
        return 2
    raw_config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    queries_dir = Path(args.queries_dir) if args.queries_dir else Path(__file__).resolve().parent

    # 装配真实检索器与重排器（重量级依赖均在各自函数内局部 import）
    retriever = build_retriever(raw_config)
    reranker = build_reranker(raw_config)

    scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for group in ("A", "B", "C"):
        for query in load_queries(queries_dir / GROUP_FILES[group], group):
            top = score_one_query(retriever, reranker, query, args.top_k)
            scores[group].append((query, top))
            logger.info("[%s] %.3f  %s", group, top, query)

    total = sum(len(v) for v in scores.values())
    if total == 0:
        logger.error("没有任何查询可跑：请检查查询集文件（至少 queries-B组.txt 应就位）")
        return 2

    # ---- 三组最高分分布 ----
    print("\n== 三组最高分分布（sigmoid 归一化后） ==")
    dist_rows: list[list[str]] = [["组", "条数", "min", "中位", "max"]]
    group_max: dict[str, float] = {}
    group_min: dict[str, float] = {}
    for group in ("A", "B", "C"):
        values = [score for _, score in scores[group]]
        if not values:
            dist_rows.append([group, "0（待补充）", "-", "-", "-"])
            continue
        group_min[group] = min(values)
        group_max[group] = max(values)
        dist_rows.append(
            [
                group,
                str(len(values)),
                f"{min(values):.3f}",
                f"{statistics.median(values):.3f}",
                f"{max(values):.3f}",
            ]
        )
    print(_fmt_table(dist_rows))

    # ---- τ 扫表：漏放率 / 误拦率 ----
    # 误拦 = A 组（域内已覆盖）被 τ 拦下；漏放 = B/C 组（近域/域外）被 τ 放行。
    a_values = [score for _, score in scores["A"]]
    bc_values = [score for _, score in scores["B"]] + [score for _, score in scores["C"]]
    print("\n== τ 扫表（漏放/误拦） ==")
    sweep_rows: list[list[str]] = [["τ", "误拦率(A)", "漏放率(B/C)"]]
    for tau in TAU_SWEEP:
        miss_rate = (
            sum(1 for v in a_values if v < tau) / len(a_values) if a_values else float("nan")
        )
        leak_rate = (
            sum(1 for v in bc_values if v >= tau) / len(bc_values)
            if bc_values
            else float("nan")
        )
        sweep_rows.append(
            [
                f"{tau:.3f}",
                "-" if a_values == [] else f"{miss_rate:.0%}",
                "-" if bc_values == [] else f"{leak_rate:.0%}",
            ]
        )
    print(_fmt_table(sweep_rows))

    # ---- 建议 τ 区间 ----
    print("\n== 建议 τ 区间 ==")
    if "A" not in group_min or "C" not in group_max:
        print(
            "A/C 组查询快照缺失，无法给出完整区间（平台下沿需要 C 组 max、上沿需要 A 组 min）。\n"
            "请确认 queries-A组.txt / queries-C组.txt 与本脚本同目录（或用 --queries-dir 指过去）。"
        )
    elif group_min["A"] > group_max["C"]:
        lower = group_max["C"]
        upper = group_min["A"]
        suggested = round(lower + (upper - lower) * 0.43, 3)  # 平台区「中部偏下」
        print(
            f"零误拦平台区 = ({lower:.3f}, {upper:.3f})：高于 C 组上沿、低于 A 组下沿。\n"
            f"建议 τ 取平台区中部偏下 ≈ {suggested:.3f}（当前默认 0.55 即按此口径定出）。"
        )
    else:
        print(
            f"A 组下沿 {group_min['A']:.3f} 未高于 C 组上沿 {group_max['C']:.3f}，"
            "两组分数重叠、不存在零误拦平台区；请检查语料/模型或重新标注分组。"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
