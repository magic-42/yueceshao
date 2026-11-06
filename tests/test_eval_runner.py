"""评测跑批的单元测试（3.4 · PR2 · feat/eval-runner）。

测试数据全部是**明显虚构的测试脚手架**（``CASE-A`` / ``REF-A`` 这类中性标识），
不含任何政策文号——红线第 1 条禁止编造文号，测试夹具也不破例。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eval.config import AblationSwitches
from eval.dataset import DatasetError, EvalCase, RegistryError, load_cases, take
from eval.gating import gated
from eval.metrics import (
    FORBIDDEN_PHRASES,
    REPORT_DISCLAIMER,
    CasePrediction,
    EvalThresholds,
    evaluate_case,
    strip_disclaimer,
    summarize,
)
from eval.runner import _load_registry, main, run_group

HEADER = ["case_id", "query", "expected_verdict", "expected_refs", "notes"]


def _write_cases(path: Path, rows: list[list[str]]) -> Path:
    """写一份标注表。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


class FixedPipeline:
    """按 case_id 返回固定预测，用于验证指标计算。"""

    name = "fixed"

    def __init__(self, predictions: dict[str, CasePrediction]) -> None:
        self._predictions = predictions
        self.calls = 0

    def predict(self, case: EvalCase) -> CasePrediction:
        self.calls += 1
        return self._predictions.get(case.case_id, CasePrediction(verdict="miss"))


class StagedPipeline:
    """带一个可消融 stage 的管线，用于验证 trace 能观测到 bypass。"""

    name = "staged"

    def __init__(self) -> None:
        self.stage_calls = 0

    def predict(self, case: EvalCase) -> CasePrediction:
        self._rerank(["x"])
        return CasePrediction(verdict="miss")

    @gated("rerank", stage="retrieve.rerank")
    def _rerank(self, candidates: list[str]) -> list[str]:
        self.stage_calls += 1
        return candidates


# ---- 数据读取 -------------------------------------------------------------


def test_load_cases_parses_rows(tmp_path: Path) -> None:
    """正常读取：判定归一化小写、refs 分号拆分去重、备注保留。"""
    path = _write_cases(
        tmp_path / "cases.csv",
        [
            ["CASE-A", "描述甲", "HIT", "REF-A; REF-B;REF-A", "备注甲"],
            ["CASE-B", "描述乙", "", "", ""],
        ],
    )
    cases = load_cases(path)
    assert [case.case_id for case in cases] == ["CASE-A", "CASE-B"]
    assert cases[0].expected_verdict == "hit"
    assert cases[0].expected_refs == ("REF-A", "REF-B")
    assert cases[0].notes == "备注甲"
    assert cases[1].expected_verdict is None
    assert cases[1].expected_refs == ()


def test_load_cases_missing_file_mentions_data_hub(tmp_path: Path) -> None:
    """找不到文件时提示标注表在团队数据仓，而不是含糊报错。"""
    with pytest.raises(DatasetError, match="团队数据仓"):
        load_cases(tmp_path / "nope.csv")


def test_load_cases_requires_columns(tmp_path: Path) -> None:
    """缺必需列直接报错。"""
    path = tmp_path / "bad.csv"
    path.write_text("case_id\nCASE-A\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="缺少必需列"):
        load_cases(path)


def test_load_cases_rejects_duplicate_id_and_bad_verdict(tmp_path: Path) -> None:
    """case_id 重复、判定取值非法都要当场报错。"""
    dup = _write_cases(tmp_path / "dup.csv", [
        ["CASE-A", "x", "", "", ""], ["CASE-A", "y", "", "", ""]])
    with pytest.raises(DatasetError, match="重复"):
        load_cases(dup)

    bad = _write_cases(tmp_path / "bad_verdict.csv", [["CASE-A", "x", "maybe", "", ""]])
    with pytest.raises(DatasetError, match="expected_verdict 非法"):
        load_cases(bad)


def test_take_limits_rows(tmp_path: Path) -> None:
    """smoke 用前 N 条。"""
    path = _write_cases(
        tmp_path / "cases.csv", [[f"CASE-{i}", f"q{i}", "", "", ""] for i in range(5)]
    )
    cases = load_cases(path)
    assert len(take(cases, 10)) == 5
    assert len(take(cases, 3)) == 3
    with pytest.raises(ValueError):
        take(cases, 0)


# ---- 单案例评测 -----------------------------------------------------------


def test_evaluate_case_matches_and_recall() -> None:
    """判定一致 + 条款级召回只计交集。"""
    case = EvalCase(
        case_id="CASE-A", query="q", expected_verdict="hit", expected_refs=("REF-A", "REF-B")
    )
    prediction = CasePrediction(verdict="hit", refs=("REF-A",), citations=("REF-A",))
    outcome = evaluate_case(case, prediction, registry={"REF-A": {}})
    assert outcome.verdict_match is True
    assert outcome.refs_hit == ("REF-A",)
    assert outcome.recall_mode == "ref-level"
    assert outcome.citations_bad == 0


def test_evaluate_case_marks_case_level_recall() -> None:
    """只标 hit 未给具体条款时退化为整案召回，并在明细里说清楚。"""
    case = EvalCase(case_id="CASE-A", query="q", expected_verdict="hit")
    outcome = evaluate_case(case, CasePrediction(verdict="hit"))
    assert outcome.recall_mode == "case-level"


def test_evaluate_case_counts_bad_citation() -> None:
    """registry 里没有的引用计为错误引用。"""
    case = EvalCase(case_id="CASE-A", query="q", expected_verdict="hit", expected_refs=("REF-A",))
    prediction = CasePrediction(verdict="hit", refs=("REF-A",), citations=("REF-A", "REF-GHOST"))
    outcome = evaluate_case(case, prediction, registry={"REF-A": {}})
    assert outcome.citations_total == 2
    assert outcome.citations_bad == 1


def test_evaluate_case_flags_forbidden_phrasing() -> None:
    """姿态禁语命中即记录（红线第 3 条）。"""
    case = EvalCase(case_id="CASE-A", query="q", expected_verdict="miss")
    text = f"经核对，该项目{next(iter(FORBIDDEN_PHRASES))}。"
    outcome = evaluate_case(case, CasePrediction(verdict="miss", text=text))
    assert outcome.forbidden_hits
    assert outcome.forbidden_hits[0] in FORBIDDEN_PHRASES


def test_forbidden_phrases_cover_both_directions() -> None:
    """正向放行与反向判定都算姿态违规，不能只挡「合规 / 可以申报」。"""
    case = EvalCase(case_id="CASE-A", query="q", expected_verdict="miss")
    for phrase in ("合规", "可以申报", "违规", "不能申报"):
        assert phrase in FORBIDDEN_PHRASES
        outcome = evaluate_case(
            case, CasePrediction(verdict="miss", text=f"经核对，{phrase}。")
        )
        assert phrase in outcome.forbidden_hits, f"{phrase} 未被计入姿态违规"


def test_disclaimer_is_not_counted_as_forbidden() -> None:
    """渲染层免责声明含「合规」二字，但不得被算作姿态违规。

    接上真实管线后每份报告都会带这句免责声明；若不剔除，姿态违规数会恒为正数，
    指标彻底失去意义。
    """
    case = EvalCase(case_id="CASE-A", query="q", expected_verdict="miss")
    text = f"四段式报告正文。\n{REPORT_DISCLAIMER}"
    outcome = evaluate_case(case, CasePrediction(verdict="miss", text=text))
    assert outcome.forbidden_hits == ()

    # 换了措辞的免责声明同样要挡住
    for hint_line in (
        "本系统不构成政策判断，请咨询主管部门。",
        "以上内容仅为申报前自查辅助参考。",
        "最终以政策法规原文及主管部门意见为准。",
    ):
        outcome = evaluate_case(
            case, CasePrediction(verdict="miss", text=f"正文。\n{hint_line}")
        )
        assert outcome.forbidden_hits == (), hint_line


def test_disclaimer_stripping_keeps_body_violation() -> None:
    """剔除免责声明时不能连带放过正文里的真实违规。"""
    case = EvalCase(case_id="CASE-A", query="q", expected_verdict="miss")
    text = f"经核对，该项目合规。\n{REPORT_DISCLAIMER}"
    outcome = evaluate_case(case, CasePrediction(verdict="miss", text=text))
    assert "合规" in outcome.forbidden_hits


def test_strip_disclaimer_handles_empty_and_plain_text() -> None:
    """无免责声明时原样返回；空串不报错。"""
    assert strip_disclaimer("") == ""
    assert strip_disclaimer("普通正文") == "普通正文"


# ---- 指标汇总 -------------------------------------------------------------


def test_summarize_computes_three_metrics() -> None:
    """三项指标：匹配=3/4、条款召回=2/3、引用错误=1/5。"""
    cases = [
        EvalCase("c1", "q", "hit", ("REF-A", "REF-B")),
        EvalCase("c2", "q", "hit", ("REF-C",)),
        EvalCase("c3", "q", "miss"),
        EvalCase("c4", "q", None),
    ]
    predictions = {
        "c1": CasePrediction("hit", ("REF-A",), ("REF-A", "REF-B")),
        "c2": CasePrediction("miss", (), ("REF-C",)),
        "c3": CasePrediction("miss", (), ("REF-D",)),
        "c4": CasePrediction("miss", (), ("REF-E",)),
    }
    registry = {"REF-A": {}, "REF-B": {}, "REF-C": {}, "REF-D": {}}
    outcomes = [evaluate_case(c, predictions[c.case_id], registry) for c in cases]
    summary = summarize(outcomes, EvalThresholds(), registry_provided=True)

    assert summary.labeled_verdict_count == 3
    assert summary.verdict_unlabeled == 1
    assert summary.match_rate == pytest.approx(2 / 3)  # c1 ✓ c2 ✗ c3 ✓
    assert summary.recall_denominator == 3
    assert summary.recall == pytest.approx(1 / 3)  # 只有 c1 命中 REF-A
    assert summary.citation_total == 5
    assert summary.citation_bad == 1
    assert summary.citation_error_rate == pytest.approx(0.2)


def test_summarize_returns_none_when_not_computable() -> None:
    """无标注判定 / 无 registry 时指标为 None（报告显示 N/A），不能拿 0 冒充。"""
    outcomes = [evaluate_case(EvalCase("c1", "q", None), CasePrediction("miss"))]
    summary = summarize(outcomes, EvalThresholds(), registry_provided=False)
    assert summary.match_rate is None
    assert summary.recall is None
    assert summary.citation_error_rate is None
    assert summary.all_pass() is False  # 有 N/A 视为未达标


def test_summary_pass_flags_follow_thresholds() -> None:
    """达标判定按手册的三个数字。"""
    outcomes = [
        evaluate_case(
            EvalCase("c1", "q", "hit", ("REF-A",)),
            CasePrediction("hit", ("REF-A",), ("REF-A",)), {"REF-A": {}}),
        evaluate_case(EvalCase("c2", "q", "miss"), CasePrediction("miss")),
    ]
    summary = summarize(outcomes, EvalThresholds(), registry_provided=True)
    assert summary.passes() == {
        "match_rate": True,
        "recall": True,
        "citation_error_rate": True,
    }
    assert summary.all_pass() is True


# ---- 跑批 ---------------------------------------------------------------


def test_run_group_records_bypass_trace(tmp_path: Path) -> None:
    """消融组跑批时 trace 里能看到被跳过的开关（PR1 与 PR2 的接缝）。"""
    path = _write_cases(tmp_path / "cases.csv", [["CASE-A", "q", "miss", "", ""]])
    cases = load_cases(path)
    pipeline = StagedPipeline()

    # 完整系统：stage 被调用，无 bypass
    full = run_group(
        cases, AblationSwitches.default(), pipeline, EvalThresholds(), registry=None
    )
    assert pipeline.stage_calls == 1
    assert full.trace["event_count"] == 0

    # 消融组：stage 不再被调用，且 trace 记录了 rerank 的 bypass
    ablated = run_group(
        cases,
        AblationSwitches.default().with_overrides(rerank=False),
        pipeline,
        EvalThresholds(),
        registry=None,
    )
    assert pipeline.stage_calls == 1  # 没有新增调用——实现确实缺席
    assert ablated.label == "w/o:rerank"
    assert ablated.trace["event_count"] == 1
    assert ablated.trace["bypassed_switches"] == ["rerank"]
    assert ablated.trace["stages"]["rerank"] == ["retrieve.rerank"]


def test_runner_writes_metrics_and_details(tmp_path: Path) -> None:
    """10 条 smoke 跑通全流程并落盘 metrics.json / details.csv。"""
    rows = [[f"CASE-{i:02d}", f"描述{i}", "miss", "", ""] for i in range(10)]
    cases_path = _write_cases(tmp_path / "cases.csv", rows)
    out_dir = tmp_path / "out"

    exit_code = main(
        ["--cases", str(cases_path), "--limit", "10", "--out", str(out_dir),
         "--log-level", "WARNING"]
    )

    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "details.csv").is_file()
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["case_count"] == 10
    assert payload["pipeline"] == "null"
    assert set(payload["switches"]) == set(AblationSwitches.default().to_dict())
    # 引用错误率无 registry → N/A → all_pass=False → 退出码 1
    assert payload["metrics"]["citation_error_rate"] is None
    assert exit_code == 1

    with (out_dir / "details.csv").open(encoding="utf-8-sig", newline="") as handle:
        detail_rows = list(csv.DictReader(handle))
    assert len(detail_rows) == 10


def test_runner_ablate_all_writes_matrix(tmp_path: Path) -> None:
    """--ablate-all 产出 7 组（完整系统 + 六组单消融）对比矩阵。"""
    cases_path = _write_cases(
        tmp_path / "cases.csv", [["CASE-00", "q", "miss", "", ""]]
    )
    out_dir = tmp_path / "out"
    main(
        [
            "--cases",
            str(cases_path),
            "--out",
            str(out_dir),
            "--ablate-all",
            "--log-level",
            "WARNING",
        ]
    )
    with (out_dir / "ablation_matrix.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert rows[0]["group"] == "full"
    expected_disabled = set(AblationSwitches.default().to_dict())
    assert {row["disabled_switches"] for row in rows[1:]} == expected_disabled


def test_runner_rejects_unknown_ablate_switch(tmp_path: Path) -> None:
    """--ablate 写错开关名返回退出码 2，不静默忽略。"""
    cases_path = _write_cases(tmp_path / "cases.csv", [["CASE-00", "q", "miss", "", ""]])
    exit_code = main(
        ["--cases", str(cases_path), "--ablate", "no_such_switch",
         "--out", str(tmp_path / "o"), "--log-level", "WARNING"]
    )
    assert exit_code == 2


# ---- registry 读取 -------------------------------------------------------


def test_load_registry_reads_mapping_and_list(tmp_path: Path) -> None:
    """两种合法形态：对象直接作索引；列表按 ref_id 字段建索引。"""
    as_object = tmp_path / "reg_object.json"
    as_object.write_text(json.dumps({"REF-A": {"title": "甲"}, "REF-B": {}}), encoding="utf-8")
    assert set(_load_registry(as_object)) == {"REF-A", "REF-B"}

    as_list = tmp_path / "reg_list.json"
    as_list.write_text(
        json.dumps([{"ref_id": "REF-A", "title": "甲"}, {"ref_id": "REF-B"}]), encoding="utf-8"
    )
    index = _load_registry(as_list)
    assert set(index) == {"REF-A", "REF-B"}
    assert index["REF-A"]["title"] == "甲"


def test_load_registry_none_means_na() -> None:
    """不传 registry 返回 None，由调用方把引用错误率记成 N/A。"""
    assert _load_registry(None) is None


def test_load_registry_rejects_list_without_ref_id(tmp_path: Path) -> None:
    """列表元素缺 ref_id 必须抛错，不能静默建出空索引。

    静默放行的后果：``registry_provided=True`` 但索引为空，所有引用都判为无法核验，
    引用错误率显示 100%——看起来像"系统全在编文号"，实际只是导出格式不对。
    """
    bad = tmp_path / "reg_bad.json"
    bad.write_text(json.dumps([{"title": "甲"}, {"title": "乙"}]), encoding="utf-8")
    with pytest.raises(RegistryError, match="ref_id"):
        _load_registry(bad)


def test_load_registry_rejects_empty(tmp_path: Path) -> None:
    """空对象 / 空列表同样抛错，理由同上（会产出误导性的 100%）。"""
    for name, payload in (("empty_obj.json", {}), ("empty_list.json", [])):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RegistryError, match="空"):
            _load_registry(path)


def test_load_registry_rejects_unsupported_shape(tmp_path: Path) -> None:
    """标量 / 字符串等不支持的形态抛错。"""
    bad = tmp_path / "reg_scalar.json"
    bad.write_text(json.dumps("REF-A"), encoding="utf-8")
    with pytest.raises(RegistryError, match="格式不支持"):
        _load_registry(bad)


def test_load_registry_missing_file(tmp_path: Path) -> None:
    """给了路径但文件不存在，报 FileNotFoundError（区别于格式错误）。"""
    with pytest.raises(FileNotFoundError):
        _load_registry(tmp_path / "nope.json")


def test_runner_exit_code_two_on_bad_registry(tmp_path: Path) -> None:
    """registry 格式错误走配置错误通道，退出码 2，不产出假指标。"""
    cases_path = _write_cases(tmp_path / "cases.csv", [["CASE-00", "q", "hit", "REF-A", ""]])
    bad_registry = tmp_path / "reg_bad.json"
    bad_registry.write_text(json.dumps([{"title": "甲"}]), encoding="utf-8")
    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--registry",
            str(bad_registry),
            "--out",
            str(tmp_path / "out"),
            "--log-level",
            "WARNING",
        ]
    )
    assert exit_code == 2
    assert not (tmp_path / "out" / "metrics.json").exists()
