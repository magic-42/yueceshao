"""tests/test_eval_rules_pipeline.py — feat/eval-rules-pipeline 单元测试。

覆盖：中文列名别名映射（真数据形态）、预期预警推断、「不覆盖本情形」段剔除、
refs 提取正则、RulesPipeline 适配（假规则库注入，零数据仓依赖）。

红线遵守：夹具全部人造标识（〔测试样例〕文号／CASE 编号），零真实语料。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.dataset import (
    COLUMN_ALIASES,
    DatasetError,
    extract_refs_from_text,
    load_cases,
)
from eval.metrics import CasePrediction
from eval.rules_pipeline import RulesPipeline
from eval.runner import _build_pipeline
from rules.loader import load_records
from rules.schema import RuleStore

FIXTURE = Path(__file__).parent / "fixtures" / "rules_sample.json"


# ---------------------------------------------------------------------------
# 列名别名与值推断
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, header: list[str], rows: list[list[str]]) -> Path:
    import csv

    path = tmp_path / name
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


CHINESE_HEADER = [
    "案例编号", "来源公告名称", "改编后申报场景描述_体检输入",
    "预期命中预警", "历史适用条款_文号及条款号", "备注",
]


class TestColumnAliases:
    def test_chinese_header_loads(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "cases.csv", CHINESE_HEADER, [
            ["C01", "测试公告", "我村拟申请资金修建灌溉设施。",
             "待P0（应命中：设施未建成即申报）",
             "粤测农〔2099〕1号第四条（预算监控口径）；现行参照载体粤测农〔2099〕2号第五条", ""],
            ["C02", "测试公告", "某产业园申请扶持资金。",
             "应提示：须依法履行采购程序",
             "粤测农〔2099〕1号第八条（不覆盖本情形）；粤测农〔2099〕3号第九条", ""],
        ])
        cases = load_cases(path)
        assert [c.case_id for c in cases] == ["C01", "C02"]
        # C01：refs 从非「不覆盖」段提取，两条款齐（归一为 文号#序号）
        assert cases[0].expected_refs == ("粤测农〔2099〕1号#4", "粤测农〔2099〕2号#5")
        # C02：「不覆盖本情形」段被剔除，只留可适用条款
        assert cases[1].expected_refs == ("粤测农〔2099〕3号#9",)
        # verdict 推断：应命中/应提示 → hit
        assert all(c.expected_verdict == "hit" for c in cases)
        assert "灌溉设施" in cases[0].query

    def test_standard_header_still_works(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "std.csv", ["case_id", "query", "expected_verdict"],
                      [["CASE-A", "描述甲", "hit"]])
        cases = load_cases(path)
        assert cases[0].expected_verdict == "hit"

    def test_standard_bad_verdict_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "bad.csv", ["case_id", "query", "expected_verdict"],
                      [["CASE-A", "描述甲", "maybe"]])
        with pytest.raises(DatasetError, match="非法"):
            load_cases(path)

    def test_chinese_uninferable_warning_becomes_unlabeled(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "cases.csv", CHINESE_HEADER,
                      [["C01", "测试公告", "描述", "（口径待定，暂无预警文案）",
                        "粤测农〔2099〕1号第四条", ""]])
        cases = load_cases(path)
        assert cases[0].expected_verdict is None

    def test_alias_table_maps_five_columns(self) -> None:
        for standard in ("case_id", "query", "expected_verdict", "expected_refs", "notes"):
            assert standard in COLUMN_ALIASES


class TestRefExtraction:
    def test_extracts_and_normalizes(self) -> None:
        """提取后归一为「文号#序号」——与规则库 source_ref 的归一形态可比。"""
        assert extract_refs_from_text(
            "依据粤财测〔2099〕1号第二十条（预算口径）") == ("粤财测〔2099〕1号#20",)
        assert extract_refs_from_text("粤测农〔2099〕1号第（二）项") == ()

    def test_bare_number_shorthand_not_matched(self) -> None:
        """「106号第十九条」简写不匹配（只认全形文号）——宁漏勿错。"""
        assert extract_refs_from_text("106号第十九条（口径）") == ()


# ---------------------------------------------------------------------------
# RulesPipeline 适配（假规则库注入）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rules_pipeline() -> RulesPipeline:
    return RulesPipeline(store=RuleStore(load_records(FIXTURE)))


class TestRulesPipeline:
    def test_hit_prediction_carries_refs(self, rules_pipeline: RulesPipeline) -> None:
        """描述含夹具条款术语（楼堂馆所）→ hit，refs 来自规则记录出处字段。"""
        from eval.dataset import EvalCase
        case = EvalCase(case_id="T1", query="我村拟申请资金建设办公用房（测试性楼堂馆所）")
        prediction = rules_pipeline.predict(case)
        assert isinstance(prediction, CasePrediction)
        assert prediction.verdict == "hit"
        assert prediction.refs and all(
            "[测试样例]" in ref or "测试样例" in ref for ref in prediction.refs)
        # 组全谱系口径：citations（直接命中）⊆ refs（含等价组成员）
        assert prediction.citations and set(prediction.citations) <= set(prediction.refs)

    def test_miss_prediction_on_clean_description(self, rules_pipeline: RulesPipeline) -> None:
        from eval.dataset import EvalCase
        case = EvalCase(case_id="T2", query="我村组织村民开展普通话培训班，无涉农资金申请内容。")
        prediction = rules_pipeline.predict(case)
        assert prediction.verdict == "miss" and prediction.refs == ()

    def test_text_empty_for_pure_deterministic_pipeline(
        self, rules_pipeline: RulesPipeline
    ) -> None:
        """纯确定性模板输出无 LLM 生成段：text 置空，姿态扫描定义上不适用。

        render_hits 的条款原文引文与等价组名可能天然含「合规/不合规」等字样
        （政策语料本身），真数据首跑实测 53 处误报——置空是诚实口径。
        """
        from eval.dataset import EvalCase
        case = EvalCase(case_id="T3", query="拟用资金进行测试性楼堂馆所建设。")
        prediction = rules_pipeline.predict(case)
        assert prediction.text == ""

    def test_runner_builds_rules_pipeline(self, monkeypatch) -> None:
        """runner._build_pipeline('rules') 在规则库不可达时给友好 SystemExit。"""
        def _boom(self, store=None, config_path=None):
            raise FileNotFoundError("config.json 不存在")

        monkeypatch.setattr("eval.rules_pipeline.RulesPipeline.__init__", _boom)
        with pytest.raises(SystemExit, match="config.json"):
            _build_pipeline("rules")
        assert _build_pipeline("null").name == "null"
