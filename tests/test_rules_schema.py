"""tests/test_rules_schema.py — feat/rules-schema 单元测试。

夹具：tests/fixtures/rules_sample.json（10 条人造样例规则，含 2 个人造等价组）。
红线遵守：测试不使用真实政策语料原文、不引用真实文号（样例文号均为〔测试样例〕前缀）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rules.loader import load_config, load_records, load_store_from_config, records_from_dicts
from rules.schema import EFFECT_VALID, GROUP_NONE, RuleRecord, RuleStore

FIXTURE = Path(__file__).parent / "fixtures" / "rules_sample.json"


@pytest.fixture(scope="module")
def store() -> RuleStore:
    return RuleStore(load_records(FIXTURE))


def _by_ref(store: RuleStore, ref: str) -> RuleRecord:
    """按「出处文号+条款号」取夹具记录。"""
    return next(r for r in store.records if r.source_ref == ref)


def test_load_sample_json_maps_all_fields(store: RuleStore) -> None:
    """10 条人造样例全部加载成功，字段映射与类型转换正确。"""
    assert len(store.records) == 10
    record = store.records[0]
    assert record.clause_no == "一"
    assert record.doc_index == 1                 # 文件序号转为 int
    assert record.carries_enumeration is False   # 布尔字段解析
    assert record.group_id == "测试组-A"


def test_active_filter_removes_invalidated(store: RuleStore) -> None:
    """已失效条款从现行规则集剔除；待人工核验保留。"""
    active = store.active_records()
    assert all(r.is_active for r in active)
    invalidated = [r for r in store.records if r.is_invalidated]
    assert len(active) == len(store.records) - len(invalidated)
    assert len(store.valid_records()) == sum(1 for r in store.records if r.is_effective)
    assert {r.effect_status for r in active} == {EFFECT_VALID, "待人工核验"}


def test_group_members_ordered_by_display_priority(store: RuleStore) -> None:
    """组内排序：现行有效优先，同效力时层级更具体优先（县>市>省>国家）。"""
    refs = [m.source_ref for m in store.group_members("测试组-A")]
    assert refs == [
        "〔测试样例〕省A-002号 第二条",   # 有效（省）
        "〔测试样例〕县A-009号 第三条",   # 待人工核验（县含乡镇，层级优先于市）
        "〔测试样例〕市A-003号 第三条",   # 待人工核验（市）
        "〔测试样例〕县A-004号 第四条",   # 已失效（县）
        "〔测试样例〕国A-001号 第一条",   # 已失效（国家）
    ]


def test_expand_equivalents_gives_other_levels(store: RuleStore) -> None:
    """等价组展开：命中一条时给出同组其他层级条款。"""
    province = _by_ref(store, "〔测试样例〕省A-002号 第二条")
    others = store.expand_equivalents(province)
    assert len(others) == 4 and province not in others
    assert {r.level for r in others} == {"国家", "市", "县", "县（含乡镇）"}


def test_display_substitutes_replaces_invalidated(store: RuleStore) -> None:
    """失效条款以组内现行有效同构条款替代展示（有效优先，其次待人工核验）。"""
    county = _by_ref(store, "〔测试样例〕县A-004号 第四条")
    assert [s.source_ref for s in store.display_substitutes(county)] == [
        "〔测试样例〕省A-002号 第二条",
        "〔测试样例〕县A-009号 第三条",
        "〔测试样例〕市A-003号 第三条",
    ]
    province = _by_ref(store, "〔测试样例〕省A-002号 第二条")
    assert store.display_substitutes(province) == [province]  # 非失效条款展示自身


def test_all_invalidated_group_and_ungrouped(store: RuleStore) -> None:
    """组内全部失效时无替代；未分组条款不参与等价组展开。"""
    city = _by_ref(store, "〔测试样例〕市C-007号 第一条")
    assert store.display_substitutes(city) == []
    assert len(store.group_members("测试组-B")) == 2
    ungrouped = _by_ref(store, "〔测试样例〕省B-005号 第一条")
    assert ungrouped.group_id == GROUP_NONE
    assert store.expand_equivalents(ungrouped) == []
    assert store.group_members(GROUP_NONE) == []


def test_missing_field_raises_with_index() -> None:
    """字段缺失时报错并定位到条目序号。"""
    with pytest.raises(ValueError, match="第 1 条记录结构不符合预期.*条款原文逐字"):
        records_from_dicts([{"层级": "省"}])


def test_top_level_not_list_raises(tmp_path: Path) -> None:
    """顶层结构不是数组时明确报错。"""
    bad = tmp_path / "bad.json"
    bad.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="顶层结构应为数组"):
        load_records(bad)


def test_data_path_comes_from_config(tmp_path: Path) -> None:
    """数据仓位置读 config.json（路径不硬编码）：改配置即可换数据源。"""
    shutil.copy(FIXTURE, tmp_path / "rules_sample.json")
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"data_repo": str(tmp_path), "rules_terms_file": "rules_sample.json"}),
        encoding="utf-8",
    )
    assert len(load_store_from_config(config_file).records) == 10
    with pytest.raises(FileNotFoundError, match="config.example.json"):
        load_config(tmp_path / "missing.json")


def test_curation_field_excludes_from_active() -> None:
    """「负面清单口径」非空注记＝curation 出局：全量保留、现行规则集剔除。"""
    base = {
        "条款原文逐字": "测试条款原文",
        "出处文号+条款号": "〔测试样例〕〔2026〕1号第一条",
        "条款号": "一",
        "承接列举": False,
        "层级": "省",
        "所在文件效力状态": "有效",
        "负面清单载体文件": False,
        "等价组建议": {"组号": "未分组", "组名": "", "组类型": ""},
        "溯源_文件序号": 1,
        "溯源_文号": "〔测试样例〕〔2026〕1号",
        "溯源_标题": "测试文件",
        "溯源_发布机关": "测试机关",
        "溯源_发布日期": "2026-01-01",
        "溯源_类别": "资金管理办法",
        "组裁定": "待人工",
    }
    curated = dict(base, 负面清单口径="排除（curation-测试：非负面清单范畴）")
    records = records_from_dicts([base, curated])
    assert records[0].curated_out is False and records[0].is_active is True
    assert records[1].curated_out is True and records[1].is_active is False
    store = RuleStore(records)
    assert len(store.records) == 2                      # 全量保留（可溯源）
    assert all(r.source_ref == base["出处文号+条款号"] for r in store.active_records())
