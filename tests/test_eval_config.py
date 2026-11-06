"""消融开关集中配置的单元测试（3.4 · PR1 · feat/eval-config）。

覆盖点：
1. 默认值为完整系统（六组全开）；
2. 六组开关名与文档说明一一对应，不允许漏项；
3. 非法开关名 / 非 bool 取值必须报错，不能静默忽略；
4. 各构造入口（mapping / json / env）行为一致；
5. ``with_overrides`` 不可变性、``disabled`` 与标签生成正确。
"""

from __future__ import annotations

import json

import pytest

from eval.config import SWITCH_DOCS, SWITCH_NAMES, AblationSwitches


def test_default_is_full_system() -> None:
    """默认构造即完整系统，六组全开。"""
    switches = AblationSwitches.default()
    assert switches.is_full_system()
    assert switches.disabled() == ()
    assert switches.ablated_label() == "full"
    for name in SWITCH_NAMES:
        assert switches.is_enabled(name) is True


def test_switch_names_and_docs_cover_each_other() -> None:
    """规范名清单与中文说明字典必须完全对应，防止加了开关忘了写说明。"""
    assert len(SWITCH_NAMES) == 6
    assert set(SWITCH_NAMES) == set(SWITCH_DOCS)
    # 手册 3.4 明确点名的六组：时序过滤 / 负面硬约束 / 重排 / 混合 / 抽取分解 / 校验器
    for expected in (
        "temporal_filter",
        "negative_hard_constraint",
        "rerank",
        "hybrid_retrieval",
        "query_decomposition",
        "citation_validator",
    ):
        assert expected in SWITCH_DOCS
        assert SWITCH_DOCS[expected].strip()


def test_unknown_switch_name_raises() -> None:
    """拼错的开关名必须抛错，避免消融配置静默失效。"""
    with pytest.raises(ValueError, match="未知的消融开关"):
        AblationSwitches.from_mapping({"temporal_filtr": False})
    with pytest.raises(ValueError, match="未知的消融开关"):
        AblationSwitches.default().is_enabled("nonexistent")


def test_non_bool_value_raises() -> None:
    """开关只能是 bool；传 0/1 这类"看似能跑"的值要当场报错。"""
    with pytest.raises(TypeError):
        AblationSwitches.from_mapping({"rerank": 0})
    with pytest.raises(TypeError):
        AblationSwitches(rerank="true")  # type: ignore[arg-type]


def test_from_mapping_only_overrides_given_keys() -> None:
    """只覆盖传入的键，其余保持默认 True。"""
    switches = AblationSwitches.from_mapping({"rerank": False})
    assert switches.is_enabled("rerank") is False
    assert switches.is_enabled("temporal_filter") is True
    assert switches.disabled() == ("rerank",)
    assert switches.ablated_label() == "w/o:rerank"


def test_from_json_file(tmp_path) -> None:
    """从 JSON 文件读取，支持只在文件里写需要覆盖的键。"""
    path = tmp_path / "ablation.json"
    path.write_text(
        json.dumps({"temporal_filter": False, "citation_validator": False}),
        encoding="utf-8",
    )
    switches = AblationSwitches.from_json_file(path)
    assert switches.disabled() == ("temporal_filter", "citation_validator")


def test_from_json_file_rejects_non_object(tmp_path) -> None:
    """JSON 顶层必须是对象，数组/标量直接报错。"""
    path = tmp_path / "bad.json"
    path.write_text("[true, false]", encoding="utf-8")
    with pytest.raises(TypeError):
        AblationSwitches.from_json_file(path)


def test_from_env_parses_truthy_and_falsy() -> None:
    """环境变量支持 1/0、true/false 等写法，非法取值报错。"""
    switches = AblationSwitches.from_env(
        environ={
            "YUECESHAO_ABLATION_RERANK": "0",
            "YUECESHAO_ABLATION_HYBRID_RETRIEVAL": "false",
            "YUECESHAO_ABLATION_TEMPORAL_FILTER": "1",
        }
    )
    assert switches.disabled() == ("rerank", "hybrid_retrieval")

    with pytest.raises(ValueError, match="取值非法"):
        AblationSwitches.from_env(environ={"YUECESHAO_ABLATION_RERANK": "maybe"})


def test_from_env_rejects_misspelled_switch_name() -> None:
    """带前缀但后缀拼错的环境变量必须抛错，不能被静默忽略。

    这是最容易踩的坑：``RERANKK`` 多打一个 K，如果只遍历已知开关名，这个变量会被
    直接跳过——使用者以为 rerank 已关闭，实际它仍开着，消融结论整个失真。
    """
    with pytest.raises(ValueError, match="无法识别"):
        AblationSwitches.from_env(environ={"YUECESHAO_ABLATION_RERANKK": "0"})

    # 多个拼错时一次全报出来，不让人一个个试；混入的合法开关不得被误报
    with pytest.raises(ValueError) as excinfo:
        AblationSwitches.from_env(
            environ={
                "YUECESHAO_ABLATION_RERANKK": "0",
                "YUECESHAO_ABLATION_HYBRID_RETRIEVALL": "0",
                "YUECESHAO_ABLATION_TEMPORAL_FILTER": "0",  # 合法，不该进报错清单
            }
        )
    message = str(excinfo.value)
    reported = message.split("无法识别：", 1)[1].split("；", 1)[0]
    assert "RERANKK" in reported
    assert "HYBRID_RETRIEVALL" in reported
    assert "TEMPORAL_FILTER" not in reported

    # 前缀不匹配的普通环境变量不能触发误报（例如别的项目变量）
    switches = AblationSwitches.from_env(
        environ={"OTHER_ABLATION_RERANKK": "0", "YUECESHAO_ABLATION_RERANK": "0"}
    )
    assert switches.disabled() == ("rerank",)


def test_with_overrides_is_immutable_and_ordered() -> None:
    """派生新快照不改动原对象，disabled() 按规范顺序输出。"""
    base = AblationSwitches.default()
    derived = base.with_overrides(temporal_filter=False, negative_hard_constraint=False)
    assert base.is_full_system()  # 原对象不受影响
    assert derived.disabled() == ("temporal_filter", "negative_hard_constraint")
    assert derived.ablated_label() == "w/o:temporal_filter+negative_hard_constraint"


def test_all_disabled_and_to_dict() -> None:
    """全关快照与 to_dict 的键顺序与规范名一致。"""
    switches = AblationSwitches.all_disabled()
    assert len(switches.disabled()) == 6
    data = switches.to_dict()
    assert tuple(data.keys()) == SWITCH_NAMES
    assert all(value is False for value in data.values())
