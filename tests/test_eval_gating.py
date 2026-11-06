"""消融开关运行时观测点的单元测试（3.4 · PR1 · feat/eval-config）。

验收标准「开关关掉的路径可被测试观测到」在这里落地：断言开关关闭时
1. 真实实现函数**一次都没被调用**（用调用计数器证明它确实缺席）；
2. ``SwitchTrace`` 里留下了对应 switch 的 bypass 记录。
"""

from __future__ import annotations

import pytest

from eval.config import AblationSwitches
from eval.gating import (
    SwitchTrace,
    ablation_context,
    current_switches,
    current_trace,
    gated,
)

CALLS: list[str] = []


@gated("rerank", stage="retrieve.rerank")
def fake_rerank(candidates: list[str]) -> list[str]:
    """模拟重排阶段：记录调用并原样返回。"""
    CALLS.append("rerank")
    return sorted(candidates)


@gated("hybrid_retrieval", fallback=lambda query: [], stage="retrieve.hybrid")
def fake_hybrid(query: str) -> list[str]:
    """模拟混合检索：开关关闭时退化为空召回。"""
    CALLS.append("hybrid")
    return [f"doc-for-{query}"]


@pytest.fixture(autouse=True)
def _clear_calls() -> None:
    """每个用例前清空调用记录。"""
    CALLS.clear()


def test_enabled_switch_calls_implementation() -> None:
    """开关全开：实现被调用，trace 无 bypass 记录。"""
    with ablation_context(AblationSwitches.default()) as trace:
        result = fake_rerank(["b", "a"])
    assert result == ["a", "b"]
    assert CALLS == ["rerank"]
    assert trace.events == []
    assert trace.was_bypassed("rerank") is False


def test_disabled_switch_skips_implementation_and_records_bypass() -> None:
    """开关关闭：实现一次都没跑，但 trace 里有 bypass 记录——这就是可观测性。"""
    switches = AblationSwitches.default().with_overrides(rerank=False)
    with ablation_context(switches) as trace:
        result = fake_rerank(["b", "a"])
    assert result is None
    assert CALLS == []  # 实现确实缺席
    assert trace.was_bypassed("rerank") is True
    assert trace.stages_of("rerank") == ("retrieve.rerank",)
    assert trace.bypassed_switches() == ("rerank",)


def test_fallback_used_when_switch_disabled() -> None:
    """提供 fallback 时用其返回值代替，且仍记录 bypass。"""
    switches = AblationSwitches.default().with_overrides(hybrid_retrieval=False)
    with ablation_context(switches) as trace:
        result = fake_hybrid("强村公司")
    assert result == []
    assert CALLS == []
    assert trace.was_bypassed("hybrid_retrieval") is True


def test_gating_scoped_to_context() -> None:
    """上下文之外按全开处理，不影响主流程默认行为。"""
    assert current_switches() is None
    assert current_trace() is None

    result = fake_rerank(["b", "a"])
    assert result == ["a", "b"]
    assert CALLS == ["rerank"]


def test_trace_summary_structure() -> None:
    """summary() 输出可直接写入评测报告。"""
    switches = AblationSwitches.default().with_overrides(rerank=False, hybrid_retrieval=False)
    with ablation_context(switches) as trace:
        fake_rerank(["x"])
        fake_hybrid("q")
    summary = trace.summary()
    assert summary["event_count"] == 2
    assert set(summary["bypassed_switches"]) == {"rerank", "hybrid_retrieval"}
    assert summary["stages"]["rerank"] == ["retrieve.rerank"]


def test_trace_can_be_reused_and_cleared() -> None:
    """trace 可跨上下文复用，clear() 能重置。"""
    shared = SwitchTrace()
    switches = AblationSwitches.default().with_overrides(rerank=False)
    with ablation_context(switches, trace=shared):
        fake_rerank(["a"])
    with ablation_context(switches, trace=shared):
        fake_rerank(["b"])
    assert len(shared.events) == 2
    shared.clear()
    assert shared.events == []


def test_gated_rejects_unknown_switch() -> None:
    """装饰不存在的开关名要当场报错，而不是悄悄失效。"""
    with pytest.raises(ValueError, match="未知的消融开关"):

        @gated("no_such_switch")
        def _boom() -> None:  # pragma: no cover - 只验证装饰期报错
            return None
