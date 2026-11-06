"""消融开关的运行时观测点（3.4 评测与消融 · PR1）。

职责：把 :class:`eval.config.AblationSwitches` 里的布尔位，落到"某段代码到底跑没跑"
这一层可观测事实上。做法是给每个可消融的 stage 套 :func:`gated` 装饰器：

* 开关开启 → 正常调用真实实现；
* 开关关闭 → **不调用**真实实现（消融语义就要求它缺席），并往当前
  :class:`SwitchTrace` 记一条 bypass 事件。

验收标准「开关关掉的路径可被测试观测到」即由 ``SwitchTrace`` 承担：测试可以断言
"实现函数一次都没被调用" + "trace 里出现了对应 switch 的 bypass 记录"。

开关来源用 :class:`contextvars.ContextVar` 传递，因此评测跑批可以在主线程里并发多个
消融组而互不串味；未进入任何上下文时按"全开"处理，避免影响线上主流程。
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TypeVar

from eval.config import AblationSwitches

__all__ = [
    "BypassEvent",
    "SwitchTrace",
    "ablation_context",
    "current_switches",
    "current_trace",
    "gated",
]

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class BypassEvent:
    """一次"因开关关闭而跳过"的事件记录。"""

    switch: str
    stage: str
    reason: str = "switch_disabled"


@dataclass
class SwitchTrace:
    """收集 bypass 事件，供测试断言与评测报告引用。

    :param events: 按发生顺序记录的 bypass 事件
    """

    events: list[BypassEvent] = field(default_factory=list)

    def record(self, switch: str, stage: str, reason: str = "switch_disabled") -> BypassEvent:
        """记录一条 bypass 事件并返回它。"""
        event = BypassEvent(switch=switch, stage=stage, reason=reason)
        self.events.append(event)
        logger.debug("消融开关 %s 关闭，跳过 %s", switch, stage)
        return event

    def was_bypassed(self, switch: str) -> bool:
        """指定开关是否留下过 bypass 记录。"""
        return any(event.switch == switch for event in self.events)

    def bypassed_switches(self) -> tuple[str, ...]:
        """去重后的被跳过开关名，保持首次出现顺序。"""
        seen: dict[str, None] = {}
        for event in self.events:
            seen.setdefault(event.switch, None)
        return tuple(seen)

    def stages_of(self, switch: str) -> tuple[str, ...]:
        """某开关下被跳过 stage 的去重列表。"""
        seen: dict[str, None] = {}
        for event in self.events:
            if event.switch == switch:
                seen.setdefault(event.stage, None)
        return tuple(seen)

    def clear(self) -> None:
        """清空事件，便于复用同一个 trace 跑下一轮。"""
        self.events.clear()

    def summary(self) -> dict[str, Any]:
        """结构化为可写入评测报告的摘要。"""
        return {
            "bypassed_switches": list(self.bypassed_switches()),
            "event_count": len(self.events),
            "stages": {
                switch: list(self.stages_of(switch)) for switch in self.bypassed_switches()
            },
        }


_SWITCHES: ContextVar[AblationSwitches | None] = ContextVar("yueceshao_switches", default=None)
_TRACE: ContextVar[SwitchTrace | None] = ContextVar("yueceshao_trace", default=None)


def current_switches() -> AblationSwitches | None:
    """当前上下文的开关快照；不在评测上下文时返回 ``None``（语义＝全开）。"""
    return _SWITCHES.get()


def current_trace() -> SwitchTrace | None:
    """当前上下文的 trace；不在评测上下文时返回 ``None``。"""
    return _TRACE.get()


@contextmanager
def ablation_context(
    switches: AblationSwitches | None = None,
    trace: SwitchTrace | None = None,
) -> Iterator[SwitchTrace]:
    """进入一个消融上下文，yield 当前 trace 供调用方读取。

    :param switches: 开关快照，缺省为完整系统
    :param trace: 复用已有 trace，缺省新建
    """
    active_switches = switches or AblationSwitches.default()
    active_trace = trace or SwitchTrace()
    token_switches = _SWITCHES.set(active_switches)
    token_trace = _TRACE.set(active_trace)
    try:
        yield active_trace
    finally:
        _SWITCHES.reset(token_switches)
        _TRACE.reset(token_trace)


def gated(
    switch: str,
    *,
    fallback: Callable[..., Any] | None = None,
    stage: str | None = None,
) -> Callable[[F], F]:
    """装饰一个可消融的 stage 函数。

    开关开启时正常调用；关闭时跳过真实实现并记录 bypass。若提供 ``fallback``，
    则用它的返回值代替（例如检索被移除时退化为空召回）；否则返回 ``None``。

    :param switch: 开关名，必须是 :data:`eval.config.SWITCH_NAMES` 之一
    :param fallback: 开关关闭时的替代实现，签名与原函数一致
    :param stage: 报告中显示的阶段名，缺省用函数限定名
    """
    if switch not in AblationSwitches.default().to_dict():
        raise ValueError(f"未知的消融开关 {switch!r}")

    def decorator(func: F) -> F:
        stage_name = stage or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            switches = current_switches()
            if switches is not None and not switches.is_enabled(switch):
                trace = current_trace()
                if trace is not None:
                    trace.record(switch=switch, stage=stage_name)
                if fallback is not None:
                    return fallback(*args, **kwargs)
                return None
            return func(*args, **kwargs)

        wrapper.__ablation_switch__ = switch
        return wrapper  # type: ignore[return-value]

    return decorator
