"""eval 包：评测与消融（团队手册 3.4）。

子模块：
* :mod:`eval.config` —— 六组消融开关的集中配置（唯一来源，不得旁路硬编码）
* :mod:`eval.gating` —— 开关的运行时观测点（bypass 记录，供测试与报告使用）
* :mod:`eval.dataset` —— 案例标注表读取与校验（数据本身在团队数据仓，不入库）
* :mod:`eval.metrics` —— 三指标口径（匹配／召回／引用错误）与逐案例明细
* :mod:`eval.pipeline` —— 被评测管线的协议与占位实现
* :mod:`eval.runner` —— 跑批入口（``python -m eval.runner``）
"""

from __future__ import annotations

from eval.config import (
    SWITCH_DOCS,
    SWITCH_NAMES,
    AblationSwitches,
)
from eval.dataset import (
    DatasetError,
    EvalCase,
    RegistryError,
    load_cases,
    take,
)
from eval.gating import (
    BypassEvent,
    SwitchTrace,
    ablation_context,
    current_switches,
    current_trace,
    gated,
)
from eval.metrics import (
    DISCLAIMER_LINE_HINTS,
    FORBIDDEN_PHRASES,
    REPORT_DISCLAIMER,
    CaseOutcome,
    CasePrediction,
    EvalThresholds,
    MetricsSummary,
    evaluate_case,
    strip_disclaimer,
    summarize,
)
from eval.pipeline import NullPipeline, Pipeline

__all__ = [
    # config
    "SWITCH_DOCS",
    "SWITCH_NAMES",
    "AblationSwitches",
    # dataset
    "DatasetError",
    "RegistryError",
    "EvalCase",
    "load_cases",
    "take",
    # gating
    "BypassEvent",
    "SwitchTrace",
    "ablation_context",
    "current_switches",
    "current_trace",
    "gated",
    # metrics
    "DISCLAIMER_LINE_HINTS",
    "FORBIDDEN_PHRASES",
    "REPORT_DISCLAIMER",
    "CaseOutcome",
    "CasePrediction",
    "EvalThresholds",
    "MetricsSummary",
    "evaluate_case",
    "strip_disclaimer",
    "summarize",
    # pipeline
    "NullPipeline",
    "Pipeline",
]
