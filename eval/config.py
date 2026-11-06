"""消融开关集中配置（3.4 评测与消融 · PR1）。

README「编码规范」要求：**全部消融开关集中在 eval/ 的配置布尔位，不得散落硬编码**。
本模块是这些开关的唯一来源——rag / rules / gen 等业务模块判断"某模块是否启用"时，
只允许读取本模块产出的 :class:`AblationSwitches`，不得自行定义常量、默认值或
`if os.getenv(...)` 之类的旁路判断。

开关语义一律为「True＝启用该模块（默认，完整系统），False＝移除该模块（消融组）」。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

__all__ = [
    "SWITCH_NAMES",
    "SWITCH_DOCS",
    "AblationSwitches",
]

#: 六组消融开关的规范名（顺序即报告与配置文件的输出顺序）。
#: 名称一经发布不得更改——评测报告要按名字做跨版本对比。
SWITCH_NAMES: tuple[str, ...] = (
    "temporal_filter",  # 时序版本过滤
    "negative_hard_constraint",  # 负面清单硬约束注入
    "rerank",  # Cross-Encoder 重排
    "hybrid_retrieval",  # BM25 + 稠密混合检索（关闭＝退化为仅稠密）
    "query_decomposition",  # 查询结构化抽取 / 术语归一化的分解步骤
    "citation_validator",  # 引用确定性校验器
)

#: 每个开关的中文说明，供报告与文档直接引用，避免同一概念在不同文件里叫法不一致。
SWITCH_DOCS: dict[str, str] = {
    "temporal_filter": "时序版本过滤：按查询时间剔除已废止条款",
    "negative_hard_constraint": "负面清单硬约束注入：命中即中断生成并给替代建议",
    "rerank": "Cross-Encoder 重排：对召回结果做精排",
    "hybrid_retrieval": "混合检索：BM25 与向量双路召回（关闭后仅走向量）",
    "query_decomposition": "查询分解：结构化抽取与术语归一化",
    "citation_validator": "引用校验器：标记存在性 / 上下文包含性 / 姿态禁语兜底",
}


def _check_switch_name(name: str) -> str:
    """校验开关名合法，非法则抛 ``ValueError``（拼写错误必须暴露，不能静默忽略）。"""
    if name not in SWITCH_DOCS:
        raise ValueError(
            f"未知的消融开关 {name!r}；合法取值：{', '.join(SWITCH_NAMES)}"
        )
    return name


@dataclass(frozen=True)
class AblationSwitches:
    """六组消融开关的不可变快照。

    默认全部为 ``True``，即完整系统。任何一组置为 ``False`` 即代表该模块被移除，
    对应调用路径应当被 :func:`eval.gating.gated` 跳过并留下可观测的 bypass 记录。

    用法::

        switches = AblationSwitches.default()
        ablated = switches.with_overrides(temporal_filter=False)
        if ablated.is_enabled("temporal_filter"):
            ...
    """

    temporal_filter: bool = True
    negative_hard_constraint: bool = True
    rerank: bool = True
    hybrid_retrieval: bool = True
    query_decomposition: bool = True
    citation_validator: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"消融开关 {field.name} 必须是 bool，收到 {type(value).__name__}: {value!r}"
                )

    # ---- 构造 -------------------------------------------------------------

    @classmethod
    def default(cls) -> AblationSwitches:
        """返回完整系统（六组开关全开）。"""
        return cls()

    @classmethod
    def all_disabled(cls) -> AblationSwitches:
        """返回全部移除的极端配置，便于快速验证 bypass 链路。"""
        return cls(**{name: False for name in SWITCH_NAMES})

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AblationSwitches:
        """从"部分覆盖"映射构造；未出现的开关保持默认 ``True``。

        未知键直接抛错——评测配置写错开关名会让消融结论失真，属于必须暴露的错误。
        """
        overrides: dict[str, bool] = {}
        for key, value in values.items():
            _check_switch_name(key)
            if not isinstance(value, bool):
                raise TypeError(f"消融开关 {key} 必须是 bool，收到 {value!r}")
            overrides[key] = value
        return cls(**overrides)

    @classmethod
    def from_json_file(cls, path: str | Path) -> AblationSwitches:
        """从 JSON 文件读取开关；文件内只需写需要覆盖的键。"""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError(f"{path} 顶层必须是 JSON 对象，收到 {type(raw).__name__}")
        return cls.from_mapping(raw)

    @classmethod
    def from_env(
        cls, prefix: str = "YUECESHAO_ABLATION_", environ: Mapping[str, str] | None = None
    ) -> AblationSwitches:
        """从环境变量读取开关，变量名形如 ``YUECESHAO_ABLATION_RERANK=0``。

        仅识别 ``1/true/yes/on`` 与 ``0/false/no/off``，其余取值视为配置错误。

        带前缀但后缀不是合法开关名的变量同样抛错，例如手滑写成
        ``YUECESHAO_ABLATION_RERANKK``。只遍历 :data:`SWITCH_NAMES` 会把这类拼写
        错误静默吞掉：使用者以为关掉了 rerank，实际它仍开着，消融组的结论直接失真。
        这与 :meth:`from_mapping` / :meth:`from_json_file` 的严格性保持一致。
        """
        env = os.environ if environ is None else environ
        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}

        unknown_env_names = sorted(
            key
            for key in env
            if key.upper().startswith(prefix.upper())
            and key[len(prefix):].lower() not in SWITCH_DOCS
        )
        if unknown_env_names:
            raise ValueError(
                "环境变量里的消融开关名无法识别："
                f"{', '.join(unknown_env_names)}；"
                "合法取名："
                f"{', '.join(prefix + name.upper() for name in SWITCH_NAMES)}"
            )

        overrides: dict[str, bool] = {}
        for name in SWITCH_NAMES:
            raw = env.get(f"{prefix}{name.upper()}")
            if raw is None:
                continue
            token = raw.strip().lower()
            if token in truthy:
                overrides[name] = True
            elif token in falsy:
                overrides[name] = False
            else:
                raise ValueError(
                    f"环境变量 {prefix}{name.upper()} 取值非法：{raw!r}（应为 1/0、true/false 等）"
                )
        return cls(**overrides)

    # ---- 查询与派生 -------------------------------------------------------

    def is_enabled(self, name: str) -> bool:
        """查询单个开关；未知名字抛 ``ValueError``。"""
        _check_switch_name(name)
        return bool(getattr(self, name))

    def disabled(self) -> tuple[str, ...]:
        """返回被关闭的开关名（按规范顺序），即本次消融的"移除项"。"""
        return tuple(name for name in SWITCH_NAMES if not getattr(self, name))

    def is_full_system(self) -> bool:
        """六组全开则为完整系统（基线组）。"""
        return not self.disabled()

    def with_overrides(self, **overrides: bool) -> AblationSwitches:
        """返回覆盖后的新快照（原对象不变，dataclass 为 frozen）。"""
        for key in overrides:
            _check_switch_name(key)
        return replace(self, **overrides)

    # ---- 序列化 -----------------------------------------------------------

    def to_dict(self) -> dict[str, bool]:
        """按规范顺序输出为普通 dict，便于写入评测报告与 JSON。"""
        data = asdict(self)
        return {name: bool(data[name]) for name in SWITCH_NAMES}

    def ablated_label(self) -> str:
        """生成简短标签：完整系统为 ``full``，否则形如 ``w/o:rerank+temporal_filter``。"""
        disabled = self.disabled()
        return "full" if not disabled else "w/o:" + "+".join(disabled)
