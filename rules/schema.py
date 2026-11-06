"""rules/schema.py — 负面清单规则数据结构定义与规则库容器（RuleStore）。

职责：
1. RuleRecord：数据仓《负面清单_条款归并.json》单条记录的结构化映射；
2. RuleStore：全量规则库容器，提供效力过滤、等价组展开、失效条款替代展示、
   展示优先级排序等查询原语，供匹配引擎（feat/rules-engine）使用。

红线遵守：本模块与测试均不含任何真实政策语料原文（语料不入代码仓）。
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---- 效力状态常量（取值与数据仓字段「所在文件效力状态」一致）----
EFFECT_VALID = "有效"           # 现行有效
EFFECT_PENDING = "待人工核验"    # 效力待核验，按保留处理
EFFECT_INVALIDATED = "已失效"    # 已失效，从现行规则集剔除

# ---- 展示优先级：层级更具体优先（县 > 市 > 省 > 国家），与开发手册 §3.2 口径一致 ----
LEVEL_RANK: dict[str, int] = {"县": 4, "县（含乡镇）": 4, "市": 3, "省": 2, "国家": 1}

# 效力状态 → 展示排序权重（现行有效优先）
_EFFECT_RANK: dict[str, int] = {EFFECT_VALID: 2, EFFECT_PENDING: 1, EFFECT_INVALIDATED: 0}

# 未分组标记（等价组建议.组号 为「未分组」时的归一值）
GROUP_NONE = "未分组"

# 组类型常量（取值与数据仓字段「等价组建议.组类型」一致）
GROUP_TYPE_STRUCTURE = "结构标注"  # 总括句等结构性标注：不参与匹配，仅作溯源


@dataclass(frozen=True, slots=True)
class RuleRecord:
    """负面清单条款归并记录（字段名与数据仓原始记录逐字对应，映射表见 loader._FIELD_MAP）。"""

    clause_text: str             # 条款原文逐字
    source_ref: str              # 出处文号+条款号
    clause_no: str               # 条款号（中文数字，或「未分条（原文无条款号）」）
    carries_enumeration: bool    # 承接列举（列举项是否承接上一行条款）
    level: str                   # 层级：国家/省/市/县/县（含乡镇）/其他（兵团）
    effect_status: str           # 所在文件效力状态：有效/待人工核验/已失效
    group_id: str                # 等价组建议.组号（未分组归一为 GROUP_NONE）
    group_name: str              # 等价组建议.组名
    group_type: str              # 等价组建议.组类型
    is_negative_list_file: bool  # 负面清单载体文件
    doc_index: int               # 溯源_文件序号
    doc_no: str                  # 溯源_文号
    doc_title: str               # 溯源_标题
    doc_issuer: str              # 溯源_发布机关
    doc_date: str                # 溯源_发布日期
    doc_category: str            # 溯源_类别
    group_decision: str          # 组裁定（接受/待人工）
    curated_out: bool = False    # 负面清单口径出局（curation 标注非空即出局；
                                 # 2026-11-02 国市监21条先例）

    @property
    def is_effective(self) -> bool:
        """是否现行有效（效力状态=有效）。"""
        return self.effect_status == EFFECT_VALID

    @property
    def is_invalidated(self) -> bool:
        """是否已失效（效力状态=已失效）。"""
        return self.effect_status == EFFECT_INVALIDATED

    @property
    def is_active(self) -> bool:
        """是否保留在现行规则集（剔除已失效与 curation 出局；待人工核验保留）。"""
        return not self.is_invalidated and not self.curated_out


def display_priority_key(record: RuleRecord) -> tuple[int, int, int]:
    """展示排序键（升序）：现行有效优先，同效力时层级更具体优先，再按文件序号稳定排序。"""
    return (
        -_EFFECT_RANK.get(record.effect_status, 0),
        -LEVEL_RANK.get(record.level, 0),
        record.doc_index,
    )


class RuleStore:
    """负面清单规则库：全量记录 + 等价组索引 + 查询原语。

    术语：全量记录＝数据仓全部条款（含已失效）；现行规则集（active）＝剔除已失效
    （含「待人工核验」）；等价组＝等价组建议.组号相同的条款（G01-G11 为省市县三级同构链）。
    """

    def __init__(self, records: list[RuleRecord]) -> None:
        """构建规则库；records 为全部条款记录（含已失效）。"""
        self.records: tuple[RuleRecord, ...] = tuple(records)
        self.group_index: dict[str, tuple[RuleRecord, ...]] = self._build_group_index()
        logger.info("规则库构建完成：全量 %d 条，等价组 %d 个",
                    len(self.records), len(self.group_index))

    def _build_group_index(self) -> dict[str, tuple[RuleRecord, ...]]:
        """按组号建索引，组内成员按展示优先级排序；未分组条款不进索引。"""
        grouped: dict[str, list[RuleRecord]] = {}
        for record in self.records:
            if record.group_id != GROUP_NONE:
                grouped.setdefault(record.group_id, []).append(record)
        return {
            gid: tuple(sorted(members, key=display_priority_key))
            for gid, members in grouped.items()
        }

    def group_members(self, group_id: str) -> list[RuleRecord]:
        """同组全部条款，按展示优先级排序（现行有效优先、层级更具体优先）。"""
        return list(self.group_index.get(group_id, ()))

    def active_records(self) -> list[RuleRecord]:
        """现行规则集：剔除已失效条款（含「有效」与「待人工核验」）。"""
        return [r for r in self.records if r.is_active]

    def valid_records(self) -> list[RuleRecord]:
        """严格现行有效口径：效力状态=有效的条款。"""
        return [r for r in self.records if r.is_effective]

    def expand_equivalents(self, record: RuleRecord) -> list[RuleRecord]:
        """等价组展开：命中一条时给出同组其他层级条款（按展示优先级排序）。

        传入的 record 应来自本规则库（按其对象身份剔除自身）。
        """
        return [r for r in self.group_members(record.group_id) if r is not record]

    def display_substitutes(self, record: RuleRecord) -> list[RuleRecord]:
        """失效条款的替代展示：组内现行有效的同构条款（按展示优先级排序）。

        口径：优先「有效」；组内无「有效」时退回「待人工核验」；组内全部失效
        （如 G10）时返回空列表，由调用方决定是否保留原条款展示。
        """
        if not record.is_invalidated:
            return [record]
        return [r for r in self.group_members(record.group_id) if r.is_active]

    def stats(self) -> dict:
        """加载统计（供交付报告引用）：总量、效力分布、层级分布、分组情况。"""
        return {
            "total": len(self.records),
            "by_effect_status": dict(Counter(r.effect_status for r in self.records)),
            "active_total": len(self.active_records()),
            "valid_total": len(self.valid_records()),
            "by_level": dict(Counter(r.level for r in self.records)),
            "group_count": len(self.group_index),
            "grouped_records": sum(len(m) for m in self.group_index.values()),
        }
