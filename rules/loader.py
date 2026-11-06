"""rules/loader.py — 从数据仓《负面清单_条款归并.json》加载规则库。

- 字段名逐字对应数据仓实际结构，映射表见 _FIELD_MAP（改字段须先对照数据仓）；
- 数据仓位置不硬编码：从代码仓根目录 config.json 读取（样例见 config.example.json）；
- 加载零报错口径：逐条结构化映射，字段缺失/类型错误即报错并定位到条目序号。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rules.schema import GROUP_NONE, RuleRecord, RuleStore

logger = logging.getLogger(__name__)

# 数据仓顶层字段名 → RuleRecord 构造参数名（逐字对应，勿凭猜测修改）
_FIELD_MAP: dict[str, str] = {
    "条款原文逐字": "clause_text",
    "出处文号+条款号": "source_ref",
    "条款号": "clause_no",
    "承接列举": "carries_enumeration",
    "层级": "level",
    "所在文件效力状态": "effect_status",
    "负面清单载体文件": "is_negative_list_file",
    "溯源_文件序号": "doc_index",
    "溯源_文号": "doc_no",
    "溯源_标题": "doc_title",
    "溯源_发布机关": "doc_issuer",
    "溯源_发布日期": "doc_date",
    "溯源_类别": "doc_category",
    "组裁定": "group_decision",
}


def _as_bool(value: Any, field: str) -> bool:
    """布尔字段解析：兼容 bool 与常见字符串写法。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "是", "1"):
            return True
        if lowered in ("false", "否", "0"):
            return False
    raise ValueError(f"字段「{field}」无法解析为布尔值：{value!r}")


def _expect(item: dict, field: str) -> Any:
    """取值并校验存在性（None 视为缺失），报错带字段名便于定位。"""
    if field not in item or item[field] is None:
        raise ValueError(f"缺少字段「{field}」")
    return item[field]


def records_from_dicts(items: list[dict]) -> list[RuleRecord]:
    """把数据仓原始字典列表转换为 RuleRecord 列表（逐条校验，零容忍错位）。"""
    records: list[RuleRecord] = []
    for index, item in enumerate(items):
        try:
            records.append(_record_from_dict(item))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"第 {index + 1} 条记录结构不符合预期：{exc}") from exc
    return records


def _record_from_dict(item: dict) -> RuleRecord:
    group = item.get("等价组建议") or {}
    return RuleRecord(
        clause_text=str(_expect(item, "条款原文逐字")),
        source_ref=str(_expect(item, "出处文号+条款号")),
        clause_no=str(_expect(item, "条款号")),
        carries_enumeration=_as_bool(_expect(item, "承接列举"), "承接列举"),
        level=str(_expect(item, "层级")),
        effect_status=str(_expect(item, "所在文件效力状态")),
        group_id=str(group.get("组号") or GROUP_NONE),
        group_name=str(group.get("组名") or ""),
        group_type=str(group.get("组类型") or ""),
        is_negative_list_file=_as_bool(_expect(item, "负面清单载体文件"), "负面清单载体文件"),
        doc_index=int(_expect(item, "溯源_文件序号")),
        doc_no=str(_expect(item, "溯源_文号")),
        doc_title=str(_expect(item, "溯源_标题")),
        doc_issuer=str(_expect(item, "溯源_发布机关")),
        doc_date=str(_expect(item, "溯源_发布日期")),
        doc_category=str(_expect(item, "溯源_类别")),
        group_decision=str(_expect(item, "组裁定")),
        # 「负面清单口径」为 curation 出局标注（非空即出局，2026-11-02 起新增，旧数据无此字段）；
        # 不进 _FIELD_MAP：它不是逐字改名，而是"非空注记→出局"语义
        curated_out=bool(str(item.get("负面清单口径") or "").strip()),
    )


def load_records(data_file: Path) -> list[RuleRecord]:
    """从条款归并 json 文件加载全部条款记录（顶层须为数组）。"""
    with open(data_file, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"顶层结构应为数组，实际为 {type(data).__name__}")
    return records_from_dicts(data)


def load_config(config_path: Path | None = None) -> dict:
    """读取配置（数据仓路径不硬编码）：缺省取代码仓根目录 config.json。"""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"缺少配置文件 {config_path}，请参照 config.example.json 创建 config.json"
        )
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def load_store_from_config(config_path: Path | None = None) -> RuleStore:
    """按 config.json 配置的数据仓路径加载规则库（加载统计见 RuleStore.stats）。"""
    config = load_config(config_path)
    data_root = Path(config["data_repo"])
    data_file = data_root / config["rules_terms_file"]
    logger.info("加载规则源文件：%s", data_file)
    return RuleStore(load_records(data_file))
