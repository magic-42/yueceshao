"""粤策哨 · 数据仓 registry 加载器。

职责
----
把数据仓（``yueceshao-data``）里的 ``00_工作区/registry.json`` 读成标准化的
:class:`DocRecord` 列表，并按开发手册的贯穿红线过滤掉**审计公告**类文档。

为什么必须过滤审计公告
----------------------
审计公告是评测的出题集（回溯预警测试的考卷）。它一旦进入检索索引，
评测时就等于答案泄露；因此过滤发生在加载层，而不是等到索引层再补救。

设计约定
--------
- 只读：本模块不写任何文件、不改数据仓；
- 字段映射以数据仓的中文键为准（数据仓由人工维护，键名不在代码侧改动）；
- 路径一律用 :mod:`pathlib`，不硬编码分隔符，兼容 Windows；
- 判定审计公告只看 ``类别`` 字段，不猜文件名或路径。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 数据仓中审计公告的类别取值（进检索索引＝评测答案泄露，必须排除）
AUDIT_CATEGORY = "审计公告"

#: 效力状态：现行有效
VALIDITY_CURRENT = "有效"
#: 效力状态：已失效
VALIDITY_EXPIRED = "已失效"
#: 效力状态：待人工核验（不得静默当作有效，交由上层决定如何处理）
VALIDITY_PENDING = "待人工核验"

#: 真实文号的形态：``〔年份〕序号号``。数据仓对「没有文号」的文件不是留空，
#: 而是写成 ``无（原因）`` 这类占位文本，若直接当作文号渲染就会产出假文号。
_DOC_NO_PATTERN = re.compile(r"〔\s*\d{4}\s*〕\s*\d+\s*号")


class RegistryFormatError(ValueError):
    """registry.json 的结构不符合预期（顶层不是数组、元素不是对象等）。"""


@dataclass(frozen=True, slots=True)
class DocRecord:
    """一份政策文件的标准化登记项。

    字段与数据仓 ``registry.json`` 的中文键一一对应，属性名用英文。
    原始记录整条保留在 :attr:`raw`，供后续模块取用未显式映射的字段。
    """

    serial: int
    """序号：数据仓内唯一编号，也是清洗文本文件名的序号前缀。"""

    doc_no: str
    """文号，如 ``粤府〔2023〕34号``。空串表示该文件本身无文号（如部分网页公告）。"""

    title: str
    """文件标题。"""

    category: str
    """类别：资金管理办法 / 申报指南 / 负面清单 / 审计公告。"""

    validity: str
    """效力状态：有效 / 已失效 / 待人工核验 / 不适用-审计公告 / 失效宣布文件。"""

    issuer: str
    """发布机关。"""

    publisher: str
    """登载机关（转载该文的政府网站主办单位）。"""

    publish_date: str
    """发布日期，形如 ``2023-05-19``；缺失时为空串。"""

    source_url: str
    """原文 URL。"""

    clean_text_path: str
    """清洗文本相对路径，形如 ``03_清洗文本/02_粤府〔2023〕34号.txt``。"""

    clean_text_chars: int
    """清洗文本字符数（数据仓登记值，非本模块统计）。"""

    snapshot_path: str
    """网页快照相对路径。"""

    attachment_path: str
    """原始附件相对路径（可能是 doc/docx/pdf/xls 等）。"""

    valid_until: str
    """有效期至；空串表示未登记截止日。"""

    superseded_by: str
    """被谁废止（替代它的文件）。"""

    supersedes: str
    """废止了谁（它替代的文件）。"""

    validity_note: str
    """效力状态说明（人工核验结论的文字依据）。"""

    verify_status: str
    """核验状态：已人工核验 / 已核验（台账）。"""

    remark: str
    """备注。"""

    fetched_date: str
    """下载日期。"""

    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)
    """原始登记记录（中文键），保留全部字段供后续模块取用。"""

    # ------------------------------------------------------------------ 派生属性

    @property
    def is_audit(self) -> bool:
        """是否为审计公告（评测题源，禁止进检索索引）。"""
        return self.category == AUDIT_CATEGORY

    @property
    def is_current(self) -> bool:
        """是否现行有效。注意：``待人工核验`` 不算有效。"""
        return self.validity == VALIDITY_CURRENT

    @property
    def has_doc_no(self) -> bool:
        """文号字段里是否载有**真实**文号。

        数据仓对没有发文字号的文件（如审计结果公告、部分网页转载）写的是
        ``无（审计结果公告通常无发文字号）`` 这类占位说明，而不是空串。
        渲染层引用文号前必须先过这一关，否则会把占位说明印成文号。
        """
        return bool(_DOC_NO_PATTERN.search(self.doc_no))

    def resolve_clean_text(self, data_repo: Path | str) -> Path:
        """把清洗文本的相对路径解析成数据仓内的绝对路径。

        :param data_repo: 数据仓根目录
        :return: 清洗文本的绝对路径（不保证文件存在，调用方需自行校验）
        """
        return Path(data_repo) / self.clean_text_path


@dataclass(frozen=True, slots=True)
class LoadResult:
    """一次加载的结果与统计。"""

    documents: tuple[DocRecord, ...]
    """过滤后保留的文档（默认已剔除审计公告）。"""

    total: int
    """registry.json 中的登记总数。"""

    excluded_audit: int
    """因属于审计公告而被剔除的条数。"""

    @property
    def kept(self) -> int:
        """保留条数。"""
        return len(self.documents)


def _as_text(payload: Mapping[str, Any], key: str) -> str:
    """取字符串字段：缺失、None、非字符串一律退化为去空白的字符串。"""
    value = payload.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _as_int(payload: Mapping[str, Any], key: str) -> int:
    """取整数字段：无法解析时返回 0，不抛异常（数据仓存在人工录入误差）。"""
    value = payload.get(key)
    if isinstance(value, bool):  # bool 是 int 子类，先排除
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        logger.warning("字段 %s 无法解析为整数：%r，按 0 处理", key, value)
        return 0


def build_doc_record(payload: Mapping[str, Any]) -> DocRecord:
    """把一条 registry 原始记录转成 :class:`DocRecord`。

    :param payload: registry.json 数组中的一个对象
    :raises RegistryFormatError: 传入的不是映射
    """
    if not isinstance(payload, Mapping):
        raise RegistryFormatError(f"registry 条目应为对象，实际为 {type(payload).__name__}")

    return DocRecord(
        serial=_as_int(payload, "序号"),
        doc_no=_as_text(payload, "文号"),
        title=_as_text(payload, "标题"),
        category=_as_text(payload, "类别"),
        validity=_as_text(payload, "效力状态"),
        issuer=_as_text(payload, "发布机关"),
        publisher=_as_text(payload, "登载机关"),
        publish_date=_as_text(payload, "发布日期"),
        source_url=_as_text(payload, "原文URL"),
        clean_text_path=_as_text(payload, "清洗文本"),
        clean_text_chars=_as_int(payload, "清洗文本字符数"),
        snapshot_path=_as_text(payload, "网页快照"),
        attachment_path=_as_text(payload, "原始附件"),
        valid_until=_as_text(payload, "有效期至"),
        superseded_by=_as_text(payload, "被谁废止"),
        supersedes=_as_text(payload, "废止了谁"),
        validity_note=_as_text(payload, "效力状态说明"),
        verify_status=_as_text(payload, "核验状态"),
        remark=_as_text(payload, "备注"),
        fetched_date=_as_text(payload, "下载日期"),
        raw=payload,
    )


def iter_documents(
    payloads: list[Any],
    *,
    include_audit: bool = False,
) -> Iterator[DocRecord]:
    """逐条产出 :class:`DocRecord`，并按需过滤审计公告。

    :param payloads: registry.json 的顶层数组
    :param include_audit: 为 True 时保留审计公告（仅调试与测试用，禁止用于建索引）
    """
    for payload in payloads:
        record = build_doc_record(payload)
        if record.is_audit and not include_audit:
            logger.debug("过滤审计公告：序号 %s %s", record.serial, record.title)
            continue
        yield record


def read_registry(registry_path: Path | str) -> list[Any]:
    """读取 registry.json，校验顶层结构。

    :raises FileNotFoundError: 文件不存在
    :raises RegistryFormatError: 顶层不是数组
    """
    path = Path(registry_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 registry.json：{path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise RegistryFormatError(
            f"registry.json 顶层应为数组，实际为 {type(data).__name__}：{path}"
        )
    return data


def load_registry(
    registry_path: Path | str,
    *,
    include_audit: bool = False,
) -> LoadResult:
    """加载 registry.json 并返回标准化结果。

    :param registry_path: ``<数据仓>/00_工作区/registry.json``
    :param include_audit: 是否保留审计公告（默认 False，遵循红线）
    :return: :class:`LoadResult`，含文档元组与统计
    """
    payloads = read_registry(registry_path)
    documents = tuple(iter_documents(payloads, include_audit=include_audit))
    excluded = (
        sum(1 for item in payloads if _as_text(item, "类别") == AUDIT_CATEGORY)
        if not include_audit else 0)

    result = LoadResult(documents=documents, total=len(payloads), excluded_audit=excluded)
    logger.info(
        "registry 加载完成：登记 %d 条，剔除审计公告 %d 条，保留 %d 条",
        result.total,
        result.excluded_audit,
        result.kept,
    )
    return result
