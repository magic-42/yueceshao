"""scripts/verify_index.py — 真实 Chroma 索引完整性验证（feat/real-retrieval 步骤 1 交付物）。

用途
----
索引 zip 解压后，先回答三个问题再谈接入：

1. **块数对不对**——collection ``yueceshao_clauses`` 的 count 是否等于随包说明的 18262；
2. **元数据全不全**——随机抽 10 块，检查引用渲染必备字段（doc_no / clause / validity /
   is_current）是否齐全；
3. **审计公告漏没漏**——用 ``rag.loader`` 的过滤口径（category == 审计公告即剔除）
   反推"应进索引"的序号集合，再与索引里实际出现的序号比对，任何不在集合内的块都算泄漏
   （红线④：审计公告是评测出卷集，进索引＝答案泄露）。

防静默空索引
------------
Chroma 在 ``persist_dir`` 指错时会**自动建一个空集合且不报错**，count 读成 0。
因此打开客户端前先断言目录里存在 ``chroma.sqlite3``，并明确要求集合存在——不能把
"路径配错"误判成"索引是空的"。

用法::

    python -m scripts.verify_index --config config.json
    python -m scripts.verify_index --config config.json --expected-count 18262 --sample 10

退出码：0=通过；1=不通过（块数不符／抽检缺字段／审计公告泄漏）。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.loader import AUDIT_CATEGORY, load_registry

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_EXPECTED_COUNT = 18262
"""kelier1 随包说明的块数（2026-11-02 索引，Qwen3-Embedding-0.6B，1024 维）。"""

REQUIRED_METADATA: tuple[str, ...] = ("doc_no", "clause", "validity", "is_current")
"""抽检必备字段。

任务口径写的是 ``clause_no``；但本索引（chunker.as_metadata）的字段名是 ``clause``
（如「第四条」），``clause_index`` 是阿拉伯序号——两者语义不同，故以 ``clause`` 为准。
"""

PAGE_SIZE = 5000
"""分页读取块元数据的批大小（18262 块约 4 页）。"""

SAMPLE_SEED = 20261106
"""抽检偏移的固定种子（可复现；与索引内容无关）。"""


# ================================================================ 数据结构


@dataclass(frozen=True)
class SampleResult:
    """单块抽检结论：chunk_id ＋ 缺失的必备字段。"""

    chunk_id: str
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """该块是否字段齐全。"""
        return not self.missing


@dataclass(frozen=True)
class IndexVerification:
    """索引验证的完整结论（供 CLI 打印与测试断言）。"""

    persist_dir: Path
    collection: str
    count: int
    expected_count: int | None
    samples: tuple[SampleResult, ...]
    doc_count: int
    audit_leaks: tuple[tuple[int, str, str], ...]
    warnings: tuple[str, ...] = ()

    @property
    def count_ok(self) -> bool:
        """块数是否符合预期（未给预期时不判）。"""
        return self.expected_count is None or self.count == self.expected_count

    @property
    def samples_ok(self) -> bool:
        """抽检块是否全部字段齐全。"""
        return all(sample.ok for sample in self.samples)

    @property
    def ok(self) -> bool:
        """三项验收全部通过。"""
        return self.count_ok and self.samples_ok and not self.audit_leaks

    def summary(self) -> str:
        """多行结论（CLI 输出与 PR 评论直接使用）。"""
        expected = "未指定" if self.expected_count is None else str(self.expected_count)
        mark = lambda flag: "✓" if flag else "✗"  # noqa: E731 - 仅本函数内使用的展示辅助
        lines = [
            f"索引目录：{self.persist_dir}",
            f"集合：{self.collection}",
            f"块数：{self.count}（期望 {expected}）{mark(self.count_ok)}",
            f"文档数（按序号去重）：{self.doc_count}",
            f"抽检 {len(self.samples)} 块元数据：{mark(self.samples_ok)}",
        ]
        for sample in self.samples:
            if not sample.ok:
                lines.append(f"  · {sample.chunk_id} 缺字段：{'、'.join(sample.missing)}")
        if self.audit_leaks:
            lines.append(f"审计公告泄漏：{len(self.audit_leaks)} 处 ✗")
            lines.extend(
                f"  · 序号 {serial}（{category}）{title}"
                for serial, category, title in self.audit_leaks
            )
        else:
            lines.append("审计公告泄漏：0（loader 过滤口径核对）✓")
        lines.extend(f"提示：{warning}" for warning in self.warnings)
        return "\n".join(lines)


# ================================================================ 纯函数


def check_required_fields(
    metadata: dict[str, Any],
    required: Sequence[str] = REQUIRED_METADATA,
) -> tuple[str, ...]:
    """返回该块缺失的必备字段（按 ``required`` 顺序）。"""
    return tuple(name for name in required if name not in metadata)


def find_audit_leaks(
    records: Iterable[tuple[int, str, str]],
    kept_serials: set[int],
) -> tuple[tuple[int, str, str], ...]:
    """找出索引里出现、但不应进索引的序号。

    两条判据任一命中都算泄漏：

    1. 块的 category 直接是 ``审计公告``（检索/入库两层的硬拦截应当已经挡掉）；
    2. 序号不在 loader 过滤口径的保留集合里——loader 只按 category 剔除审计公告，
       不在保留集合意味着该文属于被剔除的审计公告，或 registry 里根本查不到
       （数据快照未对齐时要人工确认，不能默认放行）。

    :param records: ``(序号, 类别, 标题)`` 三元组序列
    :param kept_serials: loader 口径下应进索引的序号集合
    :return: 泄漏项（去重保序）
    """
    leaks: dict[int, tuple[int, str, str]] = {}
    for serial, category, title in records:
        if category == AUDIT_CATEGORY or serial not in kept_serials:
            leaks.setdefault(serial, (serial, category, title))
    return tuple(leaks.values())


# ================================================================ 现场读取


def _as_int(value: Any) -> int | None:
    """元数据值转 int；bool/非数字返回 None（避免 True 被当成 1）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def scan_block_records(collection: Any, page_size: int = PAGE_SIZE) -> list[tuple[int, str, str]]:
    """逐页读取全部块的 ``(序号, 类别, 标题)``。

    :param collection: chromadb Collection
    :param page_size: 单页条数
    :return: 按读取顺序的元数据三元组列表
    """
    records: list[tuple[int, str, str]] = []
    offset = 0
    while True:
        page = collection.get(limit=page_size, offset=offset, include=["metadatas"])
        ids = page.get("ids") or []
        if not ids:
            return records
        for metadata in page.get("metadatas") or []:
            payload = dict(metadata or {})
            serial = _as_int(payload.get("serial"))
            records.append((
                -1 if serial is None else serial,
                str(payload.get("category") or ""),
                str(payload.get("title") or ""),
            ))
        offset += len(ids)


def _load_config(config_path: Path) -> dict[str, Any]:
    """读本机 config.json；不存在时抛错（验证工具没有默认兜底）。"""
    if not config_path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在：{config_path}（先复制 config.example.json 为 config.json）")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层应为对象：{config_path}")
    return payload


def _resolve(persist_dir: str, config_path: Path) -> Path:
    """persist_dir 相对路径按**配置文件所在目录**解析（不依赖当前工作目录）。"""
    path = Path(persist_dir)
    target = path if path.is_absolute() else config_path.parent / path
    return target.resolve()


def verify_index(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    expected_count: int | None = DEFAULT_EXPECTED_COUNT,
    sample: int = 10,
    registry_path: Path | str | None = None,
    collection: Any = None,
) -> IndexVerification:
    """验证真实索引（打开 chromadb、抽检、与 registry 对账）。

    :param config_path: 本机配置（``index.persist_dir``／``index.collection``／``data_repo``）
    :param expected_count: 期望块数；None 不检查
    :param sample: 抽检块数
    :param registry_path: 覆盖 registry 路径（默认 ``<data_repo>/<registry_path>``）
    :param collection: 注入集合对象（测试用）；None 时按配置打开真实 Chroma
    :return: 验证结论
    :raises FileNotFoundError: 索引目录／chroma.sqlite3 不存在
    :raises RuntimeError: 集合不存在
    """
    config_file = Path(config_path)
    config = _load_config(config_file)
    index_section = config.get("index") or {}
    collection_name = str(index_section.get("collection") or "yueceshao_clauses")
    persist_dir = _resolve(str(index_section.get("persist_dir") or ".chroma"), config_file)

    if collection is None:
        if not persist_dir.is_dir():
            raise FileNotFoundError(
                f"索引目录不存在：{persist_dir}（先把 .chroma/ 解压到 config 的 persist_dir）")
        if not (persist_dir / "chroma.sqlite3").is_file():
            raise FileNotFoundError(
                f"{persist_dir} 下没有 chroma.sqlite3：persist_dir 指错了。"
                "Chroma 会因此自动建空集合、count 读成 0，不要硬扛。")
        import chromadb  # 延迟导入：CI 不装 chromadb 也能 import 本模块

        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            collection = client.get_collection(collection_name)
        except Exception as error:  # noqa: BLE001 - 各版本异常类型不一致
            raise RuntimeError(
                f"集合 {collection_name} 打不开（{error}）。先确认索引已解压、集合名正确；"
                "若是版本不兼容，读 chroma.sqlite3 的 migrations 表定位 kelier1 的 chromadb 版本。"
            ) from error

    warnings: list[str] = []
    count = int(collection.count())
    if expected_count is not None and count != expected_count:
        warnings.append(
            f"块数 {count} ≠ {expected_count}：索引可能不是约定版本，"
            "或数据快照未对齐（以随包说明的三样版本信息为准）")

    samples: list[SampleResult] = []
    if count:
        rng = random.Random(SAMPLE_SEED)
        offset = rng.randrange(0, max(1, count - sample + 1))
        page = collection.get(limit=min(sample, count), offset=offset, include=["metadatas"])
        ids = page.get("ids") or []
        metadatas = page.get("metadatas") or []
        for chunk_id, metadata in zip(ids, metadatas, strict=False):
            samples.append(
                SampleResult(str(chunk_id), check_required_fields(dict(metadata or {}))))

    records = scan_block_records(collection)
    doc_count = len({serial for serial, _category, _title in records if serial >= 0})

    audit_leaks: tuple[tuple[int, str, str], ...] = ()
    if registry_path is None:
        data_repo = Path(str(config.get("data_repo") or ""))
        registry_file = data_repo / str(config.get("registry_path") or "00_工作区/registry.json")
    else:
        registry_file = Path(registry_path)
    if registry_file.is_file():
        kept = load_registry(registry_file)
        kept_serials = {doc.serial for doc in kept.documents}
        audit_leaks = find_audit_leaks(records, kept_serials)
        if kept.excluded_audit:
            logger.info(
                "loader 口径：登记 %d 条，剔除审计公告 %d 条", kept.total, kept.excluded_audit)
    else:
        warnings.append(f"未找到 registry（{registry_file}），审计公告核对跳过")

    result = IndexVerification(
        persist_dir=persist_dir,
        collection=collection_name,
        count=count,
        expected_count=expected_count,
        samples=tuple(samples),
        doc_count=doc_count,
        audit_leaks=audit_leaks,
        warnings=tuple(warnings),
    )
    logger.info("索引验证结论\n%s", result.summary())
    return result


# ================================================================ 命令行


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：``python -m scripts.verify_index``。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.verify_index",
        description="验证真实 Chroma 索引（块数／元数据抽检／审计公告泄漏）",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="本机配置文件")
    parser.add_argument(
        "--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT,
        help=f"期望块数，默认 {DEFAULT_EXPECTED_COUNT}；给 0 表示不检查",
    )
    parser.add_argument("--sample", type=int, default=10, help="抽检块数，默认 10")
    parser.add_argument("--registry", default=None, help="覆盖 registry.json 路径")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(message)s",
    )
    expected = args.expected_count if args.expected_count > 0 else None
    try:
        result = verify_index(
            config_path=args.config,
            expected_count=expected,
            sample=max(1, args.sample),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("索引验证失败：%s", error)
        return 1
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - 入口
    raise SystemExit(main())
