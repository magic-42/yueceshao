"""粤策哨 · 结构感知条款分块器。

职责
----
把清洗文本切成**可溯源到条款级**的检索单元（chunk）。实现开发手册 A1/A2：

- A1 结构感知条款分块：条＝引用单元；超长条（>300 字或 >5 款）切款级子块；
- A2 层级头注入（标题＋文号＋章节路径），引用字段一律以 :class:`~rag.loader.DocRecord`
  的元数据为准，**不从正文里猜文号**。

为什么不能只按「第X条」切
------------------------
对 718 篇可解析的清洗文本实测（序号 438 的文件名含 ``|``，Windows 落不了地，
未计入），结构分布并不单一：

===============  ======  ================================================
结构类型         篇数    说明
===============  ======  ================================================
意见方案型       396     用 ``一、`` + ``（一）`` 分层，没有「第X条」
条款型           179     法规/办法，用 ``第X章/第X节/第X条``
无结构碎片       143     短通知、废止公告，整篇即一个语义单元
===============  ======  ================================================

所以本模块按文档结构选解析器（:func:`detect_structure`），而不是把「第X条」硬套到
所有文件上——否则 75% 的语料会退化成整篇一块，条款级溯源就名存实亡。

真实语料里踩到的三个坑（均有对应用例）
------------------------------------
1. **OCR 把条号拆开**：``第 十九 条``、``第二十 一 条``。具体处数对统计口径很敏感
   （历史记录 107 处，换独立正则复核只剩 53 处），所以这里不写死数字，只保证
   **正则本身容忍序号内部空白**，归一化后再转阿拉伯数字。
2. **只有 136/732 篇带 ``#`` 头部元数据块**。因此元数据只从 DocRecord 取；文件头若存在，
   仅用于交叉校验，缺失不影响分块。
3. **附录会污染尾条**：如 ``653_财会〔2021〕37号`` 最后一条「第六十一条」若不做边界
   识别，会把 900 多行会计科目附录吞进同一个 chunk。附录单独成块并置 ``is_appendix``。
4. **目录会污染层级头**：规划类／意见方案型的目录条目（带 ``......`` 引导符）全文没有
   「第X条」，原实现因此剥不掉，随后被当成章标题拼进 header——实测序号 203 单块
   header 达 1697 字。现由 :func:`strip_toc` 按引导符剥离，与条款头无关。

设计约定
--------
- 只读：不写文件、不改数据仓；输入文本由调用方读入（便于测试注入）。
- 不做语义分块（手册 N1 明确不做）：边界只来自文本结构标记。
- 分块结果必须可复现：``chunk_id`` 由序号与顺序号确定，不含随机量。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from rag.loader import DocRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 可调参数默认值

#: 单条超过该字数即切款级子块（A1）
DEFAULT_MAX_CLAUSE_CHARS: Final[int] = 300
#: 单条款数超过该值即切款级子块（A1）
DEFAULT_MAX_KUAN: Final[int] = 5
#: 无结构文档（整篇一块）超过该字数时按段装箱
DEFAULT_FLAT_TARGET_CHARS: Final[int] = 500
#: 低于该字数的碎片并入相邻块
DEFAULT_MIN_CHUNK_CHARS: Final[int] = 30

# ---------------------------------------------------------------- 结构类型常量

STRUCTURE_CLAUSE: Final[str] = "clause"
"""条款型：``第X章/第X节/第X条``。"""

STRUCTURE_SECTION: Final[str] = "section"
"""意见方案型：``一、`` 分节，``（一）`` 为引用单元。"""

STRUCTURE_FLAT: Final[str] = "flat"
"""无结构：整篇或按段落装箱。"""

# ---------------------------------------------------------------- 正则

_CN_DIGIT: Final[dict[str, int]] = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT: Final[dict[str, int]] = {"十": 10, "百": 100, "千": 1000}

# 序号可用的单字：中文数字 + 阿拉伯数字（含全角，已在归一化里转半角）
_NUM: Final[str] = r"[零〇一二三四五六七八九十百两\d]"
# 序号主体：允许字间夹空格（OCR 产物「第二十 一 条」）
# 注意：条号后**不加**「必须有空白」的约束——部分文件写作「第一条为了规范…」，
# 条号与正文之间没有分隔符（653_财会〔2021〕37号 全篇如此）。
_NUM_BODY: Final[str] = rf"{_NUM}(?:[\s\u3000]*{_NUM}){{0,8}}?"

_CLAUSE_HEAD: Final[re.Pattern[str]] = re.compile(
    rf"^[\s\u3000]*第[\s\u3000]*({_NUM_BODY})[\s\u3000]*条"
)
_CHAPTER_HEAD: Final[re.Pattern[str]] = re.compile(
    rf"^[\s\u3000]*第[\s\u3000]*({_NUM_BODY})[\s\u3000]*章"
)
_SECTION_HEAD: Final[re.Pattern[str]] = re.compile(
    rf"^[\s\u3000]*第[\s\u3000]*({_NUM_BODY})[\s\u3000]*节"
)
# 款 / 项：（一）、(1)、1.、1、
_KUAN_HEAD: Final[re.Pattern[str]] = re.compile(
    r"^[\s\u3000]*[（(][\s\u3000]*([一二三四五六七八九十]{1,4}|\d{1,2})[\s\u3000]*[)）]"
)
_ITEM_HEAD: Final[re.Pattern[str]] = re.compile(r"^[\s\u3000]*(\d{1,2})[.、．](?!\d)")
# 意见方案型的一级标题：一、
_TOP_HEAD: Final[re.Pattern[str]] = re.compile(
    r"^[\s\u3000]*([一二三四五六七八九十]{1,3})[\s\u3000]*[、．.][\s\u3000]*\S"
)
# 目录标记：整行只有「目录」
_TOC_MARK: Final[re.Pattern[str]] = re.compile(r"^[\s\u3000]*目[\s\u3000]*录[\s\u3000]*$")
# OCR 会把「目录」拆成两行各一个字（实测序号 232 即如此）
_TOC_MARK_HEAD: Final[re.Pattern[str]] = re.compile(r"^[\s\u3000]*目[\s\u3000]*$")
_TOC_MARK_TAIL: Final[re.Pattern[str]] = re.compile(r"^[\s\u3000]*录[\s\u3000]*$")
# 目录条目行：靠**引导符**（......／······／……）识别，与文档是否有「第X条」无关
_TOC_LEADER: Final[re.Pattern[str]] = re.compile(r"[.．·…]{3,}")
# 目录区内的噪声行：罗马/阿拉伯页标、纯标点
_TOC_NOISE: Final[re.Pattern[str]] = re.compile(
    r"^[\s\u3000]*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹIVXivx]{1,4}"
    r"|[-－—]?[\s\u3000]*\d{1,4}[\s\u3000]*[-－—]?"
    r"|[.．·…、，。;；:：]+)[\s\u3000]*$"
)
# 附录/附件边界：整行是「附录」「附件」或「附件：xxx」
_APPENDIX_MARK: Final[re.Pattern[str]] = re.compile(
    r"^[\s\u3000]*(?:附[\s\u3000]*录|附[\s\u3000]*件)[\s\u3000]*(?:[:：][\s\S]*)?$"
)
# 头部元数据行（仅 136/732 篇有，不依赖它取元数据）
_HEADER_LINE: Final[re.Pattern[str]] = re.compile(r"^#\s")
_SEPARATOR_LINE: Final[re.Pattern[str]] = re.compile(r"^={3,}.*={3,}$")

#: 行尾出现这些标点即视为「该行已结束」，换行不再合并
_TERMINALS: Final[str] = "。；！？：…”』》】）)"

#: 标题行的长度上限。超过它的「标题行」实际上是标题被粘上了整段正文或表格数据
#: （实测序号 295 的表格行 891 字、序号 40 的正文 786 字）。这类行不当作章节标题，
#: 否则 header 会被撑到近千字——既把文号等真正的溯源信息挤到角落，也让头注入
#: 变成「往 embedding 输入里塞正文」。
MAX_TITLE_CHARS: Final[int] = 60


# ================================================================ 数字工具


def zh_number_to_int(text: str) -> int | None:
    """把中文/阿拉伯数字串转成整数，失败返回 ``None``。

    支持 ``二十一`` / ``一百零五`` / ``十二`` / ``29`` 等形态，并容忍字间空白
    （OCR 会把「第二十一条」拆成「第二十 一 条」）。

    :param text: 数字串
    :return: 整数；无法解析时 ``None``
    """
    cleaned = re.sub(r"[\s\u3000]", "", text)
    if not cleaned:
        return None
    if cleaned.isdigit():
        return int(cleaned)

    total = 0      # 已结转部分（十位及以上）
    section = 0    # 当前「百/十」段
    number = 0     # 当前待乘数字
    for char in cleaned:
        if char in _CN_DIGIT:
            number = _CN_DIGIT[char]
        elif char in _CN_UNIT:
            unit = _CN_UNIT[char]
            if number == 0 and unit == 10:  # 「十二」的「十」
                number = 1
            section += number * unit
            number = 0
        elif char == "万":
            section = (section + number) * 10000
            total += section
            section, number = 0, 0
        else:
            return None
    return total + section + number


# ================================================================ 数据类


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """分块参数。默认值取自开发手册 A1。"""

    max_clause_chars: int = DEFAULT_MAX_CLAUSE_CHARS
    """单条超过该字数即切款级子块。

    无结构（flat）文档的超长段落也用它当装箱上限（见 :func:`_split_unit`）。
    """

    max_kuan: int = DEFAULT_MAX_KUAN
    """单条款数超过该值即切款级子块。"""

    flat_target_chars: int = DEFAULT_FLAT_TARGET_CHARS
    """**附录／附件**单元按段装箱的目标字数。

    只作用于 ``is_appendix`` 的单元（如 653 号那份 900 多行的会计科目附录）。
    无结构文档走的是 :func:`_split_unit` 里的通用装箱，上限由 ``max_clause_chars``
    决定——两者不是同一个开关，改这里不会影响 flat 文档。
    """

    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS
    """低于该字数的碎片并入相邻块，避免「第三条 帮扶资金主要用于：」这类
    12 字的条款头独占一个检索名额。"""

    inject_header: bool = True
    """是否把层级头拼进 ``text``（A2）。关闭时 ``text`` 等于 ``body``，供消融对比。"""


@dataclass(frozen=True, slots=True)
class Chunk:
    """一个可检索、可溯源的文本单元。

    ``body`` 是条款原文（可能已切分），``header`` 是注入的层级路径，
    ``text`` 是二者拼接后真正送进 embedding 的字符串。
    """

    chunk_id: str
    """稳定标识：``{序号}-{全文顺序号}``，可复现。"""

    order: int
    """在所属文档中的顺序号（0 起）。"""

    serial: int
    """数据仓序号。"""

    doc_no: str
    """文号（可能是占位文本，渲染前须用 ``has_doc_no`` 判断）。"""

    title: str
    structure: str
    """文档结构类型：clause / section / flat。"""

    chapter: str | None
    """章或一级标题，如 ``第四章 涉农资金执行及管理``、``二、推进行业间涉农资金统筹``。"""

    section: str | None
    """节标题（仅条款型有此层级）。"""

    clause: str | None
    """引用单元标签，如 ``第二十条``、``（一）``；前言/附录为 ``None``。"""

    clause_index: int | None
    """引用单元的阿拉伯序号；用于校验条款连续性。"""

    part: int
    """该引用单元切分后的块序号（1 起，未切分时为 1）。"""

    parts_total: int
    """该引用单元共切成几块（未切分时为 1）。"""

    is_preamble: bool = False
    """是否为引用单元之前的引导段落。"""

    is_appendix: bool = False
    """是否为附录/附件内容。"""

    header: str = ""
    """注入的层级头（``inject_header=False`` 时仍会生成，便于排查）。"""

    body: str = ""
    """纯正文，不含层级头。"""

    text: str = ""
    """实际入库文本：``header + '\\n' + body``。"""

    char_count: int = 0
    """正文字数（去空白）。"""

    metadata: dict[str, object] = field(default_factory=dict)
    """随 chunk 落库的元数据（文号/效力/层级等），供 Chroma where 过滤。"""

    def as_metadata(self) -> dict[str, object]:
        """返回可写入向量库的扁平元数据。

        只放可做等值/范围过滤的标量字段；``None`` 统一转空串，避免不同后端对
        ``None`` 的处理不一致。
        """
        return {
            "chunk_id": self.chunk_id,
            "serial": self.serial,
            "doc_no": self.doc_no,
            "title": self.title,
            "structure": self.structure,
            "chapter": self.chapter or "",
            "section": self.section or "",
            "clause": self.clause or "",
            "clause_index": -1 if self.clause_index is None else self.clause_index,
            "part": self.part,
            "parts_total": self.parts_total,
            "is_preamble": self.is_preamble,
            "is_appendix": self.is_appendix,
            **self.metadata,
        }


# ================================================================ 文本预处理


def _norm_line(line: str) -> str:
    """行归一化：全角数字/字母转半角、压缩行内多余空白。"""
    trans = str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    )
    return line.translate(trans).strip().strip("\u3000")


def _is_marker(line: str) -> bool:
    """该行是否以结构标记开头。"""
    return bool(
        _CLAUSE_HEAD.match(line)
        or _CHAPTER_HEAD.match(line)
        or _SECTION_HEAD.match(line)
        or _KUAN_HEAD.match(line)
        or _ITEM_HEAD.match(line)
        or _TOP_HEAD.match(line)
        or _APPENDIX_MARK.match(line)
    )


def _is_title_line(line: str) -> bool:
    """该行是否为**纯标题行**（章、节、意见方案型一级标题）。

    与 :func:`_is_marker` 的区别：款／项（``（一）``、``1.``）和条款头后面常紧跟
    同段正文，把它们合并回来是对的；而章／节／一级标题是独立标题行，一旦被粘上
    后续整段正文甚至表格数据，后续的 ``_TOP_HEAD`` 就会把整行当成「章标题」拼进
    header（实测序号 40 达 842 字、295 达 946 字，全是正文和表格数字）。

    :param line: 已归一化的行
    :return: 是否纯标题行
    """
    return bool(
        _CHAPTER_HEAD.match(line) or _SECTION_HEAD.match(line) or _TOP_HEAD.match(line)
    )


def join_broken_lines(lines: Sequence[str]) -> list[str]:
    """合并 OCR 残留的硬换行。

    清洗文本已做过硬换行合并，但扫描件 OCR 仍有残留（如「以人民币」/「元”为金额
    单位」被切开）。规则极简：若上一行不以句末标点结尾、且本行不是结构标记、
    且上一行不是纯标题行，就把本行并回上一行。

    :param lines: 原始行
    :return: 合并后的行
    """
    merged: list[str] = []
    for raw in lines:
        line = _norm_line(raw)
        if not line:
            continue
        if (
            merged
            and not merged[-1].endswith(tuple(_TERMINALS))
            and not _is_marker(line)
            and not _is_title_line(merged[-1])
        ):
            merged[-1] = merged[-1] + line
        else:
            merged.append(line)
    return merged


def _toc_marker_span(lines: Sequence[str], index: int) -> int:
    """判断 ``index`` 处是否为「目录」标记，返回标记占用的行数（0 表示不是）。

    :param lines: 正文行
    :param index: 待判定的行下标
    :return: 1（整行「目录」）／2（OCR 拆成「目」「录」两行）／0（不是标记）
    """
    if _TOC_MARK.match(_norm_line(lines[index])):
        return 1
    if (
        index + 1 < len(lines)
        and _TOC_MARK_HEAD.match(_norm_line(lines[index]))
        and _TOC_MARK_TAIL.match(_norm_line(lines[index + 1]))
    ):
        return 2
    return 0


def strip_toc(lines: Sequence[str]) -> tuple[list[str], int]:
    """剔除目录区，返回 ``(剩余行, 剔除行数)``。

    **判据是引导符，不是「第X条」**。原实现只在找到条款头之后才敢剥目录，而意见
    方案型与规划类文档（全库 408/732 篇）通篇没有「第X条」，于是目录整段留在正文
    里，随后被 :data:`_TOP_HEAD` 逐行当成章标题拼进 header——实测序号 203 单块
    header 达 1697 字、几乎全是目录行。A2 的头注入本意是补上下文，塞进目录反而
    污染 embedding 输入，并让 header 长度远超正文。

    只剔除「目录标记 + 其后的目录条目（含引导符）／页标／空行」和散落的引导符行；
    正文本身一行不动，因此不存在「误删正文」的风险。

    :param lines: 正文行（已剥掉头部元数据块与分隔线）
    :return: ``(剩余行, 剔除行数)``；未检出目录时原样返回、计数为 0
    """
    kept: list[str] = []
    removed = 0
    index = 0
    total = len(lines)
    while index < total:
        span = _toc_marker_span(lines, index)
        if span:
            removed += span
            index += span
            while index < total:
                line = lines[index]
                if (
                    _TOC_LEADER.search(line)
                    or not line.strip()
                    or _TOC_NOISE.match(_norm_line(line))
                ):
                    removed += 1
                    index += 1
                else:
                    break
            continue
        if _TOC_LEADER.search(lines[index]):
            removed += 1
            index += 1
            continue
        kept.append(lines[index])
        index += 1
    return kept, removed


def split_body(text: str) -> tuple[list[str], int]:
    """剥掉头部元数据块、分隔线与目录，返回正文行及其在原文中的起始行号。

    目录先按引导符整段剔除（:func:`strip_toc`，对三种结构通用）；若目录条目本身
    没有引导符（少见），再退回原策略：以「首个第X条之前的最近章标题」为正文起点
    ——``653_财会〔2021〕37号`` 正是这种结构（目录里也列了全部章名，若不剥掉会
    凭空多出一整套空章）。

    :param text: 清洗文本全文
    :return: ``(正文行, 起始行号 1 起)``；目录被剔除后起始行号为近似值，仅用于
        日志排查，不参与任何分块决策
    """
    lines = text.splitlines()
    sep_index = next(
        (i for i, line in enumerate(lines) if _SEPARATOR_LINE.match(line.strip())), None
    )
    if sep_index is not None:
        start = sep_index + 1
    else:
        start = 0
        while start < len(lines) and (_HEADER_LINE.match(lines[start]) or not lines[start].strip()):
            start += 1

    body, toc_removed = strip_toc(lines[start:])
    if toc_removed:
        logger.debug("剥离目录行 %d 行（原文第 %d 行起）", toc_removed, start + 1)

    # 引导符式目录已在上面剥掉；这里再判一次是要处理「章名枚举式目录」（653 号）——
    # 它的条目没有引导符（剥不掉），只能靠「首个第X条之前最近的章标题」定位正文起点。
    # 注意条件里带上 toc_removed：目录标记被剥掉后 _TOC_MARK 不再命中，但回退仍需执行。
    if toc_removed == 0 and not any(_TOC_MARK.match(_norm_line(line)) for line in body[:60]):
        return body, start + 1

    first_clause = next(
        (i for i, line in enumerate(body) if _CLAUSE_HEAD.match(_norm_line(line))), None
    )
    if first_clause is None:
        logger.debug(
            "目录已剥离 %d 行，但全文无条款头（意见方案型／规划类），按原顺序保留", toc_removed
        )
        return body, start + 1

    for i in range(first_clause, -1, -1):
        if _CHAPTER_HEAD.match(_norm_line(body[i])):
            return body[i:], start + i + 1
    return body[first_clause:], start + first_clause + 1


def detect_structure(lines: Sequence[str]) -> str:
    """判定文档结构类型。

    判据是**行首**出现「第X条」：正文中引用条文（「依据第三条规定」）不会落在行首，
    因此行首命中即可认定该文以条款组织。哪怕只有一两条，也按条款型处理——把单条
    文件误当无结构会让条款号整块丢失，代价远大于误判一行为条款头。

    :param lines: 已合并断行的正文行
    :return: :data:`STRUCTURE_CLAUSE` / :data:`STRUCTURE_SECTION` / :data:`STRUCTURE_FLAT`
    """
    if any(_CLAUSE_HEAD.match(line) for line in lines):
        return STRUCTURE_CLAUSE
    if sum(1 for line in lines if _TOP_HEAD.match(line)) >= 2:
        return STRUCTURE_SECTION
    return STRUCTURE_FLAT


# ================================================================ 段落与装箱


#: 行内款标记（OCR 或排版偶尔把多个款并到同一行）
_INLINE_KUAN: Final[re.Pattern[str]] = re.compile(r"(?=[（(][一二三四五六七八九十]{1,4}[)）])")


def _explode_inline_kuan(line: str) -> list[str]:
    """把「…（一）…（二）…」这种一行多款拆成多行。

    只在拆出的每一片都以款标记开头时才拆，避免误伤正文里对「（一）」的引用。
    """
    pieces = [piece for piece in _INLINE_KUAN.split(line) if piece.strip()]
    if len(pieces) > 1 and all(_KUAN_HEAD.match(piece) for piece in pieces):
        return pieces
    return [line]


def _split_paragraphs(lines: Sequence[str]) -> list[list[str]]:
    """按款/项标记切成段落组，每个组是一段连续原文。"""
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for raw in lines:
        for line in _explode_inline_kuan(raw):
            if _KUAN_HEAD.match(line) or _ITEM_HEAD.match(line):
                if current:
                    paragraphs.append(current)
                current = [line]
            else:
                current.append(line)
    if current:
        paragraphs.append(current)
    return paragraphs


def _count(text: str) -> int:
    """统计去空白字数。"""
    return len(re.sub(r"\s", "", text))


def _split_sentences(paragraph: Sequence[str], limit: int) -> list[list[str]]:
    """把超长段落按句号硬切，每片不超过 ``limit`` 字。"""
    text = "".join(paragraph)
    pieces: list[str] = []
    buffer = ""
    for sentence in re.split(r"(?<=[。；！？])", text):
        if not sentence:
            continue
        if buffer and _count(buffer) + _count(sentence) > limit:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer += sentence
    if buffer:
        pieces.append(buffer)
    # 极端情况：单句本身超限，只能按字数硬切，保证不产生巨块
    result: list[list[str]] = []
    for piece in pieces:
        while _count(piece) > limit:
            result.append([piece[:limit]])
            piece = piece[limit:]
        if piece:
            result.append([piece])
    return result


def pack_paragraphs(paragraphs: Sequence[Sequence[str]], limit: int) -> list[list[str]]:
    """按段装箱：尽量少切，但每块不超过 ``limit`` 字。"""
    blocks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        para_len = _count("".join(paragraph))
        if para_len > limit:
            if current:
                blocks.append(current)
                current, current_len = [], 0
            blocks.extend(_split_sentences(paragraph, limit))
            continue
        if current and current_len + para_len > limit:
            blocks.append(current)
            current, current_len = [], 0
        current.extend(paragraph)
        current_len += para_len
    if current:
        blocks.append(current)
    return blocks


# ================================================================ 单元抽取


@dataclass(slots=True)
class _Unit:
    """解析中间态：一个引用单元（条 / （一） / 前言 / 附录）。"""

    clause: str | None
    clause_index: int | None
    chapter: str | None
    section: str | None
    lines: list[str] = field(default_factory=list)
    is_preamble: bool = False
    is_appendix: bool = False


def _find_appendix_start(lines: Sequence[str]) -> int | None:
    """找附录/附件的起始行下标，找不到返回 ``None``。

    只把「其后不再出现条款或章标题」的附录标记视为文档尾界——否则会把正文中间
    引用附件的句子当成文档结束。同时要求其前已有实质内容，避免整篇变附录。
    """
    for i, line in enumerate(lines):
        if not _APPENDIX_MARK.match(line):
            continue
        if _count("".join(lines[:i])) < 200:
            continue
        tail = lines[i + 1:]
        if any(_CLAUSE_HEAD.match(item) or _CHAPTER_HEAD.match(item) for item in tail):
            continue
        return i
    return None


def _scan_units(
    lines: Sequence[str],
    *,
    structure: str,
    unit_head: re.Pattern[str] | None,
) -> list[_Unit]:
    """按章/节/引用单元标记扫描出 :class:`_Unit` 列表。

    :param lines: 正文行
    :param structure: 结构类型
    :param unit_head: 引用单元的正则；条款型为「第X条」，意见方案型为「（X）」，
        无结构文档传 ``None``（整篇一块，由调用方直接处理）
    """
    units: list[_Unit] = []
    chapter: str | None = None
    section: str | None = None
    current: _Unit | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.lines:
            units.append(current)
        current = None

    appendix_at = _find_appendix_start(lines)
    body_lines = lines if appendix_at is None else lines[:appendix_at]
    appendix_lines = [] if appendix_at is None else list(lines[appendix_at:])

    for line in body_lines:
        # 超长「标题行」不是标题：它是标题被粘上了整段正文或表格数据。当作章/节标题
        # 会让 header 膨胀到近千字，因此让它走普通正文路径（header 少一段层级上下文，
        # 但「标题（文号）」仍在，溯源不受影响）。
        is_title_shaped = len(line) <= MAX_TITLE_CHARS

        chapter_match = _CHAPTER_HEAD.match(line)
        if chapter_match and is_title_shaped:
            flush()
            chapter = line
            section = None
            continue

        section_match = _SECTION_HEAD.match(line)
        if section_match and is_title_shaped:
            flush()
            section = line
            continue

        if structure == STRUCTURE_SECTION and _TOP_HEAD.match(line) and is_title_shaped:
            flush()
            chapter = line
            section = None
            # 这一行本身可能就带着正文（"一、科学规划产业布局。坚持疏近用远…"是
            # 意见/方案体最常见的写法）。原实现直接 continue，等于把这行丢掉：
            # 若全文只有「一、二、三」而没有「（一）」款，units 会一直是空的，
            # 整篇输出 0 块且不报错（实测序号 617：17 行、5115 字全部消失）。
            # 所以这里起一个临时单元承载该行，紧随其后的（一）（二）照旧各自成块。
            current = _Unit(None, None, chapter, section, [line])
            continue

        if unit_head is not None:
            head_match = unit_head.match(line)
            if head_match:
                flush()
                current = _Unit(
                    label_of(head_match.group(0)),
                    zh_number_to_int(head_match.group(1)),
                    chapter,
                    section,
                    [line],
                )
                continue

        if current is None:
            # 章标题后直接出现的正文，或首个引用单元之前的说明段落
            current = _Unit(None, None, chapter, section, [line], is_preamble=True)
        else:
            current.lines.append(line)

    flush()

    if appendix_lines:
        units.append(
            _Unit(None, None, None, None, appendix_lines, is_appendix=True)
        )
    return units


def label_of(raw: str) -> str:
    """把匹配到的标记整行裁成干净标签：``第 十九 条`` → ``第十九条``。

    OCR 会把条号拆成带空格的片段，标签里若保留空格，后续按标签聚合、比对、
    渲染都会各写一套清洗逻辑，因此在唯一的出口处归一化。
    """
    text = raw.strip()
    for pattern in (_CLAUSE_HEAD, _KUAN_HEAD):
        match = pattern.match(text)
        if match:
            return re.sub(r"[\s\u3000]", "", match.group(0))
    return re.sub(r"[\s\u3000]+", "", text)


def group_by_kuan(paragraphs: Sequence[Sequence[str]], max_kuan: int) -> list[list[str]]:
    """按「每块不超过 ``max_kuan`` 款」分组（A1 的款级切分）。

    款数超限时不能只靠字数装箱——七个短款合起来才 50 字，装箱会把它们又并回一块，
    等于没切。
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    count = 0
    for paragraph in paragraphs:
        is_kuan = bool(_KUAN_HEAD.match(paragraph[0]))
        if is_kuan and count >= max_kuan:
            blocks.append(current)
            current, count = [], 0
        current.extend(paragraph)
        if is_kuan:
            count += 1
    if current:
        blocks.append(current)
    return blocks


def merge_short_blocks(
    blocks: Sequence[Sequence[str]], min_chars: int, max_chars: int
) -> list[list[str]]:
    """把字数为 ``min_chars`` 的碎片并入相邻块。

    条款头常常很短（``第三条 帮扶资金主要用于：`` 仅 12 字），若单独成块，
    该块几乎不含可检索信息，却占掉一个 top-k 名额。合并方向优先向前，
    首块过短时向后并；合并后若会超出 ``max_chars`` 则放弃合并——宁可留个短块，
    也不产生超限块。
    """
    merged: list[list[str]] = []
    for block in blocks:
        if merged and _count("".join(block)) < min_chars:
            candidate = merged[-1] + list(block)
            if _count("".join(candidate)) <= max_chars:
                merged[-1] = candidate
                continue
        merged.append(list(block))
    if len(merged) > 1 and _count("".join(merged[0])) < min_chars:
        candidate = merged[0] + merged[1]
        if _count("".join(candidate)) <= max_chars:
            merged[1] = candidate
            merged.pop(0)
    return merged


def _split_unit(unit: _Unit, config: ChunkConfig) -> list[list[str]]:
    """把引用单元按 A1 切成若干块；不需要切时原样返回一块。

    附录/附件没有条款语义，按 ``flat_target_chars`` 装箱——``653_财会〔2021〕37号``
    的会计科目附录有 900 多行，若整块入库会被单块稀释掉全部可检索性。
    """
    paragraphs = _split_paragraphs(unit.lines)
    if unit.is_appendix:
        return merge_short_blocks(
            pack_paragraphs(paragraphs, config.flat_target_chars),
            config.min_chunk_chars,
            config.flat_target_chars,
        )

    text_len = _count("".join(unit.lines))
    kuan_count = sum(1 for line in unit.lines if _KUAN_HEAD.match(line))
    if text_len <= config.max_clause_chars and kuan_count <= config.max_kuan:
        return [list(unit.lines)]

    if kuan_count > config.max_kuan:
        # 款数超限时以款为界切分，不再做短块合并——否则短款会被合并回去，等于没切
        blocks = []
        for block in group_by_kuan(paragraphs, config.max_kuan):
            if _count("".join(block)) > config.max_clause_chars:
                blocks.extend(pack_paragraphs(_split_paragraphs(block), config.max_clause_chars))
            else:
                blocks.append(block)
        return blocks

    return merge_short_blocks(
        pack_paragraphs(paragraphs, config.max_clause_chars),
        config.min_chunk_chars,
        config.max_clause_chars,
    )


# ================================================================ 层级头


def build_header(
    doc: DocRecord,
    *,
    chapter: str | None,
    section: str | None,
    clause: str | None,
) -> str:
    """拼装层级头：标题（文号）｜章｜节｜条。

    文号为空或是占位文本时不写，避免把「无（审计结果公告通常无发文字号）」
    这类说明印成文号（红线：禁止编造文号）。
    """
    head = doc.title or ""
    if doc.has_doc_no:
        head = f"{head}（{doc.doc_no}）" if head else doc.doc_no
    parts = [head] if head else []
    for level in (chapter, section, clause):
        if level:
            parts.append(level)
    return "｜".join(parts)


# ================================================================ 主入口


def chunk_document(
    doc: DocRecord,
    text: str,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """把一篇清洗文本切成 chunk 列表。

    :param doc: 该文档的登记项（元数据唯一来源）
    :param text: 清洗文本全文
    :param config: 分块参数，默认按手册 A1
    :return: 按原文顺序排列的 chunk 列表；空文本返回空列表
    """
    cfg = config or ChunkConfig()
    body_lines, _start = split_body(text)
    lines = join_broken_lines(body_lines)
    if not lines:
        logger.warning("序号 %s 正文为空，跳过分块", doc.serial)
        return []

    structure = detect_structure(lines)
    if structure == STRUCTURE_CLAUSE:
        units = _scan_units(lines, structure=structure, unit_head=_CLAUSE_HEAD)
    elif structure == STRUCTURE_SECTION:
        units = _scan_units(lines, structure=structure, unit_head=_KUAN_HEAD)
    else:
        units = [_Unit(None, None, None, None, list(lines))]

    base_meta: dict[str, object] = {
        "category": doc.category,
        "validity": doc.validity,
        "is_current": doc.is_current,
        "publish_date": doc.publish_date,
        "issuer": doc.issuer,
        "source_url": doc.source_url,
        "has_doc_no": doc.has_doc_no,
    }

    chunks: list[Chunk] = []
    order = 0
    for unit in units:
        pieces = _split_unit(unit, cfg)
        total = len(pieces)
        for part, piece in enumerate(pieces, start=1):
            body = "\n".join(piece).strip()
            if not body:
                continue
            header = build_header(
                doc,
                chapter=unit.chapter,
                section=unit.section,
                clause=unit.clause,
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.serial}-{order:04d}",
                    order=order,
                    serial=doc.serial,
                    doc_no=doc.doc_no,
                    title=doc.title,
                    structure=structure,
                    chapter=unit.chapter,
                    section=unit.section,
                    clause=unit.clause,
                    clause_index=unit.clause_index,
                    part=part,
                    parts_total=total,
                    is_preamble=unit.is_preamble,
                    is_appendix=unit.is_appendix,
                    header=header,
                    body=body,
                    text=f"{header}\n{body}" if cfg.inject_header and header else body,
                    char_count=_count(body),
                    metadata=dict(base_meta),
                )
            )
            order += 1

    logger.debug(
        "序号 %s 分块完成：结构=%s，块数=%d", doc.serial, structure, len(chunks)
    )
    if not chunks:
        # 有正文却切不出任何块，说明解析器把这一篇整篇吞掉了。这类失败不会有任何
        # 异常，只有下游"怎么检索都检索不到某份文件"时才被发现（实测序号 617
        # 曾经如此：17 行 5115 字，输出 0 块，日志只有一行 DEBUG）。所以这里
        # 用 ERROR 明确点名，并让 chunk_corpus 在收尾处汇总名单。
        logger.error(
            "序号 %s 有正文但切出 0 块（结构=%s，%d 行 %d 字），该文件不会进入索引",
            doc.serial,
            structure,
            len(lines),
            _count("".join(lines)),
        )
    return chunks


def load_clean_text(doc: DocRecord, data_repo: Path | str) -> str:
    """读取文档的清洗文本。"""
    path = doc.resolve_clean_text(data_repo)
    if not path.is_file():
        raise FileNotFoundError(f"清洗文本不存在：{path}")
    return path.read_text(encoding="utf-8")


def chunk_corpus(
    documents: Iterable[DocRecord],
    data_repo: Path | str,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """批量分块。单篇失败不中断整批，错误记日志并跳过。"""
    all_chunks: list[Chunk] = []
    failed = 0
    empty: list[int] = []
    for doc in documents:
        try:
            text = load_clean_text(doc, data_repo)
        except (FileNotFoundError, UnicodeDecodeError) as error:
            logger.error("序号 %s 读取失败，跳过：%s", doc.serial, error)
            failed += 1
            continue
        chunks = chunk_document(doc, text, config)
        if not chunks and text.strip():
            empty.append(doc.serial)
        all_chunks.extend(chunks)

    logger.info("全库分块完成：%d 篇失败，共 %d 块", failed, len(all_chunks))
    if empty:
        # 汇总而不是靠人翻日志：`chunk_document` 已经逐篇打了 ERROR，这里再给一份
        # 名单，避免"跑完了看起来正常、其实少了几篇"这种最贵的成功假象
        logger.error("以下 %d 篇有正文却切出 0 块（不会进入索引）：%s", len(empty), empty)
    return all_chunks


def clause_sequence(chunks: Sequence[Chunk]) -> list[int]:
    """按出现顺序返回引用单元的阿拉伯序号（去重）。

    用于验收「条款号与 chunk 一一对齐零错位」：与原文条款序号逐一比对即可。
    """
    result: list[int] = []
    for chunk in chunks:
        if chunk.clause_index is None:
            continue
        if not result or result[-1] != chunk.clause_index:
            result.append(chunk.clause_index)
    return result


def check_clause_continuity(chunks: Sequence[Chunk]) -> list[str]:
    """报告条款型文档的条款序号断档或重排。

    只对 :data:`STRUCTURE_CLAUSE` 检查：意见方案型文档的「（一）」按一级标题重新
    编号，序号重复是正常排版，不是缺陷。

    数据仓的清洗文本本身存在缺条（如 ``26_粤财农〔2026〕100号`` 从第二十一条直接
    跳到第二十四条），这不是分块错误，但会让「第X条」与原文对不上，需要人工核对。
    本函数只诊断、不改内容。

    :return: 问题描述列表，空列表表示连续
    """
    problems: list[str] = []
    sequence = clause_sequence(
        [chunk for chunk in chunks if chunk.structure == STRUCTURE_CLAUSE]
    )
    for previous, current in zip(sequence, sequence[1:], strict=False):
        if current == previous + 1:
            continue
        if current <= previous:
            problems.append(f"条款序号倒序：第{previous}条之后出现第{current}条")
        else:
            missing = list(range(previous + 1, current))
            problems.append(
                f"条款序号断档：第{previous}条之后直接是第{current}条"
                f"（缺 {len(missing)} 条，如 {missing[:5]}）"
            )
    return problems
