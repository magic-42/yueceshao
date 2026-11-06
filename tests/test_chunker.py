"""rag.chunker 的单元测试。

对应《开发手册 v1.0》§3.1 PR#2 验收标准：
1. 对三篇指定文件的条款号与 chunk 一一对齐、零错位；
2. 边界用例（无条款文件、超长条）有测试。

除验收标准外，还固化了真实语料里踩到的三个坑，防止后续改动把它们重新踩回来：
OCR 把条号拆成带空格的片段、条号后无空格直接接正文、附录吞进尾条。

真实数据仓不存在时集成测试自动跳过，CI 与本地都能跑。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from textwrap import dedent

import pytest

from rag.chunker import (
    STRUCTURE_CLAUSE,
    STRUCTURE_FLAT,
    STRUCTURE_SECTION,
    ChunkConfig,
    check_clause_continuity,
    chunk_document,
    clause_sequence,
    detect_structure,
    join_broken_lines,
    split_body,
    zh_number_to_int,
)
from rag.loader import DocRecord, load_registry


def make_doc(**overrides: object) -> DocRecord:
    """构造一条最小可用的 DocRecord，字段可按需覆盖。"""
    base: dict[str, object] = {
        "serial": 1,
        "doc_no": "粤财农〔2020〕106号",
        "title": "广东省涉农资金统筹整合管理办法",
        "category": "资金管理办法",
        "validity": "有效",
        "issuer": "广东省财政厅",
        "publisher": "",
        "publish_date": "2020-08-24",
        "source_url": "https://example.invalid/a",
        "clean_text_path": "03_清洗文本/01_x.txt",
        "clean_text_chars": 100,
        "snapshot_path": "",
        "attachment_path": "",
        "valid_until": "",
        "superseded_by": "",
        "supersedes": "",
        "validity_note": "",
        "verify_status": "已人工核验",
        "remark": "",
        "fetched_date": "",
        "raw": {},
    }
    base.update(overrides)
    return DocRecord(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ 数字与预处理


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一", 1),
        ("十", 10),
        ("十二", 12),
        ("二十", 20),
        ("二十一", 21),
        ("二十九", 29),
        ("三十四", 34),
        ("一百零五", 105),
        ("二十 一", 21),   # OCR 拆开
        (" 十九 ", 19),
        ("29", 29),
        ("", None),
        ("甲乙", None),
    ],
)
def test_zh_number_to_int(text: str, expected: int | None) -> None:
    """中文数字转换要覆盖 OCR 拆字与无法解析的情形。"""
    assert zh_number_to_int(text) == expected


def test_join_broken_lines_merges_ocr_wrap() -> None:
    """OCR 残留的硬换行应合并，句末标点后则不合并。"""
    lines = ["第一条 应当以人民币", "元”为金额单位。", "第二条 另有规定。"]
    assert join_broken_lines(lines) == [
        "第一条 应当以人民币元”为金额单位。",
        "第二条 另有规定。",
    ]


# ------------------------------------------------------------------ 结构判定


def test_detect_structure_clause() -> None:
    """三条以上「第X条」判为条款型。"""
    lines = ["第一条 甲。", "第二条 乙。", "第三条 丙。"]
    assert detect_structure(lines) == STRUCTURE_CLAUSE


def test_detect_structure_section() -> None:
    """无条款但有两个「一、」级标题时判为意见方案型。"""
    lines = ["一、主要目标", "到2025年建成。", "二、推进整合", "（一）归并专项。"]
    assert detect_structure(lines) == STRUCTURE_SECTION


def test_detect_structure_flat() -> None:
    """短通知类判为无结构。"""
    assert detect_structure(["现将有关事项通知如下。", "请遵照执行。"]) == STRUCTURE_FLAT


def test_split_body_strips_toc() -> None:
    """目录区（含章名枚举）必须剥掉，否则会凭空多出一整套空章。"""
    text = dedent(
        """\
        目 录
        第一章 总则
        第二章 资金
        第一章 总则
        第一条 为规范管理，制定本办法。
        """
    )
    lines, _start = split_body(text)
    assert lines[0] == "第一章 总则"
    assert lines[1].startswith("第一条")
    assert sum(1 for line in lines if line.startswith("第一章")) == 1


def test_split_body_skips_header_block() -> None:
    """有 ``#`` 头部元数据块时，从 ``=====`` 之后取正文。"""
    text = dedent(
        """\
        # 粤策哨 清洗文本｜序号 1
        # 文号：粤府〔2023〕34号｜发布机关：广东省人民政府

        ===== 网页正文 =====

        第一条 为规范管理，制定本办法。
        """
    )
    lines, _start = split_body(text)
    body = join_broken_lines(lines)
    assert body[0].startswith("第一条")
    assert not any(line.startswith("#") for line in lines)


# ------------------------------------------------------------------ 条款型分块


def test_clause_basic_alignment() -> None:
    """最基本的条款型：一条一块，条款号与原文一致。"""
    text = dedent(
        """\
        第一章 总则
        第一条 为规范管理，制定本办法。
        第二条 本办法适用于本省行政区域。
        第二章 资金管理
        第三条 资金实行专户管理。
        """
    )
    chunks = chunk_document(make_doc(), text)

    assert [chunk.clause for chunk in chunks] == ["第一条", "第二条", "第三条"]
    assert clause_sequence(chunks) == [1, 2, 3]
    # 章路径跟随最近的章标题
    assert chunks[0].chapter == "第一章 总则"
    assert chunks[2].chapter == "第二章 资金管理"
    assert check_clause_continuity(chunks) == []


def test_ocr_space_in_clause_number() -> None:
    """OCR 把条号拆成「第二十 一 条」时，标签要归一化为「第二十一条」。"""
    text = dedent(
        """\
        第二十条 甲款。
        第二十 一 条 乙款。
        第 二十二 条 丙款。
        """
    )
    chunks = chunk_document(make_doc(), text)

    assert [chunk.clause for chunk in chunks] == ["第二十条", "第二十一条", "第二十二条"]
    assert clause_sequence(chunks) == [20, 21, 22]


def test_clause_head_without_space() -> None:
    """条号与正文之间无空格（``第一条为了规范…``）也必须识别。"""
    text = "第一条为了规范农民专业合作社会计工作，制定本制度。\n第二条本制度适用于合作社。"
    chunks = chunk_document(make_doc(), text)

    assert len(chunks) == 2
    assert chunks[0].clause == "第一条"
    assert chunks[0].body.startswith("第一条为了规范")


def test_long_clause_split_by_kuan() -> None:
    """超 300 字的条要切款级子块，且同一父条的 clause 保持不变。"""
    body = "".join(f"（{n}）第{n}项内容，" + "资金管理要求。" * 12 for n in "一二三四五六")
    text = f"第三条 帮扶资金主要用于：\n{body}"
    chunks = chunk_document(make_doc(), text)

    assert len(chunks) > 1
    assert {chunk.clause for chunk in chunks} == {"第三条"}
    assert {chunk.clause_index for chunk in chunks} == {3}
    assert [chunk.part for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.parts_total == len(chunks) for chunk in chunks)
    assert all(chunk.char_count <= 300 for chunk in chunks), "切分后单块不得超限"


def test_long_clause_split_by_kuan_count() -> None:
    """款数超过 5 也要切，即使总字数没超。"""
    body = "\n".join(f"（{n}）短款。" for n in "一二三四五六七")
    chunks = chunk_document(make_doc(), f"第五条 事项如下：\n{body}")

    assert len(chunks) > 1


def test_short_clause_head_merged() -> None:
    """12 字的条款头不得独占一块——会被并入后续款。"""
    body = "".join(f"（{n}）" + "具体要求。" * 10 for n in "一二三四五六")
    chunks = chunk_document(make_doc(), f"第三条 帮扶资金主要用于：\n{body}")

    assert chunks[0].char_count >= 30
    assert chunks[0].body.startswith("第三条 帮扶资金主要用于：")


def test_preamble_before_first_clause() -> None:
    """首个条款之前的说明段落单独成块，标记 is_preamble。"""
    text = dedent(
        """\
        现将有关事项通知如下。
        第一条 为规范管理，制定本办法。
        第二条 另有规定。
        第三条 另有规定。
        """
    )
    chunks = chunk_document(make_doc(), text)

    assert chunks[0].is_preamble is True
    assert chunks[0].clause is None
    assert chunks[1].clause == "第一条"


# ------------------------------------------------------------------ 意见方案型


def test_section_structure() -> None:
    """一、/（一）型：一、作章路径，（一）作引用单元。"""
    text = dedent(
        """\
        一、主要目标
        到2025年基本建成涉农资金统筹整合长效机制，实现行业内资金统筹。
        二、推进整合
        （一）归并涉农资金专项。在预算编制阶段清理整合交叉重复的专项。
        （二）设定任务清单。实行大专项加任务清单管理模式。
        """
    )
    chunks = chunk_document(make_doc(), text)

    heads = [chunk.clause for chunk in chunks]
    assert "（一）" in heads
    assert "（二）" in heads
    unit = next(chunk for chunk in chunks if chunk.clause == "（一）")
    assert unit.chapter == "二、推进整合"
    assert unit.clause_index == 1


def test_section_doc_without_kuan_keeps_all_text() -> None:
    """只有「一、二、三」而没有「（一）」款的文档，正文一个字都不能丢。

    这是真实事故：原实现把「一、」行当章标题后直接丢弃，全文没有款标记时
    units 一直为空 → 整篇输出 0 块、不报任何错（序号 617，17 行 5115 字全部消失）。
    """
    text = dedent(
        """\
        一、总体要求。坚持疏近用远、生态发展，全面推进乡村振兴。
        二、主要任务。加快建设现代农业产业园，推动产业融合发展。
        三、保障措施。加强组织领导，落实资金保障。
        """
    )
    chunks = chunk_document(make_doc(), text)

    assert chunks, "不能输出 0 块——那意味着整篇文件不会进入索引"
    joined = re.sub(r"\s", "", "".join(chunk.body for chunk in chunks))
    for line in text.strip().splitlines():
        assert re.sub(r"\s", "", line) in joined, f"这一行丢失了：{line}"


def test_section_heading_with_inline_text_is_kept() -> None:
    """「一、标题。正文」同行的写法，标题行里的正文不能被当成纯标题吞掉。"""
    text = dedent(
        """\
        一、主要目标。到2025年基本建成涉农资金统筹整合长效机制。
        （一）归并涉农资金专项。在预算编制阶段清理整合交叉重复的专项。
        二、推进整合。在预算编制阶段同步下达任务清单。
        （二）设定任务清单。实行大专项加任务清单管理模式。
        """
    )
    chunks = chunk_document(make_doc(), text)

    joined = re.sub(r"\s", "", "".join(chunk.body for chunk in chunks))
    assert "到2025年基本建成涉农资金统筹整合长效机制" in joined
    assert "在预算编制阶段同步下达任务清单" in joined
    assert "（一）" in [chunk.clause for chunk in chunks]


# ------------------------------------------------------------------ 无结构与其他边界


def test_flat_document_single_chunk() -> None:
    """短通知整篇一块。"""
    text = "现将《某某办法》印发给你们，请遵照执行。"
    chunks = chunk_document(make_doc(), text)

    assert len(chunks) == 1
    assert chunks[0].clause is None
    assert chunks[0].structure == STRUCTURE_FLAT


def test_empty_text_returns_empty() -> None:
    """空正文返回空列表，不抛异常。"""
    assert chunk_document(make_doc(), "   \n\n  \n") == []


def test_appendix_detached_from_last_clause() -> None:
    """附录须从尾条剥离：否则 653_财会〔2021〕37号 的会计科目表会吞进第六十一条。"""
    long_body = "会计处理应当遵循权责发生制原则，如实反映财务状况。" * 12
    text = dedent(
        f"""\
        第一章 总则
        第一条 {long_body}
        第二条 本制度自2023年1月1日起施行。
        附录
        一、会计科目
        顺序号 编号 会计科目名称
        """
    )
    chunks = chunk_document(make_doc(), text)

    appendix = [chunk for chunk in chunks if chunk.is_appendix]
    assert len(appendix) >= 1
    assert appendix[0].clause is None
    tail = next(chunk for chunk in chunks if chunk.clause == "第二条")
    assert "会计科目" not in tail.body


def test_appendix_marker_inside_body_kept() -> None:
    """正文中间提到「附件」但后面仍有条款时，不得截断正文。"""
    long_body = "资金管理应当遵循专款专用原则，单独核算。" * 12
    text = dedent(
        f"""\
        第一条 {long_body}
        附件：政策依据清单
        第二条 本办法自印发之日起施行。
        第三条 本办法由省财政厅解释。
        """
    )
    chunks = chunk_document(make_doc(), text)

    assert not any(chunk.is_appendix for chunk in chunks)
    assert "第三条" in [chunk.clause for chunk in chunks]


# ------------------------------------------------------------------ 层级头与元数据


def test_header_injected_with_title_and_clause() -> None:
    """A2：层级头要含标题、文号、章路径与条款号。"""
    text = "第一章 总则\n第一条 为规范管理，制定本办法。"
    chunk = chunk_document(make_doc(), text)[0]

    assert chunk.header.startswith("广东省涉农资金统筹整合管理办法（粤财农〔2020〕106号）")
    assert "第一章 总则" in chunk.header
    assert chunk.header.endswith("第一条")
    assert chunk.text.startswith(chunk.header)


def test_placeholder_doc_no_not_injected() -> None:
    """占位文号（「无（…）」）不得进入层级头——这是禁止编造文号的执行点。"""
    doc = make_doc(doc_no="无（审计结果公告通常无发文字号）")
    chunk = chunk_document(doc, "第一条 为规范管理，制定本办法。")[0]

    assert "无（" not in chunk.header
    assert "〔" not in chunk.header
    assert doc.has_doc_no is False


def test_inject_header_can_be_disabled() -> None:
    """消融对比：关闭头注入时 text 等于 body，header 仍生成以便排查。"""
    text = "第一条 为规范管理，制定本办法。"
    chunk = chunk_document(make_doc(), text, ChunkConfig(inject_header=False))[0]

    assert chunk.text == chunk.body
    assert chunk.header != ""


def test_metadata_carries_validity_fields() -> None:
    """元数据要带上效力与类目，供检索层做 where 硬过滤（A6）。"""
    doc = make_doc(validity="已失效")
    chunk = chunk_document(doc, "第一条 为规范管理，制定本办法。")[0]
    meta = chunk.as_metadata()

    assert meta["validity"] == "已失效"
    assert meta["is_current"] is False
    assert meta["doc_no"] == "粤财农〔2020〕106号"
    assert meta["clause"] == "第一条"


def test_chunk_ids_are_deterministic() -> None:
    """同一文本两次分块结果完全一致（增量重建依赖此性质）。"""
    text = "第一条 甲。\n第二条 乙。\n第三条 丙。"
    first = chunk_document(make_doc(), text)
    second = chunk_document(make_doc(), text)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_continuity_reports_gap() -> None:
    """缺条要能报出来（26_粤财农〔2026〕100号 从第二十一条直接跳到第二十四条）。"""
    text = "第二十一条 甲。\n第二十四条 乙。"
    chunks = chunk_document(make_doc(), text)

    problems = check_clause_continuity(chunks)
    assert len(problems) == 1
    assert "断档" in problems[0]


# --------------------------------------------------------------- 真实数据集成测试


def _find_data_repo() -> Path | None:
    """找数据仓根目录；找不到返回 None（集成测试自动跳过）。

    优先级：``YUECESHAO_DATA_REPO`` → ``config.json`` 的 ``data_repo`` → 同级目录的
    常见命名。读配置文件这一步不能省：README 让协作者把数据仓路径配在
    ``config.json`` 里，只认环境变量和固定目录名会让「按文档配好了却整批 skip」
    ——测试报「跳过」而不是「失败」，看起来一切正常，实际一条真实数据都没跑。
    """
    project_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []
    env = os.environ.get("YUECESHAO_DATA_REPO")
    if env:
        candidates.append(Path(env))

    config_path = project_root / "config.json"
    if config_path.is_file():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = None
        value = raw.get("data_repo") if isinstance(raw, dict) else None
        if isinstance(value, str) and value.strip():
            configured = Path(value.strip())
            candidates.append(
                configured if configured.is_absolute() else project_root / configured
            )

    for name in ("yueceshao-data", "yueceshao-data.partial", "yueceshao-data-full"):
        candidates.append(project_root.parent / name)

    for base in candidates:
        if (base / "00_工作区" / "registry.json").is_file():
            return base
    return None


DATA_REPO = _find_data_repo()

#: 验收指定的三篇（序号按数据仓 registry）
VERIFY_SERIALS = (22, 26, 35)


@pytest.mark.skipif(DATA_REPO is None, reason="本机未克隆数据仓，跳过集成测试")
@pytest.mark.parametrize("serial", VERIFY_SERIALS)
def test_real_document_clause_alignment(serial: int) -> None:
    """验收标准：条款号与 chunk 一一对齐、零错位。

    校验方式不依赖被测代码自身的正则——从原文用独立正则重新抽取条款号序列，
    再与 chunk 侧的 ``clause_sequence`` 逐项比对。
    """
    assert DATA_REPO is not None
    registry = DATA_REPO / "00_工作区" / "registry.json"
    result = load_registry(registry)
    doc = next(item for item in result.documents if item.serial == serial)
    text = doc.resolve_clean_text(DATA_REPO).read_text(encoding="utf-8")

    chunks = chunk_document(doc, text)
    assert chunks, "验收文件不应产出空结果"

    # 原文侧独立抽取：行首「第…条」，容忍字间空白
    pattern = re.compile(
        r"^\s*第\s*([一二三四五六七八九十百零〇两\d]"
        r"[一二三四五六七八九十百零〇两\d\s]{0,10}?)\s*条")
    expected: list[int] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            value = zh_number_to_int(match.group(1))
            if value is not None and (not expected or expected[-1] != value):
                expected.append(value)

    assert clause_sequence(chunks) == expected, "条款号序列必须与原文逐项一致"

    # 每个 chunk 的 clause 标签都能在原文对应条款上找到
    labels = {chunk.clause for chunk in chunks if chunk.clause}
    assert len(labels) == len(expected), "条款标签数应与原文条款数相同"


@pytest.mark.skipif(DATA_REPO is None, reason="本机未克隆数据仓，跳过集成测试")
def test_real_corpus_no_giant_chunks() -> None:
    """全库抽查：不得出现单块吞掉整篇的巨块（附录除外）。"""
    assert DATA_REPO is not None
    registry = DATA_REPO / "00_工作区" / "registry.json"
    result = load_registry(registry)

    offenders: list[tuple[int, int]] = []
    for doc in result.documents[:60]:  # 抽样 60 篇，兼顾速度
        path = doc.resolve_clean_text(DATA_REPO)
        if not path.is_file():
            continue
        chunks = chunk_document(doc, path.read_text(encoding="utf-8"))
        for chunk in chunks:
            if not chunk.is_appendix and chunk.char_count > 1200:
                offenders.append((doc.serial, chunk.char_count))

    assert offenders == [], f"存在超长块（序号, 字数）：{offenders[:5]}"


@pytest.mark.skipif(DATA_REPO is None, reason="本机未克隆数据仓，跳过集成测试")
def test_real_corpus_no_document_silently_yields_zero_chunks() -> None:
    """全库回归：不允许任何一篇"有正文却切出 0 块"。

    这条守的是最贵的一种失败——文件在索引里根本不存在，检索永远命中不了它，
    而日志里只有一行 DEBUG。序号 617 曾如此（5115 字整篇消失）：全文只有
    「一、二、三」级标题、没有「（一）」款标记。
    """
    assert DATA_REPO is not None
    registry = DATA_REPO / "00_工作区" / "registry.json"
    result = load_registry(registry)

    offenders: list[int] = []
    for doc in result.documents:
        path = doc.resolve_clean_text(DATA_REPO)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        if not chunk_document(doc, text):
            offenders.append(doc.serial)

    assert offenders == [], f"以下文档有正文却切出 0 块：{offenders}"


# --------------------------------------------------- 目录剥离与标题行（评审 P2）


def test_split_body_strips_leader_toc_without_clauses() -> None:
    """目录靠**引导符**剥离，与有没有「第X条」无关。

    全库 396 篇意见方案型／规划类文档通篇没有条款头，原实现因此整段保留目录，
    随后目录条目被当成章标题拼进 header（序号 203 单块 header 达 1697 字）。
    """
    text = dedent(
        """\
        某某区畜禽养殖污染防治规划
        目录
        一、规划背景................................................ 1
        二、规划目标................................................ 2
        为控制畜禽养殖污染无序扩张的趋势，必须强化规划政策引导。
        一、规划背景
        近年来，某某区畜禽养殖业迅速发展，既丰富了城乡居民的菜篮子。
        """
    )

    lines, _start = split_body(text)

    assert not any("......" in line for line in lines), "目录条目没剥干净"
    assert "目录" not in lines
    assert any(line.startswith("为控制畜禽养殖污染") for line in lines), "正文被误删了"


def test_split_body_strips_toc_split_by_ocr() -> None:
    """OCR 把「目录」拆成「目」「录」两行时也要认出来（序号 232 实测如此）。"""
    text = dedent(
        """\
        某某市养殖水域滩涂规划
        目
        录
        第一章 总则................................. 1
        第一节 前言................................. 2
        第一章 总则
        第一条 为规范管理，制定本办法。
        """
    )

    lines, _start = split_body(text)

    assert "目" not in lines and "录" not in lines
    assert not any("......" in line for line in lines)
    assert lines[0] == "第一章 总则"
    assert lines[1].startswith("第一条")


def test_heading_line_not_merged_with_following_paragraph() -> None:
    """标题行不粘后续正文——粘上之后整段正文都会被当成章标题。"""
    text = dedent(
        """\
        一、总体要求
        坚持以习近平新时代中国特色社会主义思想为指导，全面推进乡村振兴。
        二、主要任务
        加快建设现代农业产业园，推动产业融合发展。
        """
    )

    lines = join_broken_lines(split_body(text)[0])

    assert lines[0] == "一、总体要求"
    assert lines[1].startswith("坚持以习近平"), f"标题被粘上了正文：{lines[0][:40]}"


def test_overlong_heading_not_injected_into_header() -> None:
    """超长「标题行」其实是表格或正文，不能当章标题塞进 header。

    序号 295 的表格行达 891 字、序号 40 的正文达 786 字，改动前它们整行进 header，
    而 header 又拼进 text —— 头注入从「补上下文」变成「往 embedding 输入里塞正文」。
    """
    table_line = "二、" + "夏收粮食其中早稻种植面积亩产总产" * 30
    text = dedent(
        f"""\
        一、总体要求。坚持全面推进乡村振兴。
        （一）指导思想。以习近平新时代思想为指导。
        {table_line}
        （二）基本原则。坚持因地制宜、分类施策。
        """
    )

    chunks = chunk_document(make_doc(), text)

    assert chunks, "不应切出 0 块"
    assert all("夏收粮食" not in chunk.header for chunk in chunks)
    assert all(len(chunk.header) < 200 for chunk in chunks), [
        len(chunk.header) for chunk in chunks
    ]
    joined = "".join(chunk.body for chunk in chunks)
    assert "夏收粮食" in joined, "表格内容必须仍在正文里，不能因为不进 header 就丢了"
