"""app/real_components.py — 真实检索适配器：稠密检索接 Chroma 索引＋规则扫描接数据仓。

背景（feat/real-retrieval，承接 feat/app-skeleton 的编排骨架）
----------------------------------------------------------------
:class:`app.graph.OrchestrationComponents` 有五个回调注入点。骨架阶段（#36/#38）全部
用 Mock 夹具；本模块把其中**两个**换成真实实现，让编排图第一次跑在真数据上：

- ``retrieve``：``rag.retriever.DenseRetriever``（ChromaBackend＋RetrievalConfig）——
  查询经 Qwen3-Embedding-0.6B 编码后做 top-k 稠密检索，``is_current`` 时效硬过滤
  默认前置（A6），红线④在检索端再兜底一次；
- ``rule_scan``：``rules.loader.load_store_from_config``＋``NegRuleMatcher``——
  负面清单规则全量确定性扫描（②段不走检索），命中行转 ``gen.schema.NegativeHit``
  （hit_type="规则命中"，matched_words 按概念组配对）；
- ``extract`` / ``entail_review`` / ``generate``：本步仍用 Mock 夹具
  （``app.pipeline`` 的合法 JSON），文件内预留 :class:`QwenLLM`（OpenAI 兼容客户端，
  key 只从 ``.env``／环境变量 ``DASHSCOPE_API_KEY`` 读取，缺失即抛清晰错误）。

编排图形状的适配
----------------
图里检索命中的形状是 ``{"chunk_id", "text", "metadata"}``（与 Mock 夹具同构），且
``chunk_id`` 同时充当渲染层的引用标记 ``[Cn]``（``gen.citation.build_meta_table`` 用它
当表键，校验器的标记白名单也只认 ``[C数字]`` 形态）。真实索引的 chunk_id 是
``{序号}-{顺序号}``，因此适配器按检索顺序**重新分配标记** ``[C1]..[Ck]``，真实
chunk_id 保留在 ``metadata.source_chunk_id`` 便于溯源。

②段引用为什么也能解析
----------------------
②段（负面清单命中）是规则库全量扫描的结果，不依赖稠密召回——大多数规则条款不会
出现在 top-k 里。但图节点只用**检索命中**建元数据表，NegativeHit 标记若查不到表项，
渲染层只能输出「依据标注缺失」。所以适配器把规则命中对应的条款块也并入命中列表
（``metadata.origin = "rules"``，文号/条款号/层级/效力全部取自数据仓规则库的溯源
字段——真数据、非编造），保证②段引用完整可回溯。规则条款的正文用规则库逐字原文
（与索引块同源），不额外抓取。

红线遵守
--------
- 文号全部来自检索返回的元数据或数据仓规则库溯源字段，适配器不拼接、不猜测；
- 姿态：②段是「命中提示（建议对照原文）」，渲染层不输出结论性措辞；
- 审计公告不进检索（``rag.retriever`` 检索端兜底）；索引与模型权重本机存放、不入 git。
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.graph import OrchestrationComponents, build_checkup_graph
from app.pipeline import (
    MOCK_ENTAILMENT_JSON,
    MOCK_EXTRACTION_JSON,
    MOCK_GENERATION_JSON,
    PipelineResult,
)
from gen.prompts import (
    CONTEXT_BLOCK_FORMAT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_TEMPERATURE,
    GENERATION_USER_TEMPLATE,
    NEGATIVE_ENTAILMENT_SYSTEM_PROMPT,
    NEGATIVE_ENTAILMENT_USER_TEMPLATE,
    render_prompt,
)
from gen.schema import HIT_RULE, NegativeHit
from rag.chunker import zh_number_to_int
from rag.retriever import Hit, RetrievalResult
from rules.engine import MatchResult, NegRuleMatcher, RuleHit
from rules.lexicon import FORBIDDEN_USE_CONCEPTS
from rules.loader import load_store_from_config
from rules.schema import RuleRecord, display_priority_key

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

ORIGIN_RETRIEVAL = "retrieval"
"""命中来自稠密检索（真索引块）。"""

ORIGIN_RULES = "rules"
"""命中来自负面清单规则库（②段条款块，保证引用可解析）。"""

ORIGIN_NEGATIVE = "negative-list"
"""命中来自 G01-G11 禁止用途载体（②段蕴含复核的待审条款，不参与①段生成）。"""

_G01G11_GROUPS = frozenset(f"G{i:02d}" for i in range(1, 12))
_NEGATIVE_WORDS = ("不得", "禁止", "严禁")

_UNCLAUSED_MARK = "未分条"
_CLAUSE_NO_PATTERN = re.compile(r"^第?\s*([0-9一二三四五六七八九十百零〇两]+)\s*条?$")


class RealAdapterError(RuntimeError):
    """真实适配器装配失败（索引／模型／规则库缺失，报错带可操作提示）。"""


class QwenKeyMissingError(RealAdapterError):
    """未配置 DASHSCOPE_API_KEY（.env 或环境变量），LLM 调用被明确拒绝。"""


# ================================================================ 配置与 .env


def _load_config(config_path: Path) -> dict[str, Any]:
    """读本机 config.json；不存在时返回空配置（Mock 组件不依赖它）。"""
    import json

    if not config_path.is_file():
        logger.warning("配置文件不存在：%s（真实检索需要它指向数据仓/索引/模型）", config_path)
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_env_file(path: Path | str) -> dict[str, str]:
    """解析极简 ``.env``：``KEY=VALUE``，容忍注释、空行、``export`` 前缀与引号。

    不引入 python-dotenv 之类的额外依赖：只需要一个 key，格式也由本仓库自己约定。
    """
    values: dict[str, str] = {}
    target = Path(path)
    if not target.is_file():
        return values
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _resolve_path(raw: str, config_path: Path) -> Path:
    """相对路径按配置文件所在目录解析（不依赖当前工作目录）。"""
    path = Path(raw)
    return path if path.is_absolute() else config_path.parent / path


# ================================================================ 条款号归一


def clause_no_to_index(clause_no: str) -> int | None:
    """规则库条款号 → chunker 的阿拉伯条序号（用于标记去重）。

    规则库字段是裸中文序号（「十八」），偶有「第四条」形态；「未分条（原文无条款号）」
    返回 None。
    """
    text = (clause_no or "").strip()
    if not text or _UNCLAUSED_MARK in text:
        return None
    match = _CLAUSE_NO_PATTERN.match(text)
    if not match:
        return None
    return zh_number_to_int(match.group(1))


def format_clause_label(clause_no: str) -> str:
    """规则库条款号 → 引用展示形态：「十八」→「第十八条」；未分条返回空串。"""
    text = (clause_no or "").strip()
    if not text or _UNCLAUSED_MARK in text:
        return ""
    if text.startswith("第") and text.endswith("条"):
        return text
    return f"第{text}条"


# ================================================================ 命中形状转换


def hit_to_orchestration_hit(hit: Hit, marker: str) -> dict[str, Any]:
    """检索 Hit → 编排图命中 dict（chunk_id=标记，metadata 带渲染层字段别名）。

    ``clause_no`` 是渲染层（``gen.citation.ChunkMeta.from_mapping``）读取的键，而索引
    的字段名是 ``clause``——缺口径就在适配器这层抹平，不在渲染层改。
    """
    metadata = dict(hit.metadata)
    metadata.setdefault("clause_no", metadata.get("clause", ""))
    metadata["source_chunk_id"] = hit.chunk_id
    metadata["origin"] = ORIGIN_RETRIEVAL
    return {"chunk_id": marker, "text": hit.text, "metadata": metadata}


def rule_orchestration_hit(
    rule: RuleRecord, marker: str, *, origin: str = ORIGIN_RULES,
) -> dict[str, Any]:
    """规则库条款 → 命中 dict（溯源字段全部取自规则库，供②段引用解析）。"""
    metadata: dict[str, Any] = {
        "doc_no": rule.doc_no,
        "clause_no": format_clause_label(rule.clause_no),
        "validity": rule.effect_status,
        "level": rule.level,
        "title": rule.doc_title,
        "is_current": rule.is_effective,
        "serial": rule.doc_index,
        "category": rule.doc_category,
        "origin": origin,
    }
    return {"chunk_id": marker, "text": rule.clause_text, "metadata": metadata}


def is_g01g11_carrier(record: RuleRecord) -> bool:
    """是否 G01-G11 禁止用途载体：组号落在 G01-G11 且条款原文含禁止性措辞。

    承接列举项（如「修建楼堂馆所支出」）无「不得/禁止/严禁」字样，不进载体——
    其禁止语义由同组含「不得」的父条（如「专项资金不得用于…建设楼堂馆所…」）覆盖。
    """
    return (
        record.group_id in _G01G11_GROUPS
        and any(word in record.clause_text for word in _NEGATIVE_WORDS)
    )


def dedup_carriers(records: Sequence[RuleRecord]) -> list[RuleRecord]:
    """按条款原文去重，同文保留展示优先级最高（现行有效＋层级更具体）的载体。"""
    best: dict[str, RuleRecord] = {}
    for record in records:
        existing = best.get(record.clause_text)
        if existing is None or display_priority_key(record) < display_priority_key(existing):
            best[record.clause_text] = record
    return sorted(best.values(), key=display_priority_key)


def matched_word_pairs(
    rule_hit: RuleHit,
    concepts: Mapping[str, Sequence[str]] = FORBIDDEN_USE_CONCEPTS,
) -> tuple[tuple[str, str], ...]:
    """把引擎的两侧命中词按**概念组**配对成 ``(描述侧词, 条款侧词)``。

    引擎输出 ``matched_description_terms`` 与 ``matched_terms`` 两列，共同概念组
    （``rule_hit.concept``）下的词两两配对；概念组信息不可用时退回 zip 对齐
    （用空串补位，保证至少一条可读配对）。
    """
    term_concept = {
        term: concept for concept, terms in concepts.items() for term in terms
    }
    pairs: list[tuple[str, str]] = []
    for concept in rule_hit.concept.split("／"):
        description_terms = [
            term for term in rule_hit.matched_description_terms
            if term_concept.get(term) == concept
        ]
        clause_terms = [
            term for term in rule_hit.matched_terms if term_concept.get(term) == concept
        ]
        pairs.extend(
            (description_term, clause_term)
            for description_term in description_terms
            for clause_term in clause_terms
        )
    if not pairs:
        from itertools import zip_longest

        pairs = list(zip_longest(
            rule_hit.matched_description_terms, rule_hit.matched_terms, fillvalue=""))
    return tuple(pairs)


# ================================================================ LLM 适配器


class MockLLM:
    """骨架阶段 LLM 占位：extract/entail/generate 返回 app.pipeline 的合法 JSON 夹具。"""

    def extract(self, description: str) -> str:
        """查询结构化抽取（Mock：固定夹具）。"""
        return MOCK_EXTRACTION_JSON

    def entail_review(
        self, description: str, hits: Sequence[dict[str, Any]], feedback: str | None = None,
    ) -> str:
        """负面条款蕴含复核（Mock：固定夹具；feedback 忽略，不联网）。"""
        return MOCK_ENTAILMENT_JSON

    def generate(
        self,
        description: str,
        extraction_json: str,
        context_blocks: str,
        feedback: str | None,
    ) -> str:
        """①③④段生成（Mock：固定夹具；不联网、不花钱）。"""
        return MOCK_GENERATION_JSON


class QwenLLM:
    """qwen（OpenAI 兼容）适配器骨架：key 只从环境变量／.env 读取，绝不硬编码。"""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        api_key: str | None = None,
        env_path: Path | str = DEFAULT_ENV_PATH,
        client: Any = None,
    ) -> None:
        """:param config: 整份 config.json（取 ``generation`` 段）
        :param api_key: 显式注入（测试用）；None 时读环境变量／.env
        :param env_path: .env 路径
        :param client: 注入 OpenAI 客户端（测试用）；None 时懒创建
        """
        section = (config or {}).get("generation")
        settings: Mapping[str, Any] = section if isinstance(section, Mapping) else {}
        self.model = str(settings.get("model") or "qwen-plus")
        self.base_url = str(
            settings.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.api_key_env = str(settings.get("api_key_env") or "DASHSCOPE_API_KEY")
        # 评测纪律：生成调用一律 temperature=0，配置不得覆盖（手册 §5）
        self.temperature = GENERATION_TEMPERATURE
        self._api_key = api_key
        self._env_path = Path(env_path)
        self._client = client

    @property
    def api_key(self) -> str:
        """取 API key：显式注入 → 环境变量 → .env；全无则抛清晰错误。"""
        if self._api_key:
            return self._api_key
        found = os.environ.get(self.api_key_env) or read_env_file(self._env_path).get(
            self.api_key_env)
        if not found:
            raise QwenKeyMissingError(
                f"未找到 {self.api_key_env}：请在本机 {self._env_path} 写入 "
                f"{self.api_key_env}=<你的 key>，或导出为同名环境变量后重试。"
                "该 key 已在 .gitignore 中，永不入库。")
        self._api_key = found
        return found

    @property
    def client(self) -> Any:
        """懒创建的 OpenAI 兼容客户端。"""
        if self._client is None:
            from openai import OpenAI  # 延迟导入：骨架阶段不联网、CI 不初始化

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """一次 chat 调用（temperature=0），返回文本内容。"""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RealAdapterError("LLM 返回空内容")
        return content

    def extract(self, description: str) -> str:
        """查询结构化抽取（gen.prompts 第一段提示词）。"""
        return self.complete(
            EXTRACTION_SYSTEM_PROMPT,
            render_prompt(EXTRACTION_USER_TEMPLATE, description=description),
        )

    def entail_review(
        self, description: str, hits: Sequence[dict[str, Any]], feedback: str | None = None,
    ) -> str:
        """负面条款蕴含复核（gen.prompts 第二段提示词，条款块带标记与效力）。

        复核对象＝G01-G11 禁止用途载体（origin=negative-list），而非规则命中块——
        规则命中的条款关键词已命中、LLM 再评只做确认；真正的增量在关键词漏掉的
        禁止用途语义，故只喂 G01-G11 载体。
        """
        blocks = render_context_blocks(hits, origin=ORIGIN_NEGATIVE)
        user_prompt = render_prompt(
            NEGATIVE_ENTAILMENT_USER_TEMPLATE,
            description=description,
            clauses=blocks or "（本次没有待复核的负面条款块）",
        )
        if feedback:
            user_prompt += "\n\n" + feedback
        return self.complete(NEGATIVE_ENTAILMENT_SYSTEM_PROMPT, user_prompt)

    def generate(
        self,
        description: str,
        extraction_json: str,
        context_blocks: str,
        feedback: str | None,
    ) -> str:
        """①③④段生成（gen.prompts 第三段提示词；校验反馈作为重试提示追加）。"""
        user_prompt = render_prompt(
            GENERATION_USER_TEMPLATE,
            description=description,
            extraction_json=extraction_json,
            context_blocks=context_blocks or "（无检索上下文）",
        )
        if feedback:
            user_prompt += (
                "\n\n上一轮输出未通过确定性校验，请按以下反馈修正后重新输出完整 JSON：\n"
                + feedback
            )
        return self.complete(GENERATION_SYSTEM_PROMPT, user_prompt)


def render_context_blocks(
    hits: Sequence[dict[str, Any]],
    *,
    origin: str | None = None,
) -> str:
    """把命中 dict 列表拼成提示词上下文块（``[Cn]｜{validity}\n原文（逐字）：…``）。"""
    blocks: list[str] = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        if origin is not None and metadata.get("origin") != origin:
            continue
        marker = str(hit.get("chunk_id") or "")
        number = marker[2:-1] if marker.startswith("[C") else marker
        blocks.append(CONTEXT_BLOCK_FORMAT.format(
            n=number,
            validity=metadata.get("validity") or "时效不明",
            text=str(hit.get("text") or ""),
        ))
    return "\n\n".join(blocks)


# ================================================================ 真实组件


@dataclass(frozen=True)
class RetrievalPlan:
    """一次描述对应的完整命中计划（retrieve 与 rule_scan 共享，保证标记一致）。"""

    hits: tuple[dict[str, Any], ...]
    negative_hits: tuple[NegativeHit, ...]
    dense_hits: tuple[Hit, ...]
    rule_hits: tuple[RuleHit, ...]
    retrieval: RetrievalResult
    carrier_hits: tuple[dict[str, Any], ...] = ()
    """G01-G11 载体全集（②段蕴含复核的待审对象）。

    与 ``hits`` 的区别：``hits`` 只保证"每个**新**标记都有块"；这里保证"每个**载体**都有一条块"，
    即使它的条款已被检索命中或规则命中覆盖（那种情况复用同一标记、不重复追加进 hits）。
    这两条链路必须分开：载体若只靠 hits 的 origin 过滤取，凡是复用标记的都会被静默漏掉。
    """


class RealComponents:
    """真实检索＋真实规则扫描的组件集（LLM 默认 Mock，可注入 QwenLLM）。"""

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        *,
        retriever: Any = None,
        matcher: Any = None,
        llm: MockLLM | QwenLLM | None = None,
    ) -> None:
        """:param config_path: 本机 config.json（数据仓/索引/embedding 路径）
        :param retriever: 注入检索器（测试用）；None 时按配置装配真实 DenseRetriever
        :param matcher: 注入规则匹配器（测试用）；None 时从数据仓规则库装配
        :param llm: LLM 组件；None 用 :class:`MockLLM`
        """
        self._config_path = Path(config_path)
        self._config = _load_config(self._config_path)
        self._llm = llm if llm is not None else MockLLM()
        self._plans: dict[str, RetrievalPlan] = {}
        # 图的 retrieve 与 rule_scan 两个分支是**并发执行**的（LangGraph 线程池）。
        # 没有这把锁时两路会各自算一遍命中计划：既重复跑一次模型编码，
        # 更严重的是两路可能拿到不同批的标记（[Cn] 编号不一致 → ②段引用错位）。
        self._plan_lock = threading.Lock()

        building_retriever = retriever is None
        building_matcher = matcher is None
        if building_retriever or building_matcher:
            self._ensure_files()
        self._retriever = retriever if retriever is not None else _build_retriever(self._config)
        self._matcher = (
            matcher if matcher is not None else _build_matcher(self._config_path))
        if building_retriever:
            total = _safe_count(self._retriever)
            if total == 0:
                raise RealAdapterError(
                    "检索集合为空：persist_dir 可能指错目录（Chroma 会自动建空集合、"
                    "count 读成 0，不报错）。请先用 scripts/verify_index.py 核对索引。")

    # —— 就绪检查（文件级；报错带可操作提示）——
    def _ensure_files(self) -> None:
        index_section = self._config.get("index") or {}
        persist = _resolve_path(
            str(index_section.get("persist_dir") or ".chroma"), self._config_path)
        if not (persist / "chroma.sqlite3").is_file():
            raise RealAdapterError(
                f"未找到真实索引：{persist}/chroma.sqlite3 不存在。"
                "先把 kelier1 的 .chroma/ 解压到 config 的 persist_dir（索引不入 git）。")
        embedding = self._config.get("embedding") or {}
        model_dir = _resolve_path(
            str(embedding.get("model_dir") or "models/Qwen3-Embedding-0.6B"), self._config_path)
        if not (model_dir / "config.json").is_file():
            raise RealAdapterError(
                f"未找到 embedding 模型：{model_dir}/config.json 不存在。"
                "按手册从 ModelScope 下载 Qwen/Qwen3-Embedding-0.6B 到该目录（权重不入 git）。")

    # —— 计划（描述 → 命中＋②段）——
    def _negative_carriers(self) -> list[RuleRecord]:
        """现行有效的 G01-G11 禁止用途载体（含禁止性措辞，按原文去重）。"""
        records = [
            r for r in self._matcher.store.valid_records() if is_g01g11_carrier(r)]
        return dedup_carriers(records)

    def _plan(self, description: str) -> RetrievalPlan:
        """计算（或复用）一次描述的命中计划；加锁保证并发分支共享同一计划。"""
        key = (description or "").strip()
        with self._plan_lock:
            cached = self._plans.get(key)
            if cached is not None:
                return cached
            plan = self._compute_plan(key)
            self._plans[key] = plan
            return plan

    def _compute_plan(self, key: str) -> RetrievalPlan:
        """真正的计划计算（只在持锁期间调用，不自行读写缓存）。"""
        retrieval = (
            self._retriever.retrieve(key) if key else RetrievalResult(query="", hits=[]))
        match = (
            self._matcher.match(key) if key
            else MatchResult(hits=(), description_terms=()))

        hits: list[dict[str, Any]] = []
        dense_markers: dict[tuple[int, int], str] = {}
        for index, hit in enumerate(retrieval.hits, start=1):
            marker = f"[C{index}]"
            serial = hit.metadata.get("serial")
            clause_index = hit.metadata.get("clause_index")
            if isinstance(serial, int) and isinstance(clause_index, int) and clause_index >= 0:
                dense_markers[(serial, clause_index)] = marker
            hits.append(hit_to_orchestration_hit(hit, marker))

        negative_hits: list[NegativeHit] = []
        seen_negative: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        # 键带上条款原文：同一 (文件, 条款号) 下可能有多个款级子块（不同原文），
        # 只用 (文件, 条款号) 作键会让后一个静默共用前一个标记、整块丢失（issue #48）。
        text_markers: dict[tuple[int, str, str], str] = {}
        next_index = len(retrieval.hits) + 1
        for rule_hit in match.hits:
            rule = rule_hit.rule
            clause_index = clause_no_to_index(rule.clause_no)
            marker = (
                dense_markers.get((rule.doc_index, clause_index))
                if clause_index is not None else None
            )
            rule_key = (rule.doc_index, rule.clause_no, rule.clause_text)
            if marker is None:
                marker = text_markers.get(rule_key)
            if marker is None:
                marker = f"[C{next_index}]"
                next_index += 1
                text_markers[rule_key] = marker
                hits.append(rule_orchestration_hit(rule, marker))
            words = matched_word_pairs(rule_hit)
            signature = (marker, words)
            # 同一条款常被多条规则记录引用（如承接列举行各自成条），
            # 完全相同的（标记＋命中词）只保留一条，避免②段出现重复行。
            if signature in seen_negative:
                continue
            seen_negative.add(signature)
            negative_hits.append(NegativeHit(
                citation=marker,
                hit_type=HIT_RULE,
                matched_words=words,
            ))

        # G01-G11 禁止用途载体：给②段蕴含复核的待审条款（origin=negative-list，
        # 不参与①段生成）。含禁止性措辞＋按原文去重 → ~11-30 项，一次 LLM 调用。
        carrier_hits: list[dict[str, Any]] = []
        for carrier in self._negative_carriers():
            carrier_key = (carrier.doc_index, carrier.clause_no, carrier.clause_text)
            marker = text_markers.get(carrier_key)
            if marker is None:
                marker = f"[C{next_index}]"
                next_index += 1
                text_markers[carrier_key] = marker
                hits.append(rule_orchestration_hit(carrier, marker, origin=ORIGIN_NEGATIVE))
            # 复用标记（该条款已被检索/规则命中覆盖）时只进复核清单，不重复追加进 hits
            carrier_hits.append(
                rule_orchestration_hit(carrier, marker or "", origin=ORIGIN_NEGATIVE))

        plan = RetrievalPlan(
            hits=tuple(hits),
            negative_hits=tuple(negative_hits),
            dense_hits=tuple(retrieval.hits),
            rule_hits=tuple(match.hits),
            retrieval=retrieval,
            carrier_hits=tuple(carrier_hits),
        )
        logger.info(
            "命中计划：[%d 块稠密 + %d 条规则 + %d 项 G01-G11 载体]；%s",
            len(retrieval.hits), len(match.hits), len(carrier_hits), retrieval.summary())
        return plan

    # —— 编排图回调 ——
    def retrieve(self, description: str, extraction: Any) -> tuple[dict[str, Any], ...]:
        """编排图检索回调：真稠密检索（top-k，时效硬过滤默认开）。"""
        return self._plan(description).hits

    def rule_scan(self, description: str) -> tuple[NegativeHit, ...]:
        """编排图规则回调：真负面清单全量扫描（确定性，无 LLM）。"""
        return self._plan(description).negative_hits

    def _entail(
        self, description: str, hits: Sequence[dict[str, Any]], feedback: str | None = None,
    ) -> str:
        """②段蕴含复核：喂 **G01-G11 载体全集**（而不是"hits 里 origin=negative-list 的那些块"）。

        两者不等价：若某载体的条款已被稠密检索或规则命中覆盖，适配器会复用它已有的标记、
        不再追加新块，于是它不会以 origin=negative-list 出现在 hits 里——只按 hits 过滤就会把它
        静默漏掉（issue #48）。故这里显式取计划里的载体全集。
        """
        return self._llm.entail_review(
            description, self._plan(description).carrier_hits, feedback)

    def _generate(
        self,
        description: str,
        extraction_json: str,
        context_blocks: str,
        feedback: str | None,
    ) -> str:
        """生成回调：上下文块优先用调用方给的；为空时用本组件命中计划兜底。"""
        blocks = context_blocks or self.context_blocks(description)
        return self._llm.generate(description, extraction_json, blocks, feedback)

    def context_blocks(self, description: str) -> str:
        """按命中计划拼生成上下文（``[Cn]｜有效性\n原文（逐字）：…``）。

        只喂**检索命中**与**规则命中**：G01-G11 载体（``origin=negative-list``）只服务
        ②段蕴含复核，不进①③④段生成的提示词——否则禁止性条款会被当成「适用政策」喂给
        生成模型，既污染①段的语义口径，也白白放大上下文体积。
        """
        hits = [
            hit for hit in self._plan(description).hits
            if hit["metadata"].get("origin") != ORIGIN_NEGATIVE
        ]
        return render_context_blocks(hits)

    def components(self) -> OrchestrationComponents:
        """暴露编排图五回调（检索/规则为真，LLM 由注入决定）。"""
        return OrchestrationComponents(
            extract=self._llm.extract,
            retrieve=self.retrieve,
            rule_scan=self.rule_scan,
            entail_review=self._entail,
            generate=self._generate,
        )

    def retrieval_summary(self, description: str) -> str:
        """检索摘要（CLI 输出样例用）。"""
        return self._plan(description).retrieval.summary()

    def reset_plans(self) -> None:
        """清空命中计划缓存（每次 pipeline.run 前调用，避免跨描述复用）。"""
        with self._plan_lock:
            self._plans.clear()


class RealCheckupPipeline:
    """真实适配器管线：真检索＋真规则＋（默认 Mock）LLM，实现 CheckupPipeline 协议。"""

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        *,
        llm: MockLLM | QwenLLM | None = None,
        components: RealComponents | None = None,
    ) -> None:
        """装配管线；``components`` 可注入（测试用）。"""
        self.components = components or RealComponents(config_path, llm=llm)

    @property
    def name(self) -> str:
        """管线名（CI/接口层观察用）。"""
        llm = self.components._llm  # noqa: SLF001 - 同包内读取注入的 LLM 类型做命名
        return "real-retrieval-qwen-llm" if isinstance(llm, QwenLLM) else "real-retrieval-mock-llm"

    def run(self, description: str) -> PipelineResult:
        """输入项目描述 → 四段式报告（真检索＋真规则＋LLM 组件）。"""
        started = time.monotonic()
        self.components.reset_plans()
        graph = build_checkup_graph(self.components.components())
        final = graph.invoke({"description": description})
        elapsed = int((time.monotonic() - started) * 1000)
        return PipelineResult(
            report=final["report"],
            rendered_text=final["rendered_text"],
            degraded=final["degraded"],
            attempts=final["attempts"],
            failed_sections=final["failed_sections"],
            elapsed_ms=elapsed,
        )


# ================================================================ 装配辅助


def _build_retriever(config: Mapping[str, Any]) -> Any:
    """按配置装配真实 DenseRetriever（局部 import：不装 torch 也能 import 本模块）。"""
    from rag.retriever import build_retriever

    return build_retriever(config)


def _build_matcher(config_path: Path) -> NegRuleMatcher:
    """从数据仓规则库装配确定性匹配器。"""
    store = load_store_from_config(config_path)
    return NegRuleMatcher(store)


def _safe_count(retriever: Any) -> int:
    """取检索后端的候选块数；不支持的实现返回 -1（不阻断装配）。"""
    backend = getattr(retriever, "backend", None)
    counter = getattr(backend, "count", None)
    if not callable(counter):
        return -1
    try:
        return int(counter())
    except Exception:  # noqa: BLE001 - 诊断用途，失败不阻断装配
        logger.warning("读取集合块数失败（诊断忽略）", exc_info=True)
        return -1


# ================================================================ 命令行


def _read_cases(path: Path) -> list[str]:
    """读项目描述文件：一行一条，``#`` 开头忽略。"""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：真检索＋真规则＋Mock LLM 的端到端体检打印。"""
    parser = argparse.ArgumentParser(
        prog="python -m app.real_components",
        description="真实检索适配器端到端体检（真检索＋真规则；--qwen 换真 LLM）",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="本机配置文件")
    parser.add_argument(
        "--description", action="append", default=[],
        help="项目描述（可重复，用于多条样例）")
    parser.add_argument("--cases", default=None, help="项目描述文件：一行一条（# 注释）")
    parser.add_argument(
        "--qwen", action="store_true",
        help="用 qwen LLM 替换 Mock（需 .env 的 key）")
    parser.add_argument("--show-hits", action="store_true", help="打印检索/规则命中明细")
    parser.add_argument("--log-level", default="INFO", help="日志级别，缺省 INFO")
    args = parser.parse_args(argv)

    descriptions = list(args.description)
    if args.cases:
        descriptions.extend(_read_cases(Path(args.cases)))
    if not descriptions:
        parser.error("需要 --description 或 --cases 至少一个")

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(message)s",
    )

    config = _load_config(Path(args.config))
    llm = QwenLLM(config) if args.qwen else None
    pipeline = RealCheckupPipeline(args.config, llm=llm)
    logger.info("管线：%s（真检索＋真规则）", pipeline.name)

    for index, description in enumerate(descriptions, start=1):
        result = pipeline.run(description)
        logger.info("==== 案例 %d ====", index)
        if args.show_hits:
            plan = pipeline.components._plan(description)  # noqa: SLF001 - CLI 展示用
            logger.info("检索摘要：%s", plan.retrieval.summary())
            for hit in plan.hits:
                metadata = hit["metadata"]
                logger.info(
                    "  %s｜%s %s（%s，%s）",
                    hit["chunk_id"], metadata.get("doc_no") or "（无文号）",
                    metadata.get("clause_no") or "", metadata.get("validity") or "",
                    metadata.get("origin"))
            logger.info("②段规则命中：%d 条", len(plan.negative_hits))
        logger.info("%s", result.rendered_text)
        logger.info(
            "（耗时 %d ms · 生成尝试 %d 次 · 降级 %s）",
            result.elapsed_ms, result.attempts, result.degraded)
    return 0


if __name__ == "__main__":  # pragma: no cover - 入口
    raise SystemExit(main())
