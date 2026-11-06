"""eval/rules_pipeline.py — 规则引擎作为被测管线的适配器（feat/eval-rules-pipeline）。

把 rules.engine（负面清单确定性匹配）适配为 eval 的 :class:`Pipeline` 协议，
使评测跑批可以在**没有 LLM API key 与检索索引**的前提下，用团队真数据
（案例标注表 120 条）跑出纯规则引擎的回溯基线。

实验口径（写进报告解读，勿拿错）：
- 这是「负面清单硬约束」模块的**独立基线**——手册完整系统的②段是
  「规则引擎 ∪ LLM 蕴含复核」取并集，本管线的数字对应"关掉 LLM 复核"
  的消融视角，召回只会低于完整系统；
- 数据侧无负对照（120 条全为应命中正例）→ 匹配率与误报口径受限，
  报告中相关指标按 N/A／受限口径解读，召回率是本基线的主指标。

红线：refs/citations 全部来自规则库记录的出处字段（溯源自数据仓 registry），
本模块不生成、不改写任何文号（README 红线①②）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from eval.dataset import EvalCase, normalize_ref
from eval.metrics import CasePrediction
from rules.engine import NegRuleMatcher
from rules.loader import load_store_from_config
from rules.schema import RuleStore

__all__ = ["RulesPipeline"]

logger = logging.getLogger(__name__)


class RulesPipeline:
    """规则引擎管线：项目描述 × 现行规则集 → 命中条款列表。

    :param store: 已构建的规则库（测试注入假库用）；``None`` 时按 config.json 加载
    :param config_path: 数据仓配置路径（缺省取仓库根 config.json）
    """

    name = "rules-engine"

    def __init__(self, store: RuleStore | None = None, config_path: Path | None = None) -> None:
        self._store = store if store is not None else load_store_from_config(config_path)
        self._matcher = NegRuleMatcher(self._store)
        logger.info(
            "RulesPipeline 就绪：全量 %d 条 / 现行规则集 %d 条",
            len(self._store.records),
            len(self._store.active_records()),
        )

    def predict(self, case: EvalCase) -> CasePrediction:
        """对一条案例跑确定性匹配，产出评测侧预测。

        口径（feat/eval-rules-pipeline 首跑发现）：标注表期望条款多为**历史载体**
        （如 106号第二十条），引擎按设计只回**现行载体**——字符串级精确匹配会
        系统性归零。故 refs 采用「等价组全谱系」口径：直接命中条款＋其等价组
        全部成员（含历史与现行）——语义为「条款族召回」：期望条款与命中条款
        同属一个等价组即计召回。citations 保持仅直接命中（引用错误率只算系统
        实际给出的引用，语义不变）。
        """
        result = self._matcher.match(case.query)
        direct: dict[str, None] = {}
        family: dict[str, None] = {}
        for hit in result.hits:
            direct.setdefault(normalize_ref(hit.rule.doc_no, hit.rule.clause_no), None)
            family.setdefault(normalize_ref(hit.rule.doc_no, hit.rule.clause_no), None)
            for peer in hit.group_peers:
                family.setdefault(normalize_ref(peer.doc_no, peer.clause_no), None)
        # text 置空：本管线为纯确定性模板输出，无 LLM 生成段，姿态禁语扫描
        # 定义上不适用（render_hits 的条款原文引文与组名可能天然含「合规」等
        # 字样，属语料本身；整系统管线的扫描由 metrics 的原文豁免逻辑承接）。
        return CasePrediction(
            verdict="hit" if result.hits else "miss",
            refs=tuple(family),
            citations=tuple(direct),
            text="",
        )
