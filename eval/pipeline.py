"""被测管线的接入协议（3.4 评测与消融 · PR2 · feat/eval-runner）。

评测跑批不关心管线内部怎么实现，只要求它满足 :class:`Pipeline`：吃一条
:class:`eval.dataset.EvalCase`，吐一个 :class:`eval.metrics.CasePrediction`。

真实的 rag / rules / gen 管线尚未落地（仓库里目前只有空 ``__init__.py``），因此本模块
只定义接口，并提供一个**明确的占位实现** :class:`NullPipeline`——它不产生任何政策引用，
只用于让跑批脚本与报告格式先跑通。占位实现在报告里会被标记 ``pipeline=null``，
不要拿它的数字当实验结论。
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from eval.dataset import EvalCase
from eval.metrics import CasePrediction

__all__ = ["Pipeline", "NullPipeline"]

logger = logging.getLogger(__name__)


@runtime_checkable
class Pipeline(Protocol):
    """评测侧要求的最小管线接口。

    实现方（未来的 ``app`` 或组合脚本）需保证：
    * ``verdict`` 只用 ``hit`` / ``miss``；
    * ``refs`` / ``citations`` 里的标识来自 registry 已核验条目，不得手写文号（红线第 1 条）；
    * ``text`` 是最终报告全文，供姿态禁语扫描（红线第 3 条）。照实填全文即可：渲染层强制
      注入的免责声明（「不构成合规判定」）会在扫描前被 :func:`eval.metrics.strip_disclaimer`
      剔除，不会因此被误判为姿态违规。
    """

    name: str

    def predict(self, case: EvalCase) -> CasePrediction:
        """对单条案例产出预测结果。"""
        ...


class NullPipeline:
    """占位管线：全部判 ``miss``、不产出引用。

    用途只有一个——在真实管线落地前验证跑批链路、指标计算与报告输出。它的结果
    **没有实验意义**，报告元数据里会明确标注 ``pipeline=null``。
    """

    name = "null"

    def predict(self, case: EvalCase) -> CasePrediction:
        return CasePrediction(verdict="miss", refs=(), citations=(), text="")
