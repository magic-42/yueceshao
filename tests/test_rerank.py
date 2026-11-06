"""rag.rerank 的单元测试。

对应《开发手册 v1.0》§3.1 PR#6 验收标准：

1. **Cross-Encoder 重排 k40→top8**——``test_rerank_respects_top_k_out`` 等；
2. **τ 不过门走降级路径有测试**——``test_gate_retries_with_larger_pool`` /
   ``test_gate_honest_after_exhausting_retries``；
3. **τ 阈值标定过程写入文档**——见 ``docs/rag-rerank-标定.md``，本文档只测机制。

模型用假打分器替换：CI 装不起 2.2GB 权重，且真实分数分布属标定工作，不是单元测试
该断言的东西。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from rag.rerank import (
    HONEST_EMPTY,
    CrossEncoderScorer,
    GateRunner,
    QualityGate,
    RerankedHit,
    Reranker,
    RerankerConfig,
    RerankResult,
    build_reranker,
    resolve_model_source,
    sigmoid,
)

# ================================================================ 测试替身


class FakeScorer:
    """按正文关键词给分的假 Cross-Encoder。

    ``scores`` 是 ``{关键词: logit}``；正文含哪个关键词就用哪个 logit，
    都不含则用 ``default``。这样测试可以精确控制重排顺序。
    """

    def __init__(self, scores: Mapping[str, float], default: float = -6.0) -> None:
        self.scores = dict(scores)
        self.default = default
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """按关键词匹配给 logit。"""
        self.calls.append((query, len(passages)))
        result = []
        for passage in passages:
            value = self.default
            for keyword, logit in self.scores.items():
                if keyword in passage:
                    value = logit
                    break
            result.append(value)
        return result


class BrokenScorer:
    """返回数量不对的打分器，用于验证契约检查。"""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """少返回一个分数。"""
        return [0.0] * max(0, len(passages) - 1)


def make_hit(
    chunk_id: str, text: str, doc_no: str = "粤农农规〔2023〕6号", clause: str = "第十条"
) -> Any:
    """构造一条带 ``text`` 的候选命中（结构对齐 Hit）。"""
    return type(
        "FakeHit",
        (),
        {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": {"doc_no": doc_no, "clause": clause},
            "doc_no": doc_no,
            "clause": clause,
            "citation": f"{doc_no} {clause}".strip(),
            "reference": f"{doc_no} {clause}".strip(),
            "as_dict": lambda self: {"chunk_id": self.chunk_id, "text": self.text},
        },
    )()


# ================================================================ sigmoid


def test_sigmoid_range_and_midpoint() -> None:
    """sigmoid 把 logit 压到 (0,1)，0 点对应 0.5。"""
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-50) < 1e-6
    assert sigmoid(50) >= 1.0 - 1e-9, "大正数应饱和到 1（float64 下可能就是 1.0）"


def test_sigmoid_matches_math_formula() -> None:
    """公式与 ``1/(1+e^-x)`` 一致（含负数分支）。"""
    for value in (-3.5, -0.2, 0.0, 0.7, 4.1):
        assert sigmoid(value) == pytest.approx(1.0 / (1.0 + math.exp(-value)))


def test_sigmoid_no_overflow_on_large_negative() -> None:
    """``exp(-x)`` 在 x 很负时会溢出，实现必须走另一分支。"""
    assert math.isfinite(sigmoid(-1000))


# ================================================================ 重排


def test_rerank_reorders_by_cross_score() -> None:
    """重排顺序由 Cross-Encoder 决定，不看原顺序。"""
    hits = [
        make_hit("1-0001", "产业园申报的一般性规定"),
        make_hit("1-0002", "驻镇帮镇扶村工作队的职责"),
        make_hit("1-0003", "联农带农机制的考核办法"),
    ]
    scorer = FakeScorer({"联农带农": 4.0, "驻镇帮镇扶村": 2.0, "产业园": 0.5})
    reranker = Reranker(scorer, RerankerConfig(top_k_out=8))

    result = reranker.rerank("联农带农怎么考核", hits)

    assert [hit.chunk_id for hit in result.hits] == ["1-0003", "1-0002", "1-0001"]
    assert result.hits[0].rerank_score == pytest.approx(sigmoid(4.0))


def test_rerank_records_source_rank_and_delta() -> None:
    """``source_rank`` 记原位置，``rank_delta`` 正数表示被提前。"""
    hits = [make_hit("1-0001", "产业园"), make_hit("1-0002", "联农带农")]
    reranker = Reranker(FakeScorer({"联农带农": 4.0, "产业园": 0.5}), RerankerConfig())

    result = reranker.rerank("联农带农", hits)

    top = result.hits[0]
    assert top.chunk_id == "1-0002"
    assert top.source_rank == 2
    assert top.rank == 1
    assert top.rank_delta == 1


def test_rerank_respects_top_k_out() -> None:
    """重排后截到 ``top_k_out``（手册 A7：k40→top8）。"""
    hits = [make_hit(f"1-000{index}", f"第{index}条 产业园申报") for index in range(1, 6)]
    config = RerankerConfig(top_k_in=40, top_k_out=3)

    result = Reranker(FakeScorer({}), config).rerank("产业园", hits)

    assert len(result.hits) == 3
    assert result.candidate_count == 5


def test_rerank_respects_top_k_in() -> None:
    """只把前 ``top_k_in`` 条送进模型——Cross-Encoder 很慢，不能全量跑。"""
    hits = [make_hit(f"1-000{index}", f"第{index}条") for index in range(1, 11)]
    scorer = FakeScorer({})
    config = RerankerConfig(top_k_in=4, top_k_out=10)

    Reranker(scorer, config).rerank("查询", hits)

    assert scorer.calls[0][1] == 4


def test_rerank_empty_input() -> None:
    """无候选时返回空结果并给出诚实说明，不抛异常。"""
    result = Reranker(FakeScorer({}), RerankerConfig()).rerank("产业园申报", [])

    assert result.hits == []
    assert result.gate.reason == "empty_candidates"
    assert HONEST_EMPTY in result.gate.message


def test_rerank_empty_query_reason_differs_from_empty_candidates() -> None:
    """空查询与空候选要分开报：前者是上游传错，后者才是"真没检索到"。"""
    query_empty = Reranker(FakeScorer({}), RerankerConfig()).rerank(
        "  ", [make_hit("1-0001", "正文")])
    no_hits = Reranker(FakeScorer({}), RerankerConfig()).rerank("产业园申报", [])

    assert query_empty.gate.reason == "empty_query"
    assert no_hits.gate.reason == "empty_candidates"


def test_rerank_empty_query_does_not_call_model() -> None:
    """空查询不打模型。"""
    scorer = FakeScorer({})

    result = Reranker(scorer, RerankerConfig()).rerank("   ", [make_hit("1-0001", "正文")])

    assert result.hits == []
    assert scorer.calls == []


def test_rerank_rejects_scorer_length_mismatch() -> None:
    """打分器返回数量不对时必须报错，不能静默截断配对。"""
    with pytest.raises(RuntimeError):
        Reranker(BrokenScorer(), RerankerConfig()).rerank(
            "查询", [make_hit("1-0001", "甲"), make_hit("1-0002", "乙")]
        )


def test_reranked_hit_serialization() -> None:
    """``as_dict`` 带上重排信息，评测脚本靠它落 JSONL。"""
    hits = [make_hit("1-0001", "产业园申报")]
    result = Reranker(FakeScorer({"产业园": 3.0}), RerankerConfig()).rerank("产业园", hits)

    payload = result.hits[0].as_dict()

    assert payload["chunk_id"] == "1-0001"
    assert payload["rerank_score"] > 0.9
    assert payload["rerank_logit"] == pytest.approx(3.0)
    assert payload["rank"] == 1


# ================================================================ 质量门


def test_gate_passes_above_tau() -> None:
    """最高分过阈值即通过。"""
    hits = [make_hit("1-0001", "联农带农")]
    result = Reranker(FakeScorer({"联农带农": 1.0}), RerankerConfig(tau=0.5)).rerank(
        "联农带农怎么考核", hits
    )

    decision = QualityGate(RerankerConfig(tau=0.5)).evaluate(result)

    assert decision.passed
    assert decision.reason == "pass"
    assert decision.message == ""


def test_gate_blocks_below_tau() -> None:
    """最高分不过阈值即拦截，并给出"未覆盖"的诚实说明。"""
    hits = [make_hit("1-0001", "园区绿化养护")]
    config = RerankerConfig(tau=0.5)
    result = Reranker(FakeScorer({}, default=-6.0), config).rerank("氢能汽车补贴", hits)

    decision = QualityGate(config).evaluate(result)

    assert not decision.passed
    assert decision.reason == "below_tau"
    assert "未覆盖" in decision.message


def test_gate_message_has_no_conclusive_wording() -> None:
    """红线③：门控文案不得出现"合规""可以申报"类结论。"""
    hits = [make_hit("1-0001", "无关内容")]
    config = RerankerConfig(tau=0.9)
    result = Reranker(FakeScorer({}, default=-3.0), config).rerank("项目能不能做", hits)

    message = QualityGate(config).evaluate(result).message

    for banned in ("合规", "可以申报", "建议申报", "符合条件", "通过审核"):
        assert banned not in message, f"门控文案出现结论性措辞：{banned}"


def test_gate_empty_result_is_blocked() -> None:
    """无候选也算不过门——否则"没检索到"会被当成"通过"。"""
    decision = QualityGate(RerankerConfig(tau=0.3)).evaluate(
        Reranker(FakeScorer({}), RerankerConfig()).rerank("查询", [])
    )

    assert not decision.passed
    assert decision.reason == "empty_candidates"


def test_gate_blocks_nan_score() -> None:
    """红线：非有限分数必须按不过门处理。

    ``nan < tau`` 在 IEEE 语义下恒为 False，不显式收口的话质量门会被判成"通过"——
    这是唯一一条不报错、不改配置、却让闸门失效的路径。
    """
    class NanScorer:
        def score(self, query: str, passages: Sequence[str]) -> list[float]:
            return [float("nan") for _ in passages]

    config = RerankerConfig(tau=0.55)
    result = Reranker(NanScorer(), config).rerank("随便问问", [make_hit("1-0001", "正文")])

    assert math.isfinite(result.hits[0].rerank_score)
    decision = QualityGate(config).evaluate(result)
    assert not decision.passed
    assert decision.reason == "below_tau"


def test_gate_blocks_non_finite_top_score_from_hand_built_result() -> None:
    """手写构造的 ``RerankResult`` 绕过重排层时，门控自己也要挡住 NaN。"""
    config = RerankerConfig(tau=0.55)
    hit = RerankedHit(
        hit=make_hit("1-0001", "正文"), rerank_score=float("nan"), logit=float("nan"),
        rank=1, source_rank=1,
    )

    decision = QualityGate(config).evaluate(RerankResult(query="查询", hits=[hit]))

    assert not decision.passed
    assert decision.reason == "below_tau"


def test_result_defaults_to_ungated() -> None:
    """手写构造的结果不得天然"已通过体检"（fail-closed 默认值）。"""
    result = RerankResult(query="查询", hits=[RerankedHit(
        hit=make_hit("1-0001", "正文"), rerank_score=0.99, logit=3.0, rank=1, source_rank=1,
    )])

    assert not result.gate.passed
    assert result.gate.reason == "ungated"
    assert result.citations() == []
    assert result.doc_nos() == []


def test_citations_blocked_when_gate_not_passed() -> None:
    """未过门时不得把条款当引用交出去——那等于绕过质量门（红线③）。"""
    config = RerankerConfig(tau=0.9)
    result = Reranker(FakeScorer({}, default=-3.0), config).rerank(
        "氢能汽车补贴", [make_hit("1-0001", "无关内容")]
    )

    gate = QualityGate(config)
    assert not gate.evaluate(result).passed
    assert result.hits, "诊断用的 hits 仍要保留"
    assert result.citations() == []

    result.gate = gate.evaluate(result)
    assert result.citations() == []


def test_gate_disabled_always_passes() -> None:
    """消融对照：关掉门控后即使分数极低也放行。"""
    config = RerankerConfig(tau=0.99, enable_gate=False)
    result = Reranker(FakeScorer({}, default=-6.0), config).rerank(
        "氢能汽车补贴", [make_hit("1-0001", "无关内容")]
    )

    decision = QualityGate(config).evaluate(result)
    result.gate = decision

    assert decision.passed
    assert decision.reason == "gate_disabled", "必须与真通过区分开"
    assert decision.tau == 0.99, "阈值仍要如实报告，便于识别这是消融组"
    assert "≥" not in result.summary(), "消融组不能打出最高分 ≥ τ 的假不等式"


def test_rerank_only_result_summary_has_no_inequality() -> None:
    """只重排没过门时，摘要不得声称"≥ τ"。"""
    result = Reranker(FakeScorer({"产业园": 3.0}), RerankerConfig()).rerank(
        "产业园", [make_hit("1-0001", "产业园")]
    )

    assert "未施加质量门" in result.summary()
    assert "≥" not in result.summary()


def test_doc_nos_excludes_placeholder_doc_no() -> None:
    """红线①：占位文号不得出现在文号列表里。"""
    hit = make_hit("1-0001", "正文", doc_no="无（申报通知未载文号）", clause="第四条")
    config = RerankerConfig()
    result = Reranker(FakeScorer({"正文": 3.0}), config).rerank("正文", [hit])
    result.gate = QualityGate(config).evaluate(result)

    assert result.gate.passed
    assert result.doc_nos() == []
    assert result.citations() == [], "占位文号拼进引用位就是伪造（红线①）"


def test_reranked_hit_forwards_is_current() -> None:
    """``RerankedHit`` 要转发现行有效标记，否则下游没法复核红线③。"""
    hit = type(
        "FakeHit",
        (),
        {
            "chunk_id": "1-0001",
            "text": "正文",
            "metadata": {"is_current": True},
            "is_current": True,
            "doc_no": "粤农农规〔2023〕6号",
            "clause": "第十条",
            "citation": "粤农农规〔2023〕6号 第十条",
            "reference": "粤农农规〔2023〕6号 第十条",
        },
    )()

    result = Reranker(FakeScorer({"正文": 3.0}), RerankerConfig()).rerank("正文", [hit])

    assert result.hits[0].is_current is True
    assert result.hits[0].has_doc_no is True


# ================================================================ 降级重试


def test_gate_retries_with_larger_pool() -> None:
    """不过门时放大候选池重试（降级路径），且时效条件不受影响。

    这里用"候选池越大才出现的块"来模拟：第一次 40 条里没有能答的，
    放大到 80 条才出现高分块。
    """
    weak = [make_hit(f"9-{index:04d}", "无关内容") for index in range(40)]
    strong = make_hit("1-0001", "联农带农机制考核")
    fetched: list[int] = []

    def fetch(size: int) -> Sequence[Any]:
        fetched.append(size)
        return (weak + [strong])[:size]

    config = RerankerConfig(top_k_in=40, top_k_out=3, tau=0.5, max_attempts=2)
    runner = GateRunner(
        Reranker(FakeScorer({"联农带农": 3.0}, default=-8.0), config),
        QualityGate(config),
        config,
    )

    result = runner.run("联农带农怎么考核", fetch)

    assert fetched == [40, 80], "重试必须放大候选池"
    assert result.attempts == 2
    assert result.gate.passed
    assert result.hits[0].chunk_id == "1-0001"


def test_gate_honest_after_exhausting_retries() -> None:
    """重试用尽仍不过门 → 诚实输出，且不伪造依据。"""
    config = RerankerConfig(top_k_in=5, top_k_out=3, tau=0.9, max_attempts=2)
    runner = GateRunner(
        Reranker(FakeScorer({}, default=-8.0), config),
        QualityGate(config),
        config,
    )

    result = runner.run("氢能汽车补贴", lambda size: [make_hit("1-0001", "无关内容")])

    assert result.attempts == 2
    assert not result.gate.passed
    assert result.gate.reason == "below_tau"
    assert "未覆盖" in result.gate.message
    assert result.hits, "命中仍保留供诊断，但下游不得据此输出结论"


def test_gate_passes_on_first_attempt_without_retry() -> None:
    """第一次就过门时不重试，省一次模型前向。"""
    config = RerankerConfig(top_k_in=10, tau=0.3, max_attempts=3)
    fetched: list[int] = []

    def fetch(size: int) -> Sequence[Any]:
        fetched.append(size)
        return [make_hit("1-0001", "联农带农机制")]

    runner = GateRunner(
        Reranker(FakeScorer({"联农带农": 3.0}), config), QualityGate(config), config
    )
    result = runner.run("联农带农", fetch)

    assert fetched == [10]
    assert result.attempts == 1
    assert result.gate.passed


def test_gate_max_attempts_one_disables_retry() -> None:
    """``max_attempts=1`` 时不重试，直接诚实输出。"""
    config = RerankerConfig(top_k_in=5, tau=0.9, max_attempts=1)
    fetched: list[int] = []

    def fetch(size: int) -> Sequence[Any]:
        fetched.append(size)
        return [make_hit("1-0001", "无关")]

    result = GateRunner(
        Reranker(FakeScorer({}, default=-5.0), config), QualityGate(config), config
    ).run("查询", fetch)

    assert fetched == [5]
    assert not result.gate.passed


def test_gate_runner_summary_reports_gate_state() -> None:
    """摘要必须写明门控结论，便于 CLI 与日志排查。"""
    config = RerankerConfig(top_k_in=5, tau=0.9, max_attempts=1)
    result = GateRunner(
        Reranker(FakeScorer({}, default=-5.0), config), QualityGate(config), config
    ).run("查询", lambda size: [make_hit("1-0001", "无关")])

    assert "未过质量门" in result.summary()


def test_retry_stops_when_candidate_pool_is_saturated(caplog: pytest.LogCaptureFixture) -> None:
    """候选池比请求数小时，再放大也是重复喂同一批，应尽早停手。"""
    config = RerankerConfig(top_k_in=4, tau=0.99, max_attempts=3, retry_factor=2)
    requested: list[int] = []

    def fetch(size: int) -> list[Any]:
        requested.append(size)
        return [make_hit("1-0001", "无关内容")]  # 池子里就一条

    runner = GateRunner(Reranker(FakeScorer({}, default=-6.0), config), QualityGate(config), config)
    with caplog.at_level(logging.INFO):
        result = runner.run("查询", fetch)

    assert not result.gate.passed
    assert len(requested) == 2, "第二轮发现池子没变大就该停，不必跑满 max_attempts"
    assert "饱和" in caplog.text


# ================================================================ 配置与装配


def test_config_from_config() -> None:
    """配置段读取：缺字段回落默认值。"""
    default = RerankerConfig.from_config(None)
    assert default.top_k_in == 40
    assert default.top_k_out == 8
    assert default.enable_gate is True

    custom = RerankerConfig.from_config(
        {"reranker": {"tau": 0.42, "top_k_out": 5, "max_attempts": 3}}
    )
    assert custom.tau == 0.42
    assert custom.top_k_out == 5
    assert custom.max_attempts == 3
    assert custom.top_k_in == 40


def test_config_rejects_zero_attempts() -> None:
    """``max_attempts=0`` 会让流程一次都不跑，必须收敛到至少 1。"""
    assert RerankerConfig.from_config({"reranker": {"max_attempts": 0}}).max_attempts == 1


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"tau": 0}, "tau"),
        ({"tau": 1.0}, "tau"),
        ({"tau": -0.3}, "tau"),
        ({"top_k_in": 0}, "top_k_in"),
        ({"top_k_in": -1}, "top_k_in"),
        ({"top_k_out": 0}, "top_k_out"),
        ({"max_length": 0}, "max_length"),
        ({"retry_factor": 1}, "retry_factor"),
        ({"enable_gate": "false"}, "enable_gate"),
        ({"enable_gate": ""}, "enable_gate"),
    ],
)
def test_config_rejects_values_that_silently_break_the_gate(
    payload: dict[str, Any], field: str
) -> None:
    """每一个越界值都能把质量门悄悄关掉，所以必须在构造时报错而不是静默兜底。

    典型危险值：``tau=0`` 让门恒过、``top_k_out=0`` 让每次查询都报"未覆盖"、
    ``retry_factor=1`` 让降级重试拿同一批候选、``enable_gate=""`` 用 ``bool()``
    会被判成 False（安全开关 fail-open）。
    """
    with pytest.raises(ValueError, match=field):
        RerankerConfig.from_config({"reranker": payload})


def test_config_accepts_explicit_values_without_falling_back() -> None:
    """显式写的合法值要真的生效，不能被"默认值兜底"吞掉。"""
    config = RerankerConfig.from_config(
        {"reranker": {"tau": 0.4, "top_k_in": 20, "retry_factor": 3, "enable_gate": False}}
    )

    assert (config.tau, config.top_k_in, config.retry_factor) == (0.4, 20, 3)
    assert config.enable_gate is False


def test_resolve_model_source_prefers_existing_dir(tmp_path: Any) -> None:
    """本地权重目录存在时直接用，不走 Hub。"""
    assert resolve_model_source(RerankerConfig(model_dir=str(tmp_path))) == str(tmp_path)


def test_resolve_model_source_blank_falls_back_to_hub() -> None:
    """``model_dir`` 留空才允许走 Hub，两个打分器共用同一套规则。"""
    resolved = resolve_model_source(RerankerConfig(model_dir="   "))

    assert resolved == "BAAI/bge-reranker-v2-m3"


def test_resolve_model_source_missing_dir_raises(tmp_path: Any) -> None:
    """目录不存在必须报错，不能静默退回 Hub 去下载 2.2GB。"""
    with pytest.raises(FileNotFoundError, match="重排权重目录不存在"):
        resolve_model_source(RerankerConfig(model_dir=str(tmp_path / "不存在的目录")))


def test_build_reranker_uses_given_scorer() -> None:
    """显式传入打分器时不再加载真实模型（CI 靠这条避免拉 2.2GB 权重）。"""
    scorer = FakeScorer({})
    reranker = build_reranker({"reranker": {"tau": 0.4}}, scorer=scorer)

    assert reranker.scorer is scorer
    assert reranker.config.tau == 0.4


def test_cross_encoder_scorer_is_lazy() -> None:
    """构造打分器不加载权重（导入与构造阶段不得碰 torch）。"""
    scorer = CrossEncoderScorer(RerankerConfig(model_dir="models/不存在的目录"))

    assert scorer._model is None  # noqa: SLF001 - 断言懒加载状态


def test_reranked_hit_rank_delta_zero_when_unchanged() -> None:
    """顺序没变时 ``rank_delta`` 为 0。"""
    hit = make_hit("1-0001", "产业园")
    result = Reranker(FakeScorer({"产业园": 1.0}), RerankerConfig()).rerank(
        "产业园", [hit]
    )

    assert result.hits[0].rank_delta == 0
    assert isinstance(result.hits[0], RerankedHit)
