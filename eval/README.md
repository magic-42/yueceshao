# eval/ —— 评测与消融（团队手册 3.4）

本目录承载评测跑批与消融实验。两份交付分两个 PR：

| PR | 分支 | 内容 | 验收 |
|---|---|---|---|
| 1 | `feat/eval-config` | 六组消融开关的配置 schema，集中布尔位 | 开关关掉的路径可被测试观测到 |
| 2 | `feat/eval-runner` | 回溯评测跑批＋三指标统计＋逐案例明细 | 10 条 smoke 跑通全流程出报告 |

## 消融开关（唯一来源：`eval/config.py`）

README 编码规范要求「全部消融开关集中在 `eval/` 的配置布尔位，不得散落硬编码」。
`AblationSwitches` 是这些开关的**唯一**定义处，业务模块不得自建常量或 `os.getenv` 旁路。

六组开关，语义统一为 **`True`＝启用该模块（默认，完整系统）**，`False`＝移除（消融组）：

| 开关 | 模块 |
|---|---|
| `temporal_filter` | 时序版本过滤：按查询时间剔除已废止条款 |
| `negative_hard_constraint` | 负面清单硬约束注入：命中即中断生成并给替代建议 |
| `rerank` | Cross-Encoder 重排：对召回结果做精排 |
| `hybrid_retrieval` | 混合检索：BM25 与向量双路召回（关闭后仅走向量） |
| `query_decomposition` | 查询分解：结构化抽取与术语归一化 |
| `citation_validator` | 引用校验器：标记存在性 / 上下文包含性 / 姿态禁语兜底 |

配置来源三选一，优先级由调用方决定：

```python
from eval.config import AblationSwitches

AblationSwitches.default()                                   # 完整系统
AblationSwitches.from_mapping({"rerank": False})             # 代码内指定
AblationSwitches.from_json_file("eval/ablation.json")        # 配置文件
AblationSwitches.from_env()                                  # YUECESHAO_ABLATION_RERANK=0
```

`eval/ablation.example.json` 是样例；真实 `ablation.json` 不入库（`.gitignore` 已挡
`eval/ablation.json`）。开关名写错、取值非 bool 一律**当场报错**，不静默忽略——消融配置
静默失效会让实验结论失真。这条严格性对三个入口一致：`from_mapping` / `from_json_file` /
`from_env` 遇到无法识别的开关名都抛 `ValueError`（`from_env` 会连同前缀扫一遍环境变量，
手滑写成 `YUECESHAO_ABLATION_RERANKK` 也会被抓住，不会因为多打一个 K 就静默放行）。

## 开关如何被观测（`eval/gating.py`）

给可消融的 stage 套 `@gated("rerank")`：

* 开关开启 → 正常调用真实实现；
* 开关关闭 → **不调用**真实实现（消融语义要求它缺席），并往当前 `SwitchTrace` 记一条
  bypass 事件（开关名 + 阶段名）。

```python
from eval.gating import ablation_context, gated

@gated("rerank", stage="retrieve.rerank")
def rerank(candidates): ...

with ablation_context(switches) as trace:
    rerank(cands)          # 开关关闭时不会执行，且 trace 留痕
assert trace.was_bypassed("rerank")
```

这一点就是 PR1 的验收依据：测试同时断言「实现函数调用次数为 0」与「trace 里有记录」，
见 `tests/test_eval_gating.py`。

## 跑批（`python -m eval.runner`）· PR2 落地

> 本 PR（`feat/eval-config`）只交付配置层与观测点。`eval/runner.py`、`eval/metrics.py`、
> `eval/pipeline.py`、`eval/dataset.py` 均在 **PR2（`feat/eval-runner`）** 落地；本 PR 的
> 分支上还没有这些文件，下面的命令在 PR2 合并后才可执行。

```bash
# 10 条 smoke，验证全流程出报告
python -m eval.runner --cases <数据仓>/审计公告/案例标注表.csv --limit 10 --out runs/smoke

# 全量 + 逐模块消融（1 组完整系统 + 6 组单消融 = 7 组对比矩阵）
python -m eval.runner --cases <数据仓>/审计公告/案例标注表.csv --ablate-all --out runs/ablation
```

产出 `metrics.json`（三指标 + 达标判定 + 元数据）、`details.csv`（逐案例明细）、
`ablation_matrix.csv`（多组对比，仅多组时生成）。

**数据不随仓库分发。** 标注表在团队数据仓（`审计公告/案例标注表.csv`，120 条），
红线第 4 条：审计公告是评测出题集，进检索索引＝答案泄露。`.gitignore` 已挡 `data/`
与 `runs/`。

## 三项指标口径（`eval/metrics.py`）· PR2 落地

手册 3.4 的验收线：**匹配 ≥85% / 召回 ≥80% / 引用错误 ≤5%**。三个词的定义：

1. **匹配率** `match_rate`：系统判定（`hit`/`miss`）与标注 `expected_verdict` 的一致率。
   分母只计已标注判定的案例；未标注的进明细并计入 `verdict_unlabeled`。
2. **召回率** `recall`：正例条款召回，micro 口径 `Σ|预测∩标注| / Σ|标注|`。
   只标了 `hit` 未给具体条款时退化为整案召回，明细 `recall_mode` 标 `case-level`。
3. **引用错误率** `citation_error_rate`：输出引用中**无法在 registry 核验**的比例。
   未提供 registry 时该指标为 `None`（N/A）——无法判断真假时不能默认它是对的。
   另单独统计 `forbidden_violations`：输出出现「合规 / 可以申报」类姿态禁语即违规
   （红线第 3 条）。**正向放行（合规、可以申报）与反向判定（违规、不能申报）同属禁语**
   ——两者都是系统在替主管部门下结论。

   > 免责声明必须剔除后再扫。渲染层强制注入的「本系统仅为申报前自查辅助参考，不构成合规
   > 判定……」本身含「合规」二字，直接扫会让接上真实管线后的每份报告都误报姿态违规。
   > `eval.metrics.strip_disclaimer()` 先按 :data:`REPORT_DISCLAIMER` 精确匹配、再按行丢弃
   > 含 :data:`DISCLAIMER_LINE_HINTS` 的免责声明行；正文里的真实违规不会被连带放过。

   registry 导出读入是**严格**的：空对象、空列表、列表元素缺 `ref_id`、不支持的形态
   一律抛 `RegistryError`（与 `DatasetError` 同等严厉）。静默建出空索引会让人看到
   引用错误率 100%，看起来像「系统全在编文号」，实际只是导出格式不对——这类假结果比
   报错危险得多。

**不可计算的指标一律 `None`，报告显示 N/A，绝不拿 0 或 1 冒充。** 有 N/A 项时
`all_pass` 为 `false`（不能靠缺失项过关），命令行退出码为 1。

## 当前状态：占位管线（PR2 落地）

rag / rules / gen 三个包目前只有空 `__init__.py`，真实管线尚未落地。`eval/pipeline.py`
定义了 `Pipeline` 协议（`predict(case) -> CasePrediction`），并提供一个 `NullPipeline`
占位实现，仅用于验证跑批链路与报告格式。报告元数据里 `pipeline=null`，
**它的数字没有实验意义**，接上真实管线后重跑才作数。

## 测试

```bash
python -m pytest tests/ -q
```
