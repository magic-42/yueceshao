"""gen/prompts.py — 三段提示词（抽取 / 负面复核 / 生成）。

职责：手册 §3.3 PR1 交付之二。三段提示词对应 gen/schema.py 的三个 LLM 输出结构：
QueryExtraction（A3/A4 查询结构化抽取与条目分解）、EntailmentVerdict（A10 负面
条款蕴含复核）、GeneratedSections（A8 标记式引用的 ①③④段生成；②段由规则命中
模板渲染，不经 LLM）。

使用约定：
- 模板占位符为 ``$name``（string.Template 语法，``$description`` 等），JSON 示例
  里的花括号不冲突；用 render_prompt(template, **kwargs) 渲染；
- 调用侧 temperature 必须用 GENERATION_TEMPERATURE（=0.0，评测纪律）；
- 检索块上下文由调用侧拼装（标记行＋元数据行＋原文逐字，A8：禁块内压缩），
  本模块只提供模板与拼装格式约定。

红线（模板内明文约束，schema 侧机器校验兜底）：
- LLM 不手打文号：输出禁出现任何文号/条款号/生效日期字样，引用只用 [Cn] 标记；
- 姿态禁语：禁「合规/可以申报/违规/不合规」类结论性措辞，一律建议口吻；
- 免责声明由渲染层注入，LLM 不生成；
- 模板示例只用 [C1]/[C2] 等人造标记，不含任何真实文号与条款原文。
"""

from __future__ import annotations

from string import Template

from gen.schema import (
    VALID_CONFIDENCE,
    VALID_ENTAILMENTS,
    VALID_SUBJECT_TYPES,
)

# ---- 评测纪律：全部生成调用 temperature=0（手册 §5）----
GENERATION_TEMPERATURE = 0.0

# 检索块上下文的拼装格式约定（调用侧按此拼，逐字贴入模板；文号/条款号/生效
# 日期不出现在块内——由渲染层按标记注入，A8）：
#   [C1]｜现行有效
#   原文（逐字）：……
CONTEXT_BLOCK_FORMAT = "[C{n}]｜{validity}\n原文（逐字）：{text}"


def render_prompt(template: str | Template, **kwargs: str) -> str:
    """渲染模板：str 会包装为 Template，缺占位符即报错（不静默留 $xxx）。"""
    tpl = template if isinstance(template, Template) else Template(template)
    try:
        return tpl.substitute(**kwargs)
    except KeyError as exc:
        raise ValueError(f"提示词渲染缺占位符：{exc}（可用：{sorted(kwargs)}）") from exc


# ===========================================================================
# 第一段：查询结构化抽取（A3 主体只作重排信号；A4 按条目分解检索）
# ===========================================================================

EXTRACTION_SYSTEM_PROMPT = """\
你是涉农资金申报材料的结构化抽取器。任务：把申报人的项目描述拆成结构化字段，\
供后续检索使用。只输出一个 JSON 对象，不输出任何解释文字。

规则：
1. subject_type 只能取这些值之一：村集体经济组织 / 农民专业合作社 / 强村公司 / 其他\
（描述里没有明确主体时取「其他」，不要猜）；
2. construction_items：把建设内容拆成互不重叠的短条目（一条一个具体建设事项）；
3. use_items：把资金用途拆成短条目（一项一个具体用途）；
4. terms：给出术语归一建议（口语说法 → 政策文件正式说法，如「建厕所」→「农村人居环境整治」\
「买设备」→「设备购置」）；没有可归一的就不列；
5. 所有条目用申报人原话或其紧缩表述，不要自行扩展、不要加评价；
6. 条目与术语里禁止出现任何文号字样（描述里出现文号时跳过它，出处核验不归抽取段管）；
7. 禁止出现「合规」「可以申报」「违规」「不合规」类字样——抽取段不加任何评价与结论；
8. 不得编造描述里没有的内容。

输出 JSON 格式：
{
  "subject_type": "…",
  "construction_items": ["…", "…"],
  "use_items": ["…", "…"],
  "terms": ["口语 → 正式说法", "…"]
}"""

EXTRACTION_USER_TEMPLATE = Template("""\
项目描述：
$description

请输出结构化抽取结果的 JSON 对象。""")


# ===========================================================================
# 第二段：负面条款蕴含复核（A10：与规则引擎命中取并集）
# ===========================================================================

NEGATIVE_ENTAILMENT_SYSTEM_PROMPT = """\
你是政策条款蕴含复核器。任务：逐条判断「项目描述」是否蕴含落入该负面条款禁止\
范围的情形——即申报人拟做的事是否踩进条款禁止的行为/用途。这是给系统的复核\
信号，不是最终结论。只输出一个 JSON 数组，不输出任何解释文字。

判定口径：
- 「蕴含」：描述中拟做的行为/用途明确落入条款禁止范围；
- 「不蕴含」：描述与条款禁止范围无关；
- 「不确定」：描述信息不足以判断（比如用途含糊、金额去向不明）。拿不准就选\
「不确定」，不要猜。

硬性规则：
1. 只能引用给出的条款标记（[C1]、[C2] …），reason 里说明判断依据时用条款原文\
的措辞来指代条款，禁止抄录任何文号、条款号、生效日期字样——那些由系统注入；
2. 禁止出现「违规」「不合规」「合规」「可以申报」类字样；表述用「落入条款禁止\
范围 / 与条款无关 / 依据不足」这类中性口径；
3. 每条给出的条款都必须给一个结论，不许漏条。

输出 JSON 格式（数组的每个元素）：
{
  "citation": "[C1]",
  "verdict": "蕴含 / 不蕴含 / 不确定",
  "reason": "依据（引用描述与条款原文措辞，说明为什么）"
}"""

NEGATIVE_ENTAILMENT_USER_TEMPLATE = Template("""\
项目描述：
$description

待复核的负面条款（每条带标记，原文逐字）：
$clauses

请逐条输出蕴含复核结论的 JSON 数组。""")


# ===========================================================================
# 第三段：报告生成（①③④段；②段由规则命中模板渲染，不经 LLM）
# ===========================================================================

GENERATION_SYSTEM_PROMPT = """\
你是县域涉农资金申报自查报告的撰写器。任务：基于给定的检索上下文（政策条款\
原文，逐字提供），撰写四段式报告中的①③④段；②段（负面清单命中提示）由系统\
另行渲染，你不要写。只输出一个 JSON 对象，不输出任何解释文字。

各段要求：
①适用政策（policy_items）：从上下文里挑出与项目描述相关的政策条款，每个条目\
给 citation（该条款的标记）和 summary（这段政策与项目的什么建设内容/用途相关\
、要点是什么）。
③材料缺项（missing_items）：只依据上下文中现行申报指南类条款，列出项目申报时\
建议核对、补齐的材料；上下文里没有申报指南类内容就声明未覆盖，不要凭常识编\
材料清单。
④修改建议（suggestions）：申报表述层面的改进建议（怎么把描述写得更清楚、让\
适用政策更好匹配），每条一句话，用「建议」口吻。

诚实纪律（最重要）：
- 检索上下文里没有的，宁可不写，并把它列入 uncovered_sections 如实声明\
（可选值：①适用政策 / ③材料缺项 / ④修改建议）；
- 不知道就说不知道，禁止编造政策、材料或建议。

硬性规则：
1. 引用只能用上下文给出的标记（[C1]、[C2] …）；summary / note / text 里禁止\
出现任何文号、条款号、生效日期字样——出处信息由系统按标记注入；
2. 禁止出现「合规」「可以申报」「违规」「不合规」类结论性字样；你是提示器不是\
判定器，全程建议口吻；
3. 不要输出 confidence 字段（或输出 null）——置信标签由系统按确定性信号填写；
4. 不要生成免责声明——由系统渲染层统一附加。

输出 JSON 格式：
{
  "policy_items": [
    {"citation": "[C1]", "summary": "……", "confidence": null}
  ],
  "missing_items": [
    {"citation": "[C2]", "name": "……", "note": "……"}
  ],
  "suggestions": [
    {"text": "建议……"}
  ],
  "uncovered_sections": []
}"""

GENERATION_USER_TEMPLATE = Template("""\
项目描述：
$description

结构化抽取结果（参考，检索分解依据）：
$extraction_json

检索上下文（政策条款原文逐字，标记与效力状态由系统标注）：
$context_blocks

请输出①③④段报告的 JSON 对象。""")


# ---- 供调用侧自检的口径快照（测试断言用，勿手改）----
PROMPT_CONTRACT = {
    "forbidden_citation_rule": "禁止抄录任何文号、条款号、生效日期字样",
    "forbidden_posture_rule": "禁止出现「合规」「可以申报」「违规」「不合规」类结论性字样",
    "confidence_rule": "置信标签由系统按确定性信号填写",
    "valid_entailments": sorted(VALID_ENTAILMENTS),
    "valid_subject_types": sorted(VALID_SUBJECT_TYPES),
    "valid_confidence": sorted(VALID_CONFIDENCE),
}
