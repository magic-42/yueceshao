/* ============================================================
   粤策哨 · 静态演示版 · 交互逻辑
   纯前端：无后端、无请求、无密钥；内容不出本机。
   ============================================================ */

(function () {
  "use strict";

  /* ---------- 元素引用 ---------- */
  const form = document.getElementById("check-form");
  const input = document.getElementById("project-input");
  const checkBtn = document.getElementById("check-btn");
  const randomBtn = document.getElementById("random-btn");
  const clearBtn = document.getElementById("clear-btn");
  const charCount = document.getElementById("char-count");
  const caseList = document.getElementById("case-list");
  const notice = document.getElementById("notice");
  const noticeBack = document.getElementById("notice-back");
  const reportWrap = document.getElementById("report-wrap");
  const reportBody = document.getElementById("report-body");
  const analyzePanel = document.getElementById("analyze");
  const analyzeSteps = document.getElementById("analyze-steps");

  const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let activeCaseId = null;
  let analyzeTimer = null;
  let analyzeFallbackTimer = null;

  /* ---------- 工具函数 ---------- */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null && text !== "") node.textContent = text;
    return node;
  }

  function normalize(text) {
    return text.replace(/[\s\u3000，。；：、！？""''（）()《》【】\-—,.;:!?'"‘’]/g, "");
  }

  /* 数字滚动计数（尊重减少动效偏好，直接落在终值） */
  function animateCount(node, target) {
    if (REDUCED_MOTION || target <= 0) {
      node.textContent = String(target);
      return;
    }
    const start = performance.now();
    const duration = 600;
    let settled = false;
    function finish() {
      if (settled) return;
      settled = true;
      node.textContent = String(target);
    }
    function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      node.textContent = String(Math.round(target * (1 - Math.pow(1 - p, 3))));
      if (p < 1) {
        requestAnimationFrame(tick);
      } else {
        finish();
      }
    }
    requestAnimationFrame(tick);
    setTimeout(finish, duration + 400); // 兜底：极端节流环境（后台标签页等）也保证终值正确
  }

  function findMatchedCase(text) {
    const n = normalize(text);
    if (!n) return null;
    for (const c of PRESET_CASES) {
      const nc = normalize(c.inputText);
      if (n === nc) return c;
      if (n.length >= 60 && nc.includes(n)) return c;
    }
    return null;
  }

  function updateCharCount() {
    charCount.textContent = input.value.length + " 字";
  }

  function setStatus(message, isError) {
    let status = document.getElementById("form-status");
    if (!status) {
      status = el("p", "form-status");
      status.id = "form-status";
      status.setAttribute("role", "alert");
      form.querySelector(".form-actions").after(status);
    }
    status.textContent = message || "";
    status.classList.toggle("is-error", !!isError);
    status.hidden = !message;
  }

  function setActiveCard(id) {
    activeCaseId = id;
    caseList.querySelectorAll(".case-card").forEach(function (card) {
      const active = card.dataset.caseId === id;
      card.classList.toggle("is-active", active);
      card.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  /* ---------- 典型案例卡片 ---------- */
  function renderCases() {
    PRESET_CASES.forEach(function (c) {
      const li = el("li");
      const card = el("button", "case-card");
      card.type = "button";
      card.dataset.caseId = c.id;
      card.setAttribute("aria-pressed", "false");

      card.appendChild(el("span", "case-tag", c.tag));
      card.appendChild(el("h3", "case-title", c.title));
      card.appendChild(el("p", "case-desc", c.brief));
      card.appendChild(el("p", "case-src", "案例来源：" + c.source));
      card.appendChild(el("p", "case-fill-hint", "点击填入输入框 →"));

      card.addEventListener("click", function () {
        input.value = c.inputText;
        updateCharCount();
        setActiveCard(c.id);
        setStatus("已填入「" + c.tag + "」案例描述，点击「开始体检」查看报告。");
        input.focus();
        input.scrollIntoView({ behavior: "smooth", block: "center" });
      });

      li.appendChild(card);
      caseList.appendChild(li);
    });
  }

  /* ---------- 报告渲染 ---------- */
  function renderReport(caseDef) {
    const report = REPORTS[caseDef.id];
    const meta = report.meta;
    const doc = el("article", "report-doc");

    /* 报告头部 */
    const head = el("header", "report-head");
    head.appendChild(el("p", "report-title-line", "粤策哨 · 四段式申报前自查报告"));

    const metaList = el("dl", "report-meta");
    const mkMeta = function (dtText, ddText) {
      const row = el("div");
      row.appendChild(el("dt", null, dtText));
      row.appendChild(el("dd", null, ddText));
      metaList.appendChild(row);
    };
    mkMeta("体检输入", "「" + meta.caseTag + "」" + meta.caseTitle);

    /* 行政信息折叠：案例来源 / 来源链接 / 生成方式默认收起 */
    const metaFold = el("div", "meta-fold");
    const foldDetails = el("details", "meta-details");
    const foldSummary = el("summary", "meta-summary", "案例来源 · 链接 · 生成方式");
    const foldRows = el("div", "meta-rows");
    const mkFoldRow = function (k, valueNode) {
      const row = el("div", "meta-row");
      row.appendChild(el("span", "meta-k", k));
      const v = el("span", "meta-v");
      if (valueNode) v.appendChild(valueNode);
      row.appendChild(v);
      foldRows.appendChild(row);
    };
    mkFoldRow("案例来源", document.createTextNode(meta.caseSource));
    if (meta.sourceUrl) {
      const a = el("a", null, meta.sourceUrl);
      a.href = meta.sourceUrl;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      mkFoldRow("来源链接", a);
    }
    mkFoldRow("生成方式", document.createTextNode("静态预置输出（本页内置，无模型推理）"));
    foldDetails.appendChild(foldSummary);
    foldDetails.appendChild(foldRows);
    metaFold.appendChild(foldDetails);
    metaList.appendChild(metaFold);
    head.appendChild(metaList);
    doc.appendChild(head);

    if (meta.sourceNote) {
      const noteP = el("p", "sec-note");
      noteP.textContent = meta.sourceNote;
      const wrap = el("div", "report-head-note");
      wrap.appendChild(noteP);
      doc.appendChild(wrap);
    }

    /* 第①段 适用政策清单 */
    const s1 = el("section", "report-sec");
    s1.id = "report-sec-1";
    s1.appendChild(el("h3", null, ""));
    s1.querySelector("h3").appendChild(el("span", "sec-no", "1"));
    s1.querySelector("h3").appendChild(document.createTextNode("适用政策清单"));
    s1.appendChild(el("p", "sec-note",
      "按项目主题列出可能相关的政策文件与条款。时效状态取自知识库注册表（registry.json）与政策文件清单；「申报已截止」指文件载明的受理窗口已过，仅作参考。"));
    const policyList = el("ul", "policy-list");
    report.s1.forEach(function (p) {
      const item = el("li", "policy-item");
      const headRow = el("div", "policy-head");
      headRow.appendChild(el("span", "policy-wenhao", p.wenh));
      headRow.appendChild(el("span", "policy-title", p.title));
      const pillClass = p.status === "valid" ? "pill-valid" : p.status === "deadline" ? "pill-deadline" : "pill-void";
      headRow.appendChild(el("span", "pill " + pillClass, p.statusLabel));
      item.appendChild(headRow);
      item.appendChild(el("p", "policy-meta", p.dateLine));
      if (p.clauses && p.clauses.length) {
        const chips = el("div", "clause-chips");
        p.clauses.forEach(function (cl) {
          const chip = el("span", "clause-chip");
          const b = el("b", null, cl.no);
          chip.appendChild(b);
          chip.appendChild(document.createTextNode("　" + cl.text));
          chips.appendChild(chip);
        });
        item.appendChild(chips);
      }
      if (p.note) item.appendChild(el("p", "policy-note", "说明：" + p.note));
      policyList.appendChild(item);
    });
    s1.appendChild(policyList);
    doc.appendChild(s1);

    /* 第②段 负面清单命中提示 */
    const s2 = el("section", "report-sec");
    s2.id = "report-sec-2";
    const s2h = el("h3", null, "");
    s2h.appendChild(el("span", "sec-no", "2"));
    s2h.appendChild(document.createTextNode("负面清单命中提示"));
    s2.appendChild(s2h);
    s2.appendChild(el("p", "sec-note",
      "命中分两级：规则命中、疑似命中（建议对照原文）。每条附依据文号、条款号与日期；条款原文摘录自知识库清洗文本，逐字核对。"));
    const hitList = el("ol", "hit-list");
    report.s2.forEach(function (hit) {
      const item = el("li", "hit-item" + (hit.level === "rule" ? " is-rule" : " is-suspect"));
      const headRow = el("div", "hit-head");
      headRow.appendChild(el("span", "badge " + (hit.level === "rule" ? "badge-rule" : "badge-suspect"),
        hit.level === "rule" ? "规则命中" : "疑似命中，建议对照原文"));
      headRow.appendChild(el("span", "hit-title", hit.title));
      item.appendChild(headRow);
      item.appendChild(el("p", "hit-desc", hit.desc));

      const evDetails = el("details", "ev-details");
      const evSummary = el("summary", "ev-summary", "展开依据（文号 · 条款原文 · 来源）");
      const ev = el("div", "hit-evidence");
      hit.evidence.forEach(function (line) {
        if (!line.v && !line.k) return;
        const row = el("div", "ev-line");
        row.appendChild(el("span", "ev-k", line.k));
        if (line.k === "条款原文摘录" && line.v) {
          const quote = el("p", "ev-quote", line.v);
          ev.appendChild(quote);
          return;
        }
        row.appendChild(el("span", "ev-v", line.v || "——"));
        ev.appendChild(row);
      });
      evDetails.appendChild(evSummary);
      evDetails.appendChild(ev);
      item.appendChild(evDetails);
      if (hit.note) item.appendChild(el("p", "hit-note", "说明：" + hit.note));
      hitList.appendChild(item);
    });
    s2.appendChild(hitList);
    doc.appendChild(s2);

    /* 第③段 材料缺项清单 */
    const s3 = el("section", "report-sec");
    s3.id = "report-sec-3";
    const s3h = el("h3", null, "");
    s3h.appendChild(el("span", "sec-no", "3"));
    s3h.appendChild(document.createTextNode("材料缺项清单"));
    s3.appendChild(s3h);
    s3.appendChild(el("p", "sec-note", "对照第①段命中的现行政策文件，列出建议核对、补齐的申报材料（清单为提示性质）。"));
    const missList = el("ul", "missing-list");
    report.s3.forEach(function (m) {
      const item = el("li", "missing-item");
      const body = el("div", "missing-body");
      body.appendChild(el("p", "missing-name", m.name));
      body.appendChild(el("p", "missing-basis", "依据：" + m.basis));
      item.appendChild(body);
      missList.appendChild(item);
    });
    s3.appendChild(missList);
    doc.appendChild(s3);

    /* 第④段 修改建议 */
    const s4 = el("section", "report-sec");
    s4.id = "report-sec-4";
    const s4h = el("h3", null, "");
    s4h.appendChild(el("span", "sec-no", "4"));
    s4h.appendChild(document.createTextNode("修改建议"));
    s4.appendChild(s4h);
    s4.appendChild(el("p", "sec-note", "以下为申报表述层面的改进建议（建议口吻，不做任何判定）。"));
    const adviceList = el("ul", "advice-list");
    report.s4.forEach(function (a) {
      const item = el("li", "advice-item");
      item.appendChild(el("p", null, a));
      adviceList.appendChild(item);
    });
    s4.appendChild(adviceList);
    doc.appendChild(s4);

    /* 段落切换条：四段改为切换视图，避免长滚动信息过载 */
    const tabSpecs = [
      { sec: s1, label: "适用政策", count: report.s1.length },
      { sec: s2, label: "负面命中", count: report.s2.length },
      { sec: s3, label: "材料缺项", count: report.s3.length },
      { sec: s4, label: "修改建议", count: report.s4.length }
    ];
    const tabBar = el("div", "report-tabs");
    tabBar.setAttribute("role", "tablist");
    tabBar.setAttribute("aria-label", "报告段落切换");
    const tabs = [];
    function activateTab(idx, moveFocus) {
      tabSpecs.forEach(function (spec, j) {
        spec.sec.hidden = j !== idx;
        const t = tabs[j];
        t.classList.toggle("is-active", j === idx);
        t.setAttribute("aria-selected", j === idx ? "true" : "false");
        t.setAttribute("tabindex", j === idx ? "0" : "-1");
      });
      if (moveFocus) tabs[idx].focus();
    }
    tabSpecs.forEach(function (spec, idx) {
      spec.sec.hidden = idx !== 0;
      spec.sec.setAttribute("role", "tabpanel");
      const t = el("button", "report-tab" + (idx === 0 ? " is-active" : ""));
      t.type = "button";
      t.id = "report-tab-" + (idx + 1);
      t.setAttribute("role", "tab");
      t.setAttribute("aria-selected", idx === 0 ? "true" : "false");
      t.setAttribute("tabindex", idx === 0 ? "0" : "-1");
      t.setAttribute("aria-controls", spec.sec.id);
      spec.sec.setAttribute("aria-labelledby", t.id);
      t.appendChild(document.createTextNode(spec.label + " "));
      const num = el("b", null, "0");
      t.appendChild(num);
      animateCount(num, spec.count);
      t.addEventListener("click", function () { activateTab(idx, false); });
      tabs.push(t);
      tabBar.appendChild(t);
    });
    tabBar.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      let cur = 0;
      for (let j = 0; j < tabs.length; j++) {
        if (tabs[j].classList.contains("is-active")) { cur = j; break; }
      }
      const delta = e.key === "ArrowRight" ? 1 : -1;
      activateTab((cur + delta + tabs.length) % tabs.length, true);
    });
    head.appendChild(tabBar);

    /* 报告尾部：免责声明 + 操作 */
    const foot = el("footer", "report-foot");
    foot.appendChild(el("p", "report-disclaimer", "免责声明：" + DISCLAIMER_TEXT));
    const actions = el("div", "report-foot-actions");
    const backBtn = el("button", "btn btn-ghost", "返回输入框");
    backBtn.type = "button";
    backBtn.addEventListener("click", function () {
      input.scrollIntoView({ behavior: "smooth", block: "center" });
      input.focus();
    });
    actions.appendChild(backBtn);
    foot.appendChild(actions);
    foot.appendChild(el("p", "report-foot-note", "本报告为静态演示输出。全部引用文号、条款号与日期来源见页面底部「演示数据来源说明」。"));
    doc.appendChild(foot);

    reportBody.replaceChildren(doc);
    reportWrap.hidden = false;

    /* 焦点移交：标题可聚焦，键盘与读屏用户随滚动一起进入报告 */
    const titleEl = doc.querySelector(".report-title-line");
    titleEl.setAttribute("tabindex", "-1");
    titleEl.focus({ preventScroll: true });
  }

  /* ---------- 模拟体检过程（演示动画，非真实推理） ---------- */
  function playAnalyze(onDone) {
    clearTimeout(analyzeFallbackTimer);
    checkBtn.disabled = true;
    reportWrap.hidden = true;
    notice.hidden = true;
    analyzePanel.hidden = false;
    const steps = Array.prototype.slice.call(analyzeSteps.children);
    steps.forEach(function (li) { li.classList.remove("is-active", "is-done"); });

    /* 进度反馈：进度条 + 百分比 + 分步文字 */
    const fill = document.getElementById("analyze-fill");
    const pct = document.getElementById("analyze-pct");
    const statusText = document.getElementById("analyze-status-text");
    const stepTexts = ["正在解析项目描述", "正在扫描政策语料库", "正在进行负面清单比对", "正在汇总结果"];
    fill.style.width = "0%";
    pct.textContent = "0%";
    statusText.textContent = stepTexts[0];

    analyzePanel.scrollIntoView({ behavior: "smooth", block: "center" });
    steps[0].classList.add("is-active");
    let i = 0;
    clearInterval(analyzeTimer);
    analyzeTimer = setInterval(function () {
      steps[i].classList.remove("is-active");
      steps[i].classList.add("is-done");
      i += 1;
      fill.style.width = i * 25 + "%";
      pct.textContent = i * 25 + "%";
      statusText.textContent = stepTexts[Math.min(i, 3)];
      if (i < steps.length) {
        steps[i].classList.add("is-active");
      } else {
        finishAnalyze(onDone);
      }
    }, 520);
    /* 兜底：极端节流环境（后台标签页等）下计时器停摆，也保证结果最终出现 */
    analyzeFallbackTimer = setTimeout(function () {
      if (!analyzePanel.hidden) finishAnalyze(onDone);
    }, 520 * 4 + 900);
  }

  function finishAnalyze(onDone) {
    clearInterval(analyzeTimer);
    clearTimeout(analyzeFallbackTimer);
    checkBtn.disabled = false;
    analyzePanel.hidden = true;
    onDone();
  }

  function showReport(caseDef) {
    renderReport(caseDef);
    reportWrap.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
  }

  function showNotice() {
    notice.hidden = false;
    notice.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "center" });
  }

  /* ---------- 体检流程 ---------- */
  function runCheck() {
    if (!analyzePanel.hidden) return; // 演示动画进行中，忽略重复提交
    const text = input.value.trim();

    if (!text) {
      setStatus("请先输入项目描述，或点击下方典型案例一键填入。", true);
      input.classList.add("is-error");
      input.focus();
      return;
    }
    input.classList.remove("is-error");

    const matched = findMatchedCase(text);
    if (matched) {
      setActiveCard(matched.id);
      setStatus("");
      if (REDUCED_MOTION) {
        showReport(matched);
      } else {
        playAnalyze(function () { showReport(matched); });
      }
    } else {
      setStatus("");
      reportWrap.hidden = true;
      if (REDUCED_MOTION) {
        showNotice();
      } else {
        playAnalyze(showNotice);
      }
    }
  }

  /* ---------- 事件绑定与初始化 ---------- */
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    runCheck();
  });

  input.addEventListener("input", function () {
    updateCharCount();
    if (input.value.trim()) {
      input.classList.remove("is-error");
      setStatus("");
    }
  });

  /* ---------- 随机生成项目描述 ---------- */
  randomBtn.addEventListener("click", function () {
    let pool = PRESET_CASES;
    if (activeCaseId && PRESET_CASES.length > 1) {
      const others = PRESET_CASES.filter(function (c) { return c.id !== activeCaseId; });
      /* 排除当前项后仅剩 1 条或为空时降级回全池，保证随机性/可用性（当前 3 条案例不会触发，纯防御） */
      if (others.length > 1) pool = others;
    }
    if (!pool.length) {
      setStatus("暂无可抽取的典型案例，请直接选择下方卡片。", true);
      return;
    }
    const c = pool[Math.floor(Math.random() * pool.length)];
    input.value = c.inputText;
    updateCharCount();
    setActiveCard(c.id);
    input.classList.remove("is-error");
    setStatus("已随机生成一份「" + c.tag + "」项目描述，点击「开始体检」查看报告。");
    input.focus();
  });

  clearBtn.addEventListener("click", function () {
    input.value = "";
    updateCharCount();
    input.classList.remove("is-error");
    setStatus("");
    setActiveCard(null);
    reportWrap.hidden = true;
    notice.hidden = true;
    clearInterval(analyzeTimer);
    clearTimeout(analyzeFallbackTimer);
    analyzePanel.hidden = true;
    checkBtn.disabled = false;
    input.focus();
  });

  noticeBack.addEventListener("click", function () {
    notice.hidden = true;
    document.getElementById("presets").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  renderCases();
  updateCharCount();
})();
