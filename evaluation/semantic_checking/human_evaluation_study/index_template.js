function escapeHtml(s) {
  return s.replace(/[&<>]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
}

function highlightUnder(el) {
  if (!el) return;
  // Prism
  if (window.Prism && typeof window.Prism.highlightAllUnder === "function") {
    window.Prism.highlightAllUnder(el);
  }
}

async function main() {
  const status = document.getElementById("status");

  // Pick which survey JSON to load:
  // 1) URL query param: ?survey=survey_xxx.json
  // 2) <body data-survey-json="survey_xxx.json">
  // 3) fallback default
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("survey");
  const fromAttr = document.body?.dataset?.surveyJson;
  const surveyFile = (fromQuery || fromAttr || "").trim();
  if (!surveyFile) {
    throw new Error(
      "No survey specified. Use ?survey=xxx.json or <body data-survey-json=...>"
    );
  }

  status.textContent = `Loading ${surveyFile} ...`;

  const res = await fetch(`../dist_surveys/${surveyFile}`);
  const surveyJson = await res.json();

  // 关闭 SurveyJS 对 HTML 的净化，否则会把 pre 的 class（line-numbers）剥掉
  Survey.settings.sanitizeHtml = false;

  // 有些版本用的是另一个开关（保险起见一起加）
  Survey.settings.htmlSanitize = false;

  const survey = new Survey.Model(surveyJson);

  // 不显示默认完成页（否则你只看到 Thank you）
  survey.showCompletedPage = false;

  // ✅ 把文本里的换行符 \n 转成 <br/>，让 matrix 列标题也能换行
  survey.onTextMarkdown.add(function (_, options) {
    const txt = options.text || "";
    options.html = txt.replace(/\n/g, "<br/>");
  });

  survey.onAfterRenderPage.add(function (survey, options) {
    if (window.Prism) {
      Prism.highlightAllUnder(options.htmlElement);
    }
  });

  // ✅ 每次问卷/页面渲染后都进行高亮（翻页时尤其需要）
  survey.onAfterRenderSurvey.add((sender, opts) => {
    highlightUnder(opts.htmlElement);
  });
  survey.onAfterRenderPage.add((sender, opts) => {
    highlightUnder(opts.htmlElement);
  });

  function highlightPrism() {
    const root = document.getElementById("surveyContainer");
    if (!root || !window.Prism) return;
    Prism.highlightAllUnder(root);
  }
  
  // ✅ 第一次渲染后跑一次
  highlightPrism();
  
  // ✅ 每次翻页后再跑一次（非常关键）
  survey.onCurrentPageChanged.add(function() {
    // 等 DOM 更新完再高亮
    setTimeout(highlightPrism, 0);
  });
  
  // ✅ 保险：页面渲染完成事件（不同版本 SurveyJS 有时更准）
  survey.onAfterRenderPage.add(function() {
    setTimeout(highlightPrism, 0);
  });

  // If a respondent changes the semantic correctness score back to 5,
  // clear any previously selected reasons to avoid stale data.
  survey.onValueChanged.add((sender, options) => {
    const qname = options.name;
    if (!qname || !qname.startsWith("rating_")) return;

    const v = options.value;
    const score = v && typeof v === "object"
      ? (v.semantic_correctness ?? v.semantic_accuracy)
      : undefined;

    if (Number(score) === 5) {
      const taskId = qname.substring("rating_".length);
    
      // 延迟到当前变更完全结束之后再清 reason
      setTimeout(() => {
        // 再确认一次 rating 仍然是 5（防抖）
        const cur = sender.getValue(qname);
        const curScore =
          cur && typeof cur === "object"
            ? cur.semantic_correctness ?? cur.semantic_accuracy
            : undefined;
    
        if (Number(curScore) === 5) {
          sender.setValue(`reason_${taskId}`, []);
        }
      }, 0);
    }
  });

  survey.onComplete.add(async (sender) => {
    try {
      // Package full payload (so server can attribute results)

      // 先拷贝一份 data，避免直接改 sender.data 引发副作用
      const data = JSON.parse(JSON.stringify(sender.data || {}));

      // === Merge reason_* into corresponding rating_* ===
      Object.keys(data).forEach((k) => {
        if (!k.startsWith("reason_")) return;

        const taskId = k.slice("reason_".length); // e.g. easy_task_2
        const ratingKey = `rating_${taskId}`;

        const reasons = data[k];
        if (data[ratingKey] && Array.isArray(reasons)) {
          // Attach reasons to rating object
          data[ratingKey].reasons = reasons;
        }

        // Remove standalone reason entry
        delete data[k];
      });

      // 取到 llm_code_selection（你现在已经在输出 meta 里了，说明这里能拿到同一份 surveyJson）
      const llmSel =
        (surveyJson && surveyJson.llm_code_selection) ||
        (surveyJson && surveyJson.meta && surveyJson.meta.llm_code_selection) ||
        {}; // 兜底

      Object.keys(data).forEach((k) => {
        if (!k.startsWith("rating_")) return;
        const v = data[k];
        if (!v || typeof v !== "object") return;

        // rating_easy_task_12 -> easy_task_12
        const taskId = k.slice("rating_".length);

        const selectedFile = llmSel[taskId];
        if (!selectedFile) return;

        // 你想存什么格式都可以：
        v.generated_code_id = selectedFile; // e.g. "generated_code_1.py"

        // 如果你想存不带 .py 的 id：
        // v.generated_code_id = String(selectedFile).replace(/\\.py$/, "");
      });

      const payload = {
        surveyFile,
        meta: {
          href: window.location.href,
          ts: new Date().toISOString(),
        },
        data,
      };

      const dataStr = JSON.stringify(payload, null, 2);

      // 1) Try to submit to backend
      let submitMsg = "";
      try {
        status.textContent = "Submitting results...";
        const resp = await fetch("/api/submit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) {
          const t = await resp.text();
          throw new Error(t || `HTTP ${resp.status}`);
        }
        const ret = await resp.json();
        submitMsg = `Submitted ✓ (${ret.file || "saved"})`;
      } catch (e) {
        console.warn("Submit failed, will still download locally:", e);
        submitMsg = "Submit failed (downloaded locally)";
      }

      // 2) Show results JSON in page
      status.innerHTML = `
        <div><b>${escapeHtml(submitMsg)}</b></div>
        <div style="opacity:.8; margin-top:4px;">Below is the result JSON (a local copy will be downloaded automatically)</div>
        <pre><code class="language-json">${escapeHtml(dataStr)}</code></pre>
      `;
      status.scrollIntoView({ behavior: "smooth", block: "start" });

      // Highlight JSON (optional)
      highlightUnder(status);

      // 3) Always download a local backup JSON
      const blob = new Blob([dataStr], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      // Build a clean local filename consistent with backend naming
      let suffix = "";
      if (surveyFile) {
        suffix = surveyFile
          .replace(/^survey_/, "")
          .replace(/\.json$/i, "")
          .replace(/[^a-zA-Z0-9._-]+/g, "_");
      }

      const now = new Date();
      const y = now.getFullYear();
      const m = String(now.getMonth() + 1).padStart(2, "0");
      const d = String(now.getDate()).padStart(2, "0");
      const h = String(now.getHours()).padStart(2, "0");
      const min = String(now.getMinutes()).padStart(2, "0");
      
      const dateStr = `${y}${m}${d}_${h}${min}`;

      const filename = suffix
        ? `result_${suffix}_${dateStr}.json`
        : `result_${dateStr}.json`;

      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      console.error("onComplete error:", e);
      status.textContent = "onComplete 出错：" + e.message;
    }
  });

  survey.render(document.getElementById("surveyContainer"));
  // Toggle side-by-side reference compare (event delegation)
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-toggle-ref]");
    if (!btn) return;

    const wrapper = btn.closest(".code-compare");
    if (!wrapper) return;

    wrapper.classList.toggle("is-compare");
    const isCompare = wrapper.classList.contains("is-compare");

    // Update button label
    btn.textContent = isCompare ? "Hide Reference" : "Compare with Reference";

    // a11y flag
    const refPanel = wrapper.querySelector(".code-panel.ref");
    if (refPanel) refPanel.setAttribute("aria-hidden", isCompare ? "false" : "true");

    // Ensure Prism highlights the newly shown panel
    highlightUnder(wrapper);
  });

  status.textContent = "Ready.";
}

main().catch((err) => {
  console.error(err);
  document.getElementById("status").textContent = "Error: " + err.message;
});