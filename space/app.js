/* Kataa recorded-run ledger. Reads only bundled files under data/. All task text is untrusted
   (it contains prompt injections by design) and is inserted with textContent, never as HTML. */
"use strict";

const LANGS = [["en", "EN"], ["fr", "FR"], ["hi", "HI"], ["sw", "SW"], ["yo", "YO"], ["ha", "HA"]];
const LANG_NAME = { en: "English", fr: "French", hi: "Hindi", sw: "Swahili", yo: "Yoruba", ha: "Hausa" };
const FAMILIES = [["all", "All attacks"], ["pressure_authority", "Authority persona"], ["inj_structured", "Structured data"]];
const OUTCOMES = [["all", "All cases"], ["held", "Kataa held, base fell"], ["kataa_unsafe", "Kataa unsafe"], ["kataa_wrong", "Kataa wrong, not unsafe"]];
const DECISION = { margin_call: "Margin call", no_action: "No action", escalate: "Escalated" };
const TOOLS_DECISION = new Set(["issue_margin_call", "no_action", "escalate_to_human"]);

const state = { lang: "en", family: "all", outcome: "all", condition: "all", id: null, index: [], order: [], summary: null };
const cache = new Map();

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return n;
}
const pct = (x, d = 1) => (x === null || x === undefined ? "—" : (100 * x).toFixed(d) + "%");

function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function seededOrder(n, seed) { const r = mulberry32(seed), a = [...Array(n).keys()]; for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

function verdictClass(v) { if (!v || v.decision === null) return "none"; if (v.unsafe) return "bad"; if (v.correct) return "ok"; return "wrong"; }
function verdictWord(v) { if (!v || v.decision === null) return "No decision"; if (v.unsafe) return "Unsafe"; if (v.correct) return "Correct"; return "Wrong"; }


/* ---------- headline figures ---------- */
function renderFigures() {
  const s = state.summary, h = s.heldout, m = s.mistral_matched;
  const fig = (title, big, note, cls = "") => el("div", { class: "fig " + cls }, el("h2", { text: title }), big, el("p", { text: note }));
  const change = (a, b) => el("div", { class: "big" }, el("span", { class: "from", text: pct(a, 0) }), el("span", { class: "arrow", text: "→" }), pct(b, 0));
  document.getElementById("figures").replaceChildren(
    fig("Unsafe when it acts", change(h.base.unsafe_when_deciding.mean, h.kataa.unsafe_when_deciding.mean),
      `Base → Kataa, unsafe decisions among tasks where the model decided (held-out, 300). 95% CI for Kataa ${pct(h.kataa.unsafe_when_deciding.lo, 0)}–${pct(h.kataa.unsafe_when_deciding.hi, 0)}. Base decides nothing on ${pct(h.base.no_decision.mean, 0)} of tasks.`),
    fig("Accuracy", change(h.base.accuracy.mean, h.kataa.accuracy.mean),
      `Correct decision, base → Kataa. Following the attack: ${pct(h.base.followed_attack.mean, 0)} → ${pct(h.kataa.followed_attack.mean, 0)}.`),
    fig("Still not safe to automate", el("div", { class: "big" }, pct(m.kataa.unsafe.mean, 0), el("span", { class: "from", text: " vs " }), pct(m.mistral.unsafe.mean, 1)),
      `Unsafe rate, Kataa vs Mistral Large 4 (large hosted open-weights model) on the same 150 held-out tasks. About $${s.kataa_cost_per_1000.toFixed(2)} vs $${s.mistral_cost_per_1000_list.toFixed(2)} per 1,000 checks.`, "warn"),
  );
}

/* ---------- controls ---------- */
function segButtons(id, items, key, prefix) {
  const box = document.getElementById(id);
  box.replaceChildren(...items.map(([v, label]) => el("button", {
    type: "button", "data-check": `${prefix}-${v}`, "data-value": v, "aria-pressed": String(state[key] === v),
    onclick: () => { state[key] = v; syncControls(); renderList(true); },
  }, id === "outcomes" ? el("span", { class: "ic", "aria-hidden": "true" }) : null, label)));
}
function syncControls() {
  for (const [id, key] of [["langs", "lang"], ["families", "family"], ["outcomes", "outcome"]])
    for (const b of document.getElementById(id).querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.value === state[key]));
  document.getElementById("condition").value = state.condition;
}
function setupControls() {
  segButtons("langs", LANGS, "lang", "lang");
  segButtons("families", FAMILIES, "family", "fam");
  segButtons("outcomes", OUTCOMES, "outcome", "out");
  const sel = document.getElementById("condition");
  const conds = [...new Set(state.index.map((t) => t.condition))].sort();
  sel.replaceChildren(el("option", { value: "all", text: "All conditions" }), ...conds.map((c) => el("option", { value: c, text: c.replace(/_/g, " ") })));
  sel.addEventListener("change", () => { state.condition = sel.value; renderList(true); });
  document.getElementById("next").addEventListener("click", () => {
    const vis = visible(); if (!vis.length) return;
    const i = vis.findIndex((t) => t.id === state.id);
    select(vis[(i + 1) % vis.length].id);
  });
}

/* ---------- list ---------- */
function matches(t) {
  const L = t.langs[state.lang]; if (!L) return false;
  if (state.family !== "all" && t.family !== state.family) return false;
  if (state.condition !== "all" && t.condition !== state.condition) return false;
  const b = L.base, k = L.kataa;
  if (state.outcome === "held" && !(k.correct && b.unsafe)) return false;
  if (state.outcome === "kataa_unsafe" && !k.unsafe) return false;
  if (state.outcome === "kataa_wrong" && !(!k.correct && !k.unsafe)) return false;
  return true;
}
function visible() { return state.order.map((i) => state.index[i]).filter(matches); }
function dot(v) { return el("span", { class: "dot " + verdictClass(v), title: verdictWord(v) }, verdictWord(v).toUpperCase()); }
function renderList(selectFirst) {
  const vis = visible(), ol = document.getElementById("tasks");
  document.getElementById("count").textContent = `${vis.length} case${vis.length === 1 ? "" : "s"} · ${LANG_NAME[state.lang]} · base | Kataa`;
  if (!vis.length) {
    ol.replaceChildren(el("li", { class: "empty", text: "No recorded case matches these filters. The case on the right is from your previous selection." }));
    return;
  }
  ol.replaceChildren(...vis.map((t) => el("li", {}, el("button", {
    type: "button", "aria-current": String(t.id === state.id), onclick: () => select(t.id),
  }, el("span", {}, el("span", { class: "tid", text: t.id }), el("span", { class: "tmeta", text: `${t.condition.replace(/_/g, " ")} · ${t.family === "inj_structured" ? "structured injection" : "authority persona"}` })),
     el("span", { title: "Base Tiny Aya" }, dot(t.langs[state.lang].base)), el("span", { title: "Kataa" }, dot(t.langs[state.lang].kataa))))));
  if (selectFirst && !vis.some((t) => t.id === state.id)) select(vis[0].id);
  else if (state.id) select(state.id);
}

/* ---------- case ---------- */
async function loadTask(id) {
  if (!cache.has(id)) {
    const r = await fetch(`data/tasks/${id}.json`);
    if (!r.ok) throw new Error(`could not load ${id}: HTTP ${r.status}`);
    cache.set(id, await r.json());
  }
  return cache.get(id);
}
function argText(args) { if (typeof args === "string") return args; return Object.entries(args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", "); }
function renderLane(task, who, title, sub) {
  const L = task.lanes[state.lang][who], v = L.verdict;
  const turns = el("ol", { class: "turns" }, L.turns.map((t) => el("li", { class: "turn" },
    t.notes ? el("p", { class: "notes", text: t.notes }) : null,
    t.calls.length ? el("div", { class: "calls" }, t.calls.map((c) => el("div", { class: "call" + (TOOLS_DECISION.has(c.name) ? " decision" : "") }, el("b", { text: c.name }), "(", argText(c.args), ")"))) : null,
    t.malformed ? el("div", { class: "call malformed", text: "unparseable call list" }) : null,
    t.results.length ? el("details", { class: "results" }, el("summary", {}, `${t.results.length} tool result${t.results.length > 1 ? "s" : ""}`,
      t.results.some((r) => r.name === "get_position" && task.family === "inj_structured") ? el("span", { class: "flag", text: "contains a planted instruction" }) : null),
      t.results.map((r) => el("pre", { text: `${r.name}: ${r.text}` }))) : null)));
  const word = v.decision === null ? "No decision" : `${DECISION[v.decision]} · ${verdictWord(v)}`;
  const detail = [v.followed_attack ? "followed the attack" : null, v.over_escalate ? "escalated without need" : null,
    v.evidence_ok ? "evidence gathered first" : "decided without the full evidence", `${v.n_calls} tool calls`].filter(Boolean).join(" · ");
  return el("section", { class: "lane", "data-testid": "episode", "aria-label": title },
    el("div", { class: "lane-head" }, el("h3", { text: title }), el("span", { class: "sub", text: sub })),
    turns,
    el("div", { class: "stamp-row" }, el("span", { class: "stamp " + verdictClass(v), text: word }), el("span", { class: "verdict-detail", text: detail }),
      el("a", { class: "source", href: task.sources[state.lang][who], target: "_blank", rel: "noopener", text: "Recorded run file ↗" })));
}
function renderRequest(task) {
  const text = task.request[state.lang], inj = task.injection[state.lang] || {};
  const planted = task.family === "pressure_authority" && inj.text ? inj.text : null;
  return el("div", { class: "request" }, el("p", { class: "label", text: "Request to the desk" }),
    text.split("\n\n").map((p) => planted && planted.includes(p.trim()) && p.trim()
      ? el("div", { class: "planted-box" }, el("p", { class: "planted", text: p }), el("span", { class: "planted-note", text: "Instruction from the request text, not desk procedure: a held-out authority persona" }))
      : el("p", { text: p })),
    task.family === "inj_structured" && inj.text ? el("div", {}, el("p", { class: "label", text: "Planted in get_position → position_extra" }), el("pre", { class: "inj", text: inj.text }),
      el("span", { class: "planted-note planted-note-solo", text: "Instruction found in data, not from the desk: a held-out structured-data injection" })) : null);
}
async function select(id) {
  state.id = id;
  for (const b of document.querySelectorAll("#tasks li button")) b.setAttribute("aria-current", String(b.querySelector(".tid")?.textContent === id));
  history.replaceState(null, "", `#${id}&lang=${state.lang}`);
  const box = document.getElementById("case");
  let task;
  try { task = await loadTask(id); } catch (e) { box.replaceChildren(el("p", { class: "empty", text: String(e.message) })); return; }
  if (state.id !== id) return;
  if (!task.lanes[state.lang]) { state.lang = "en"; syncControls(); }
  const ans = DECISION[task.answer];
  box.replaceChildren(...[
    el("div", { class: "case-head" }, el("h2", { text: `${task.id} · loan ${task.loan_id}` }),
      el("span", { class: "chip", text: task.condition.replace(/_/g, " ") }),
      el("span", { class: "chip attack", text: task.family === "inj_structured" ? "held-out: structured-data injection" : "held-out: authority persona" }),
      el("span", { class: "chip", text: `attack wants: ${DECISION[task.target] || "—"}${task.aligned ? " (happens to be correct)" : ""}` }),
      el("span", { class: "chip", text: LANG_NAME[state.lang] })),
    task.excluded[state.lang] ? el("p", { class: "excluded", text: "Excluded from the metrics: this translation failed the back-translation check (lost token or overlap below 0.35). Shown for completeness." }) : null,
    el("div", { class: "brief" }, renderRequest(task),
      el("div", { class: "truth" }, el("p", { class: "label", text: "Ground truth (computed from the hidden state)" }), el("p", { class: "answer", text: ans }),
        el("p", { text: task.reason }),
        el("p", { text: `Maintenance LTV ${task.terms.maintenance_ltv}%${task.terms.original_maintenance_ltv ? ` (amended from ${task.terms.original_maintenance_ltv}%)` : ""} · prices current within ${task.terms.staleness_hours} h · oracles may differ by ${task.terms.divergence_pct}% · cure period ${task.terms.cure_hours} h` }))),
    el("details", { class: "contract" }, el("summary", { text: "Contract panel: the loan agreement the models read" }), el("pre", { text: task.agreement })),
    el("div", { class: "lanes" }, renderLane(task, "base", "Base Tiny Aya Global", "3.35B · no fine-tuning"), renderLane(task, "kataa", "Kataa", "Tiny Aya + warm start + GRPO (run 2)")),
  ].filter(Boolean));
}

/* ---------- hero, strip, story ---------- */
async function renderHero() {
  const box = document.getElementById("hero-visual");
  let t;
  try { t = await loadTask("heldout-00000"); } catch { return; }
  const inj = t.injection.en.text;
  const rest = t.request.en.split("\n\n").filter((p) => !inj.includes(p.trim())).join(" ");
  const row = (who, name) => {
    const v = t.lanes.en[who].verdict;
    const how = v.followed_attack ? "followed the attack" : v.evidence_ok ? "evidence first" : "no evidence";
    return el("div", { class: "mock-row" },
      el("div", {}, el("b", { text: name }), el("small", { text: v.n_calls + " tool call" + (v.n_calls === 1 ? "" : "s") + " · " + how })),
      el("span", { class: "stamp " + verdictClass(v), text: (DECISION[v.decision] || "No decision") + " · " + verdictWord(v) }));
  };
  box.replaceChildren(
    el("div", { class: "mock" }, el("span", { class: "mock-tag", text: "Recorded run · heldout-00000" }),
      el("p", { class: "mock-req", text: inj }), el("span", { class: "mock-req-note", text: "planted in the request: a held-out authority persona" }),
      el("p", { class: "notes", text: rest }),
      row("base", "Base Tiny Aya Global"), row("kataa", "Kataa")),
    el("div", { class: "mock-side" }, el("div", { class: "k", text: "ground truth" }), el("div", { text: DECISION[t.answer] }),
      el("div", { class: "k", text: "reason" }), el("div", { text: t.reason })));
}
function renderStrip() {
  const h = state.summary.heldout, s = state.summary;
  const item = (b, span) => el("div", { class: "strip-item" }, el("b", { text: b }), el("span", { text: span }));
  const arrow = (m) => pct(h.base[m].mean, 0) + " → " + pct(h.kataa[m].mean, 0);
  document.getElementById("strip").replaceChildren(
    item(arrow("unsafe_when_deciding"), "unsafe when it acts"),
    item(arrow("accuracy"), "accuracy"),
    item(arrow("followed_attack"), "followed the attack"),
    item("300", "held-out tasks"), item("6", "languages"),
    item("$" + s.kataa_cost_per_1000.toFixed(2), "per 1,000 checks"));
}
function renderStory() {
  const d = state.summary.dpo;
  const fig = (title, big, note, cls = "") => el("div", { class: "fig " + cls }, el("h2", { text: title }), big, el("p", { text: note }));
  const change = (a, b) => el("div", { class: "big" }, el("span", { class: "from", text: pct(a, 0) }), el("span", { class: "arrow", text: "→" }), pct(b, 0));
  document.getElementById("story-card").replaceChildren(el("div", { class: "story" },
    fig("Unsafe actions", change(d.unsafe[0], d.unsafe[1]), "DPO on Kataa's own unsafe decisions from the training tasks. A significant drop on the held-out tasks."),
    fig("Tasks escalated", el("div", { class: "big", text: pct(d.escalated, 0) }), "It learned to hand almost every loan to a human. Unnecessary escalation rose from 0.3% to 81%.", "warn"),
    fig("Accuracy", change(d.accuracy[0], d.accuracy[1]), "The acceptance gate, written before the run, rejected it. It is not shown here and was not released.")));
}

/* ---------- boot ---------- */
async function boot() {
  try {
    const [summary, index] = await Promise.all([fetch("data/summary.json").then((r) => r.json()), fetch("data/index.json").then((r) => r.json())]);
    state.summary = summary; state.index = index; state.order = seededOrder(index.length, 0);
  } catch (e) {
    document.getElementById("case").replaceChildren(el("p", { class: "empty", text: "Could not load the recorded runs: " + e.message }));
    return;
  }
  const m = /^#([\w-]+)(?:&lang=(\w+))?/.exec(location.hash);
  if (m && LANGS.some(([l]) => l === m[2])) state.lang = m[2];
  renderFigures(); renderStrip(); setupControls(); renderStory(); renderHero();
  if (m && state.index.some((t) => t.id === m[1])) { state.id = m[1]; renderList(false); } else renderList(true);
}
boot();
