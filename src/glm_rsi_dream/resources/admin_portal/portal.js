"use strict";

const pages = ["rules", "queue", "dreams", "pockets", "autonomy", "system"];
const titles = {
  rules: "Rules",
  queue: "Learning queue",
  dreams: "Dreams & schedule",
  pockets: "Knowledge pockets",
  autonomy: "Autonomy",
  system: "System",
};
const descriptions = {
  rules: "Choose what belongs in your learning workspace.",
  queue: "Review candidate facts and their proposed destinations.",
  dreams: "Dream features are under development.",
  pockets: "Keep related raw facts together, ready to share.",
  autonomy: "Choose how much the system can decide on its own.",
  system: "A clear view of the workspace and its connections.",
};
const labels = {
  strategy: "Strategy",
  probe_samples: "Independent prior answers",
  probe_temperature: "Prior temperature",
  agreement_threshold: "Agreement threshold",
  extraction_instructions: "Extraction instructions",
  prior_instructions: "Prior-answer instructions",
  comparison_instructions: "Comparison instructions",
  assessment_instructions: "Claim-assessment instructions",
  instructions: "Custom grading instructions",
  routing_rules: "Destination rules",
  max_attempts: "Repair attempts",
  mode: "Execution location",
  autonomy: "Decision level",
  permission_file: "Signed permission file",
  auto_routes: "Automatic destinations",
  min_score: "Minimum confidence",
  name: "Name",
  id: "Identifier",
  enabled: "Enabled",
  domain: "Domain",
  action: "Retention action",
  route: "Destination",
  pocket: "Named pocket",
  priority: "Priority",
  user_ids: "Included users",
  exclude_user_ids: "Excluded users",
  source_kinds: "Source types",
  patterns: "Text patterns",
  notes: "Notes",
  topic: "Single topic",
  capacity: "Fact capacity",
  expected_revision: "Current revision",
  timezone: "Time zone",
  windows: "Approved windows",
  min_idle_seconds: "Minimum idle time (seconds)",
  max_active_requests: "Maximum active requests",
  run_id: "Staged run",
  not_before: "Earliest start",
  at: "Check time",
  idle_seconds: "Current idle time (seconds)",
  active_requests: "Current active requests",
  hold: "Hold for review",
  note: "Review note",
  text: "Candidate fact",
  prompt_evidence: "Literal source evidence",
  candidate_ids: "Accepted candidate identifiers",
  fact_ids: "Fact identifiers",
  sources: "Source pockets",
  package: "Shareable pocket package",
  preset: "Preset",
  user_id: "User identifier",
};
const state = {
  page: "rules",
  advanced: false,
  demo: new URLSearchParams(location.search).get("demo") === "1",
  connected: false,
  token: "",
  engineConnection: null,
  session: null,
  data: null,
  spec: null,
  error: "",
  filter: "all",
  search: "",
  selected: new Set(),
  logs: [],
  pending: null,
  policyDraft: null,
  form: null,
  busy: false,
  autonomyIssue: null,
};

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value == null ? "" : String(value);
  const encodedText = element.innerHTML;
  const encodedQuotes = encodedText.replaceAll('"', "&quot;");
  return encodedQuotes.replaceAll("'", "&#39;");
}

function label(name) {
  if (labels[name]) return labels[name];
  return name.replaceAll("_", " ");
}

function button(text, action, kind = "secondary", id = "") {
  return `<button type="button" class="${kind}" data-action="${escapeHtml(action)}" data-id="${escapeHtml(id)}">${escapeHtml(text)}</button>`;
}

function pill(text, color = "") {
  return `<span class="pill ${color}">${escapeHtml(text)}</span>`;
}

function valueText(value) {
  if (value === null) return "Default";
  if (value === true) return "On";
  if (value === false) return "Off";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "None";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function displayMode(mode) {
  return mode === "spark" ? "Spark-side" : "External harness";
}

function getGrade(candidate) {
  return candidate.proposal?.grade || candidate.grade || {};
}

function collection(name) {
  if (!state.data || !Array.isArray(state.data[name])) return [];
  return state.data[name];
}

function dreamsEnabled(system = state.data?.system) {
  return !state.demo && system?.features?.dreams?.enabled === true;
}

function isDreamPath(path) {
  const route = path.replace(/^\/api\/v1/, "");
  return /^\/(dreams|schedule)(\/|$)/.test(route);
}

function dreamUnavailableHtml() {
  return `<div class="surface-body"><h3>Under development</h3><p>Dreams, scheduling and consolidation are unavailable while neuron recall qualification is in progress.</p><p class="muted small">Queue review, rules, grading and raw-fact pockets remain available.</p></div>`;
}

function showDreamUnavailable() {
  openDrawer("Dream features", dreamUnavailableHtml(), "Under development");
}

function recordLog(method, path, status) {
  state.logs.unshift({
    time: new Date().toLocaleTimeString([], { hour12: false }),
    method,
    path,
    status,
  });
  if (state.logs.length > 80) state.logs.pop();
}

async function api(path, method = "GET", body = null, representation = {}) {
  const headers = { Accept: "application/json" };
  if (state.token) headers.Authorization = "Bearer " + state.token;
  const options = { method, headers, cache: "no-store" };
  if (body !== null) {
    headers["Content-Type"] = "application/json";
    options.body = representation.requestText ?? JSON.stringify(body);
  }
  const response = await fetch("/api/v1" + path, options);
  recordLog(method, path, response.status);
  let result;
  const jsonText = await response.text();
  try {
    result = JSON.parse(jsonText);
  } catch {
    throw new Error("The admin API returned an unreadable reply.");
  }
  if (!response.ok) {
    let message = result.error || "The admin API rejected this request.";
    if (response.status === 401) {
      message = "Sign in to continue.";
      if (state.session?.hosted) {
        state.session.authenticated = false;
        state.data = null;
      }
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return representation.responseText ? jsonText : result;
}

function listFrom(result) {
  if (!Array.isArray(result.items))
    throw new Error("The admin API returned an unsupported collection format.");
  return result.items;
}

async function sessionRequest(path, method = "GET", body = null) {
  const options = {method, credentials: "same-origin", headers: {Accept: "application/json"}};
  if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch("/session/" + path, options);
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(response.status === 401 ? "That password didn't work. Try again." : result.error || "Couldn't update portal access.");
    error.status = response.status;
    throw error;
  }
  return result;
}

async function discoverSession() {
  const response = await fetch("/session/status", {credentials: "same-origin", cache: "no-store"});
  if (response.status === 404) return {hosted: false};
  if (!response.ok) throw new Error("Couldn't check portal access. Refresh to try again.");
  const status = await response.json();
  return status.hosted === true ? status : {hosted: false};
}

async function refresh() {
  state.error = "";
  try {
    if (!state.demo) state.session = await discoverSession();
    if (state.session?.hosted) state.token = "";
    if (!state.demo && state.session?.hosted && !state.session.authenticated) {
      state.data = null;
      state.connected = false;
      render();
      return;
    }
    if (!state.demo && !state.session?.hosted && !state.token) {
      render();
      return;
    }
    if (state.demo) {
      if (!state.data) state.data = await (await fetch("demo.json")).json();
      if (!state.spec) state.spec = await (await fetch("openapi.json")).json();
    } else {
      const system = await api("/system");
      const control = await api("/control");
      const rules = await api("/rules");
      const queue = await api("/queue");
      let schedule = null;
      let dreams = { items: [] };
      if (dreamsEnabled(system)) {
        schedule = await api("/schedule");
        dreams = await api("/dreams");
      }
      const pockets = await api("/pockets");
      const profile = await api("/policy");
      state.data = {
        system,
        control,
        schedule,
        rules: listFrom(rules),
        queue: listFrom(queue),
        dreams: listFrom(dreams),
        pockets: listFrom(pockets),
        policy: profile.policy,
        policyProfile: profile,
        authority: system.authority || control.permission || {},
      };
      const specHeaders = {Accept: "application/json"};
      if (state.token) specHeaders.Authorization = "Bearer " + state.token;
      const specResponse = await fetch("/openapi.json", {
        headers: specHeaders,
        cache: "no-store",
      });
      if (specResponse.ok) state.spec = await specResponse.json();
    }
    const pendingIds = new Set();
    for (const item of collection("queue")) {
      if (item.status === "pending") pendingIds.add(item.id);
    }
    for (const id of state.selected) {
      if (!pendingIds.has(id)) state.selected.delete(id);
    }
    state.connected = true;
  } catch (error) {
    state.error = error.message;
    state.connected = false;
  }
  render();
}

function heading(actions = "") {
  return `<div class="page-heading"><div><h1>${titles[state.page]}</h1><p>${descriptions[state.page]}</p></div><div class="heading-actions">${actions}</div></div>`;
}

function section(title, subtitle, body, action = "") {
  return `<section class="surface"><div class="surface-heading"><div><h2>${escapeHtml(title)}</h2>${subtitle ? `<p class="muted">${escapeHtml(subtitle)}</p>` : ""}</div>${action}</div>${body}</section>`;
}

function renderRules() {
  let rows = "";
  for (const rule of collection("rules")) {
    const route = rule.route || rule.destination;
    rows += `<div class="rule-row"><span class="rule-mark" aria-hidden="true">◇</span><div><h3>${escapeHtml(rule.name)}</h3><p>${escapeHtml(rule.domain || rule.preset || "General")} ${rule.notes ? ": " + escapeHtml(rule.notes) : "retention rule"}</p></div><div class="rule-routing">${pill(route === "pocket" ? rule.pocket || "Named pocket" : route)}${pill(rule.enabled ? "Active" : "Paused", rule.enabled ? "green" : "gray")}</div>${button("Edit", "rule-edit", "text-button", rule.id)}</div>`;
  }
  if (!rows)
    rows = empty(
      "Create your first save rule",
      "Choose a topic or start from a companion preset.",
      button("Create rule", "rule-create", "primary"),
    );
  let html = heading(button("Create rule", "rule-create", "primary"));
  html += section(
    "Save rules",
    "Explicit decisions about what to retain and where",
    rows,
    pill(collection("rules").length + " rules", "gray"),
  );
  html += `<div class="section-label"><h2>Start with a preset</h2>${button("About presets", "help-rules", "text-button")}</div><div class="preset-strip"><button type="button" class="preset" data-action="preset" data-id="companion" aria-label="Add companion preset"><span class="preset-icon" aria-hidden="true">◎</span><span><h3>Companion</h3><p>Preferences and personal context, saved to memory.</p></span><span class="plus" aria-hidden="true">+</span></button><button type="button" class="preset" data-action="preset" data-id="persona" aria-label="Add persona preset"><span class="preset-icon" aria-hidden="true">◇</span><span><h3>Persona</h3><p>Approved identity and voice, kept together.</p></span><span class="plus" aria-hidden="true">+</span></button></div>`;
  html += `<div class="insight-row"><span aria-hidden="true">ⓘ</span><span>Rules shape proposals. Your autonomy level decides who approves them.</span>${button("View autonomy", "navigate", "text-button", "autonomy")}</div>`;
  if (state.advanced) html += renderPolicy();
  return html;
}

function settingRows(values, prefix, editable = false) {
  let html = "";
  for (const [name, value] of Object.entries(values || {})) {
    if (name === "_comments") continue;
    const path = prefix ? prefix + "." + name : name;
    const description =
      typeof value === "string" && value.length > 65
        ? "Open instructions"
        : valueText(value);
    html += `<div class="setting-row"><div><span class="setting-name">${escapeHtml(label(name))}</span><span class="setting-path">${escapeHtml(path)}</span></div>${editable ? button(description, "policy-field", "setting-value editable", path) : `<span class="setting-value" title="${escapeHtml(valueText(value))}">${escapeHtml(description)}</span>`}</div>`;
  }
  return html;
}

function renderPolicy() {
  const policy = state.policyDraft || state.data.policy;
  let html = `<div class="section-label"><h2>Learning policy</h2><div class="heading-actions">${button("Import policy", "policy-import")}${button("Export policy", "policy-export")}${button("Save policy", "policy-save", "primary")}${button("Raw values", "policy-raw")}</div></div>`;
  if (!policy)
    return (
      html +
      `<div class="insight-row">Import a policy file to inspect and edit every collection and grading setting. Export creates the configuration for a new run.</div>`
    );
  html += `<div class="setting-grid">`;
  for (const name of ["extract", "grade", "control"]) {
    html += section(
      name === "extract"
        ? "Collection & novelty"
        : name === "grade"
          ? "Grading & destinations"
          : "Run configuration",
      "Next-run policy values",
      settingRows(policy[name], name, true),
    );
  }
  html += section(
    "Policy identity",
    "Run envelope and revision metadata",
    settingRows(
      { name: policy.name, schema_version: policy.schema_version },
      "",
      true,
    ),
  );
  return (
    html +
    `</div><div class="insight-row">Policy edits create a next-run draft. Existing sealed runs retain their original policy.</div>`
  );
}

function empty(title, detail, action = "") {
  return `<div class="empty-state"><div class="empty-symbol" aria-hidden="true">◇</div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(detail)}</p>${action}</div>`;
}

function queueRows() {
  const rows = [];
  for (const item of collection("queue")) {
    if (state.filter !== "all" && item.status !== state.filter) continue;
    const searchText = (
      item.text +
      " " +
      (item.source?.ref || "")
    ).toLowerCase();
    if (state.search && !searchText.includes(state.search.toLowerCase()))
      continue;
    rows.push(item);
  }
  return rows;
}

function renderQueue() {
  const counts = { pending: 0, accepted: 0, rejected: 0 };
  for (const item of collection("queue")) {
    if (["accepted", "staged", "absorbed"].includes(item.status))
      counts.accepted += 1;
    else if (counts[item.status] !== undefined) counts[item.status] += 1;
  }
  let actions = button("Learn this", "learn-open", "primary");
  if (state.advanced) actions += button("Export queue", "queue-export");
  else actions += button("Refresh", "refresh");
  let html = heading(actions);
  html += `<div class="queue-summary"><span><strong>${counts.pending}</strong>Waiting for review</span><span><strong>${counts.accepted}</strong>Accepted</span><span><strong>${counts.rejected}</strong>Rejected</span></div>`;
  html += `<div class="surface"><div class="queue-toolbar"><label class="search-wrap"><span aria-hidden="true">⌕</span><input id="queue-search" type="search" aria-label="Search queue" placeholder="Search facts or sources" value="${escapeHtml(state.search)}"></label><div class="queue-tools"><select id="queue-filter" aria-label="Filter queue">`;
  for (const status of [
    "all",
    "pending",
    "accepted",
    "rejected",
    "staged",
    "absorbed",
  ])
    html += `<option value="${status}" ${state.filter === status ? "selected" : ""}>${status === "all" ? "All statuses" : label(status)}</option>`;
  html += `</select>${state.advanced ? button("Select visible", "select-visible") : pill("Human review", "gray")}</div></div>`;
  if (state.advanced && state.selected.size)
    html += `<div class="bulk-bar"><strong>${state.selected.size} selected</strong>${button("Approve proposals", "bulk-approve", "primary")}${button("Hold selected", "bulk-hold")}${button("Reject selected", "bulk-reject", "danger")}</div>`;
  const items = queueRows();
  if (!items.length)
    html += empty(
      "Your queue is clear",
      state.search
        ? "Try another search or status filter."
        : "New candidate facts will appear here for review.",
    );
  else {
    html += `<div class="table-wrap"><table><thead><tr>${state.advanced ? "<th scope='col'>Select</th>" : ""}<th scope="col">Candidate knowledge</th><th scope="col">Destination</th><th scope="col">Status</th><th scope="col">Review</th></tr></thead><tbody>`;
    for (const item of items) {
      const grade = getGrade(item);
      const destination =
        grade.route === "pocket"
          ? grade.pocket || "Pocket"
          : grade.route || "Not graded";
      let reviewButtons = button("View", "queue-detail", "text-button", item.id);
      if (item.status === "pending") {
        if (item.source?.kind === "manual" && !item.proposal) {
          reviewButtons = button("Grade fact", "queue-grade", "primary", item.id);
        } else {
          reviewButtons = button(grade.verdict === "reject" ? "Confirm rejection" : "Approve", "queue-approve", "primary", item.id);
        }
        reviewButtons += button("Reject", "queue-reject", "secondary", item.id);
      }
      html += `<tr>${state.advanced ? `<td><input type="checkbox" class="candidate-check" aria-label="Select ${escapeHtml(item.id)}" data-id="${escapeHtml(item.id)}" ${state.selected.has(item.id) ? "checked" : ""}></td>` : ""}<td class="candidate-cell"><button class="candidate-title" data-action="queue-detail" data-id="${escapeHtml(item.id)}">${escapeHtml(item.text)}</button><div class="candidate-source">${escapeHtml(item.source?.ref || "No source")} ${state.advanced ? " | " + escapeHtml(item.id) : ""}</div></td><td>${pill(destination)}</td><td>${pill(item.admin_review?.hold || item.hold ? "On hold" : item.status, item.status === "pending" ? "amber" : item.status === "accepted" ? "green" : "gray")}</td><td><div class="review-buttons">${reviewButtons}</div></td></tr>`;
    }
    html += `</tbody></table></div>`;
  }
  return (
    html +
    `<div class="table-end"><span>${items.length} of ${collection("queue").length} facts</span><span>${state.advanced ? "Tab to select, Ctrl+Enter to approve" : "Decisions are recorded by the admin API"}</span></div></div>`
  );
}

function scheduleValues() {
  const schedule = state.data.schedule || {};
  return schedule.schedule || schedule;
}

function renderDreams() {
  if (!dreamsEnabled()) {
    let unavailable = heading(pill("Under development", "gray"));
    unavailable += section("Dream features", "Under development", dreamUnavailableHtml());
    if (state.advanced) {
      const feature = state.data?.system?.features?.dreams || {
        enabled: false,
        status: "Under development",
      };
      unavailable += section("Server feature status", "Read-only host configuration", settingRows(feature, "dreams"));
    }
    return unavailable;
  }
  const schedule = scheduleValues();
  const windows = schedule.windows || schedule.approved_windows || [];
  let week = `<div class="week-strip">`;
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  for (let index = 0; index < 7; index += 1) {
    let slots = "";
    for (const window of windows) {
      if (
        window.days.includes(index) ||
        window.days.includes(
          [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
          ][index],
        )
      )
        slots += `<span class="window">${escapeHtml(window.start)}<br>${escapeHtml(window.end)}</span>`;
    }
    week += `<div class="day"><strong>${days[index]}</strong>${slots || "<span class='muted'>Rest</span>"}</div>`;
  }
  week += `</div>`;
  let html = heading(pill("Under development", "gray") + button("Prepare a plan", "dream-create", "primary"));
  let plans = "";
  for (const dream of collection("dreams"))
    plans += `<div class="dream-plan"><div><h3>${escapeHtml(dream.run_id || dream.id)}</h3><p>${escapeHtml(dream.status || "planned")}${dream.candidate_count ? " | " + dream.candidate_count + " candidate facts" : ""}</p></div>${button("Review plan", "dream-detail", "text-button", dream.id)}</div>`;
  if (!plans)
    plans = empty(
      "No dreams queued",
      "Development plans use an accepted staged run.",
    );
  html += `<div class="split-workspace"><div>${section("Dream workspace", "Under development: manual plan preparation", `<div class="dream-status"><div class="dream-orbit" aria-hidden="true">☾</div><div><h3>${collection("dreams").length ? "Ready for review" : "Waiting for a staged run"}</h3><p>This API prepares manual plans and performs no neuron training.</p></div></div>${plans}`)}<div class="section-label"><h2>Approved time windows</h2>${button("Edit schedule", "schedule-edit", "text-button")}</div>${section("Weekly schedule", schedule.timezone || "Time zone not configured", week, pill(schedule.enabled ? "Enabled" : "Paused", schedule.enabled ? "green" : "gray"))}</div><div>${section("Dream conditions", "Your learning window", `<div class="detail-list"><div class="detail-row"><span>Schedule</span><span>${schedule.enabled ? "Enabled" : "Paused"}</span></div><div class="detail-row"><span>Idle for at least</span><span>${Math.round((schedule.min_idle_seconds ?? schedule.idle_seconds ?? 0) / 60)} minutes</span></div><div class="detail-row"><span>Maximum active requests</span><span>${schedule.max_active_requests ?? "Not configured"}</span></div><div class="detail-row"><span>Time zone</span><span>${escapeHtml(schedule.timezone || "Not configured")}</span></div></div>`, button("Edit", "schedule-edit", "text-button"))}<div class="insight-row">Under development. The host flag opens manual plan preparation only.</div></div></div>`;
  if (state.advanced)
    html += `<div class="section-label"><h2>Raw schedule & jobs</h2>${button("Export plans", "dream-export")}</div>${section("Schedule values", "Every current scheduling value", settingRows(schedule, "schedule"), button("Raw values", "schedule-raw", "text-button"))}`;
  return html;
}

function pocketUsed(pocket) {
  if (Array.isArray(pocket.facts)) return pocket.facts.length;
  return pocket.used || 0;
}

function renderPockets() {
  let html = heading(
    `${state.advanced ? button("Import pocket", "pocket-import") + button("Merge pockets", "pocket-merge") : ""}${button("Create pocket", "pocket-create", "primary")}`,
  );
  html += `<div class="pocket-grid">`;
  for (const pocket of collection("pockets")) {
    const used = pocketUsed(pocket);
    const capacity = pocket.capacity || 0;
    html += `<article class="pocket-card"><div class="pocket-glyph" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div><div class="pocket-top"><h2>${escapeHtml(pocket.name || pocket.id)}</h2>${pill(pocket.shared ? "Shared" : "Local", pocket.shared ? "green" : "gray")}</div><p class="topic">${escapeHtml(pocket.topic)}</p><div class="capacity-track"><div class="capacity-fill" data-used="${used}" data-capacity="${capacity}"></div></div><div class="capacity-label"><span>${used} facts</span><span>${capacity} capacity</span></div><div class="pocket-actions">${button("Manage", "pocket-detail", "text-button", pocket.id)}${button("Share", "pocket-export", "text-button", pocket.id)}</div>${state.advanced ? `<div class="progress-note">${escapeHtml(pocket.id)} | revision ${pocket.revision || 1}</div>` : ""}</article>`;
  }
  if (!collection("pockets").length)
    html += empty(
      "Give your knowledge a home",
      "Create a single-topic pocket or import a shared package.",
      button("Create pocket", "pocket-create", "primary"),
    );
  html += `</div><div class="section-label"><h2>The knowledge journey</h2>${button("How pockets work", "help-pockets", "text-button")}</div><div class="surface pocket-flow"><div class="flow-step"><span class="flow-dot">1</span><span><strong>Collect & review</strong><small>Supported candidate facts</small></span></div><span class="flow-line"></span><div class="flow-step"><span class="flow-dot">2</span><span><strong>Keep in a pocket</strong><small>Single-topic knowledge pills</small></span></div><span class="flow-line"></span><div class="flow-step"><span class="flow-dot">3</span><span><strong>Dream features</strong><small>Under development</small></span></div></div>`;
  if (state.advanced)
    html += `<div class="insight-row">Shared packages contain raw facts with provenance. Export, copy and merge preserve that distinction from learned model weights.</div>`;
  return html;
}

function controlValues() {
  const control = state.data.control || state.data.policy?.control || {};
  return control.settings || control.control || control;
}

function renderAutonomy() {
  const control = controlValues();
  const current = control.autonomy || "manual";
  const options = [
    ["manual", "Manual", "Review every proposed fact before it is saved."],
    ["supervised", "Supervised", "Save confident facts automatically. Review uncertain facts and changes."],
    ["automatic", "Automatic", "Save facts automatically after the usual checks."],
  ];
  let choices = "";
  for (const [value, title, detail] of options) {
    let issue = "";
    if (state.autonomyIssue?.level === value) {
      issue = `<div class="autonomy-issue" role="alert"><span>${escapeHtml(state.autonomyIssue.message)}</span>${button(state.autonomyIssue.button, state.autonomyIssue.action, "text-button", value)}</div>`;
    }
    choices += `<div class="autonomy-choice"><button type="button" aria-label="Select ${escapeHtml(title)}" class="autonomy-option ${current === value ? "selected" : ""}" data-action="autonomy-select" data-id="${value}" aria-pressed="${current === value}"><span class="radio-mark" aria-hidden="true"></span><span><h3>${title}</h3><p>${detail}</p></span>${current === value ? pill("Selected", "green") : ""}</button>${issue}</div>`;
  }
  const destinations = [];
  for (const route of control.auto_routes || []) {
    destinations.push(route === "memory" ? "Personal notes" : route === "pocket" ? "Knowledge pockets" : "General knowledge");
  }
  let html = heading();
  html += `<div class="split-workspace"><div>${section("Decision level", "Select a level to apply it", choices)}</div><div>${section("When supervised", "Choose what can be saved automatically", `<div class="detail-list"><div class="detail-row"><span>Save to</span><span>${escapeHtml(destinations.join(", ") || "Review every destination")}</span></div><div class="detail-row"><span>Confidence needed</span><span>${Math.round((control.min_score ?? 0.9) * 100)}%</span></div><div class="detail-row"><span>Uncertain facts & changes</span><span>Ask you first</span></div></div>`)}<div class="insight-row">You can hold or reject individual facts in the learning queue.</div></div></div>`;
  if (state.advanced) {
    html += `<div class="section-label"><h2>Detailed settings</h2>${button("Access details", "authority", "text-button")}</div>${section("Decision settings", "Destinations, confidence and saved values", settingRows(state.data.control || control, "control"), button("Edit settings", "control-edit", "text-button"))}${section("Where it runs", "Configured location", `<div class="surface-body"><h3>${displayMode(state.data.system?.mode || control.mode)}</h3></div>`)}`;
  }
  return html;
}

async function selectAutonomy(level, approve = false) {
  const control = controlValues();
  if (control.autonomy === level && !state.autonomyIssue) return;
  state.autonomyIssue = null;
  try {
    await mutate(approve ? "POST" : "PUT", approve ? "/control/approve" : "/control", {
      expected_revision: revision(state.data.control || control),
      autonomy: level,
      auto_routes: control.auto_routes || ["memory"],
      min_score: control.min_score ?? 0.9,
    });
    toast("Decision level saved");
  } catch (error) {
    let message = "Couldn't save this level.";
    let buttonText = "Try again";
    let action = "autonomy-select";
    if (error.status === 403) {
      message = "Needs your approval to turn on.";
      buttonText = "Approve";
      action = "autonomy-approve";
    } else if (error.status === 401) {
      message = "Sign in to apply this level.";
      buttonText = "Sign in";
      action = "connection";
    } else if (error.status === 409) {
      message = "Settings changed. Refresh and try again.";
      buttonText = "Refresh";
      action = "refresh";
    }
    state.autonomyIssue = {level, message, button: buttonText, action};
    render();
  }
}

function renderLogs() {
  let rows = "";
  for (const entry of state.logs)
    rows += `<div class="log-row"><span class="muted">${escapeHtml(entry.time)}</span><span>${escapeHtml(entry.method)} ${entry.status}</span><span>${escapeHtml(entry.path)}</span></div>`;
  return (
    rows ||
    `<div class="surface-body muted small">No API operations in this browser session.</div>`
  );
}

function sessionTimeoutForm() {
  const current = state.session.session_seconds ?? 86400;
  const presets = [[900, "15 min"], [3600, "1 hour"], [28800, "8 hours"], [86400, "1 day"]];
  let choices = "";
  let preset = false;
  for (const [seconds, name] of presets) {
    if (seconds === current) preset = true;
    choices += `<option value="${seconds}" ${seconds === current ? "selected" : ""}>${name}</option>`;
  }
  if (state.advanced) choices += `<option value="custom" ${preset ? "" : "selected"}>Custom</option>`;
  else if (!preset) choices += `<option value="${current}" selected>${current / 60} min (custom)</option>`;
  const custom = state.advanced ? `<label for="timeout-minutes">Custom timeout (minutes)</label><input id="timeout-minutes" name="minutes" type="number" min="5" max="1440" step="1" value="${current / 60}">` : "";
  return `<form id="portal-timeout-form" class="portal-access-form" method="dialog"><label for="session-timeout">Session timeout</label><select id="session-timeout" name="seconds">${choices}</select>${custom}<p class="field-help">Used for new sign-ins. Activity does not extend a session.</p><div class="form-actions"><button class="primary" type="submit">Save timeout</button>${button("Sign out other sessions", "portal-logout-others")}</div></form>`;
}

async function saveSessionTimeout(form) {
  let seconds = Number(form.elements.seconds.value);
  if (form.elements.seconds.value === "custom") seconds = Number(form.elements.minutes.value) * 60;
  if (!Number.isInteger(seconds) || seconds < 300 || seconds > 86400) throw new Error("Choose a timeout from 5 minutes to 1 day.");
  state.session = await sessionRequest("settings", "PUT", {session_seconds: seconds});
  await refresh();
  if (state.error) throw new Error(state.error);
}

function portalAccessSettings() {
  let html = "";
  if (state.session?.hosted) {
    const locked = state.session.password_enabled === true;
    html += section("Security", locked ? "One password protects this portal" : "Anyone on this network can open the portal", `<div class="service-row"><div><h3>${locked ? "Lock on" : "Lock off"}</h3><p>This browser is signed in.</p></div>${button(locked ? "Change password" : "Set password", "portal-password")}${locked ? button("Sign out", "portal-logout") : ""}</div><div class="surface-body">${sessionTimeoutForm()}${state.advanced && locked ? button("Turn lock off", "portal-unlock") : ""}</div>`);
  }
  if (state.advanced) {
    html += section("Engine connection", "Connect to another engine", `<div class="surface-body"><p>Use its URL and API key. Keep the key private.</p>${button("Connect to an engine", "engine-connection")}${state.engineConnection?.connected ? button("Use this host's engine", "local-engine") : ""}</div>`);
  }
  return html;
}

function renderSystem() {
  const system = state.data.system || {};
  let html = heading(button("Refresh status", "refresh"));
  html += `<div class="system-layout"><div>${section("Workspace", "Runtime and API connection", `<div class="service-row"><div><h3>Admin API</h3><p>${state.demo ? "Fictional local demonstration" : "Same-origin authenticated connection"}</p></div>${pill(state.demo ? "Demo" : state.connected ? "Connected" : "Offline", state.connected && !state.demo ? "green" : "gray")}</div><div class="service-row"><div><h3>Funnel mode</h3><p>${displayMode(system.mode || controlValues().mode)}</p></div>${pill("Portable")}</div><div class="service-row"><div><h3>Model serving</h3><p>${escapeHtml(system.model || "Owned by the configured runtime")}</p></div>${pill("Separate owner", "gray")}</div>`)}${portalAccessSettings()}</div><div>${section("Traffic collection", "Development policy", `<div class="service-row"><div><h3>Traffic tap</h3><p>Off and undeployed</p></div>${pill("Off", "gray")}</div><div class="detail-list"><div class="detail-row"><span>Per-user exclusions</span><span>Required before activation</span></div><div class="detail-row"><span>Approved time window</span><span>Required before activation</span></div><div class="detail-row"><span>Testing authority</span><span>Explicit owner go</span></div></div>`, button("Policy details", "help-tap", "text-button"))}<div class="insight-row">This portal manages the learning workspace. Traffic collection remains inactive during development.</div></div></div>`;
  if (state.advanced) {
    html += `<div class="section-label"><h2>System values</h2><div class="heading-actions">${button("API explorer", "api-explorer")}${button("Export values", "system-export")}</div></div>${section("Runtime readback", "Every field returned by the admin API", settingRows(system, "system"), button("Raw JSON", "system-raw", "text-button"))}<div class="section-label"><h2>Session activity</h2>${button("Export log", "logs-export", "text-button")}</div>${section("Request log", "Methods, paths and outcomes; credentials are excluded", `<div class="log-list" role="region" aria-label="Recent API requests" tabindex="0">${renderLogs()}</div>`)}`;
  }
  return html;
}

function render() {
  document.title = "RSI Dreambox | " + titles[state.page];
  for (const link of document.querySelectorAll("nav [data-page]")) {
    if (link.dataset.page === state.page)
      link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  document
    .querySelector('[data-action="simple"]')
    .setAttribute("aria-pressed", String(!state.advanced));
  document
    .querySelector('[data-action="advanced"]')
    .setAttribute("aria-pressed", String(state.advanced));
  document.getElementById("connection-label").textContent = state.demo
    ? "Demo workspace"
    : state.connected
      ? "Workspace connected"
      : "Workspace waiting";
  document.getElementById("connection-dot").className =
    "status-dot " + (state.demo ? "demo" : state.connected ? "" : "quiet");
  const note = document.getElementById("mode-note");
  note.hidden = !state.demo;
  note.textContent =
    "Demo data. Changes stay in this browser; no model, server or GPU is affected.";
  document.getElementById("admin-label").textContent = state.demo
    ? "Demo admin"
    : state.connected
      ? "Portal settings"
      : "Portal access";
  const main = document.getElementById("main");
  let content = "";
  if (state.error)
    content += `<div role="alert" class="error-banner">${escapeHtml(state.error)} ${state.connected ? button("Dismiss notice", "error-dismiss", "text-button") : button("Connect", "connection", "text-button")}</div>`;
  if (!state.data) content += renderWelcome();
  else {
    const renderers = {
      rules: renderRules,
      queue: renderQueue,
      dreams: renderDreams,
      pockets: renderPockets,
      autonomy: renderAutonomy,
      system: renderSystem,
    };
    content += renderers[state.page]();
  }
  main.innerHTML = content;
  let pending = 0;
  for (const item of collection("queue"))
    if (item.status === "pending") pending += 1;
  document.getElementById("queue-count").textContent = pending;
  document.getElementById("footer-status").textContent = state.demo
    ? "Demo workspace | Simple and Advanced share the same state"
    : state.connected
      ? "Authenticated local workspace | " +
        displayMode(state.data?.system?.mode || controlValues().mode)
      : "Waiting for your workspace";
  for (const bar of document.querySelectorAll(".capacity-fill")) {
    const percent = Number(bar.dataset.capacity)
      ? (100 * Number(bar.dataset.used)) / Number(bar.dataset.capacity)
      : 0;
    bar.style.width = Math.min(100, percent) + "%";
  }
}

function openDrawer(title, body, kicker = titles[state.page]) {
  document.getElementById("drawer-title").textContent = title;
  document.getElementById("drawer-kicker").textContent = kicker;
  document.getElementById("drawer-body").innerHTML = body;
  const drawer = document.getElementById("drawer");
  if (!drawer.open) drawer.showModal();
}

function toast(message) {
  const element = document.getElementById("toast");
  element.textContent = message;
  element.hidden = false;
  setTimeout(() => {
    element.hidden = true;
  }, 4000);
}

function exportJson(filename, value, jsonText = null) {
  const content = jsonText ?? JSON.stringify(value, null, 2) + "\n";
  const blob = new Blob([content], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
  toast("Exported " + filename);
}

function showRaw(title, value) {
  openDrawer(
    title,
    `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`,
    "Advanced",
  );
}

function schema(name) {
  return state.spec?.components?.schemas?.[name] || null;
}

function fieldHtml(name, value, definition = {}) {
  const title = escapeHtml(label(name));
  const id = "field-" + name;
  let field;
  const type = Array.isArray(definition.type)
    ? definition.type[0]
    : definition.type;
  if (type === "boolean" || typeof value === "boolean")
    field = `<label class="checkbox-line"><input id="${id}" name="${name}" type="checkbox" ${value ? "checked" : ""}>${title}</label>`;
  else if (definition.enum) {
    field = `<label for="${id}">${title}</label><select id="${id}" name="${name}">`;
    for (const choice of definition.enum)
      field += `<option value="${escapeHtml(choice)}" ${choice === value ? "selected" : ""}>${escapeHtml(choice)}</option>`;
    field += `</select>`;
  } else if (type === "array" && definition.items?.enum) {
    field = `<fieldset data-array-name="${name}"><legend>${title}</legend>`;
    for (const choice of definition.items.enum) {
      field += `<label class="checkbox-line"><input type="checkbox" value="${escapeHtml(choice)}" ${value?.includes(choice) ? "checked" : ""}>${escapeHtml(choice)}</label>`;
    }
    field += `</fieldset>`;
  } else if (
    type === "array" ||
    type === "object" ||
    (value !== null && typeof value === "object")
  )
    field = `<label for="${id}">${title}</label><textarea id="${id}" name="${name}" data-json="true">${value === undefined && definition.default === undefined ? "" : escapeHtml(JSON.stringify(value ?? definition.default, null, 2))}</textarea>`;
  else if (type === "integer" || type === "number" || typeof value === "number")
    field = `<label for="${id}">${title}</label><input id="${id}" name="${name}" type="number" step="${type === "integer" ? "1" : "any"}" ${definition.minimum !== undefined ? `min="${definition.minimum}"` : ""} ${definition.maximum !== undefined ? `max="${definition.maximum}"` : ""} value="${escapeHtml(value ?? definition.default ?? "")}">`;
  else if (
    definition.multiline ||
    name.includes("instructions") ||
    name === "routing_rules" ||
    name === "prompt_evidence" ||
    name === "text" ||
    name === "notes"
  )
    field = `<label for="${id}">${title}</label><textarea id="${id}" name="${name}">${escapeHtml(value ?? "")}</textarea>`;
  else
    field = `<label for="${id}">${title}</label><input id="${id}" name="${name}" type="text" value="${escapeHtml(value ?? definition.default ?? "")}">`;
  if (definition.description)
    field += `<div class="field-help">${escapeHtml(definition.description)}</div>`;
  return field;
}

function openForm(title, schemaName, values, method, path, options = {}) {
  if (isDreamPath(path) && !dreamsEnabled()) {
    showDreamUnavailable();
    return;
  }
  const definition = schema(schemaName);
  if (!definition)
    throw new Error(
      "The API schema for this action is unavailable. Connect or refresh the API specification.",
    );
  state.form = { schemaName, values, method, path, options };
  let html = options.description
    ? `<p>${escapeHtml(options.description)}</p>`
    : "";
  html += `<form id="operation-form" method="dialog">`;
  const basic = options.basic || [
    "id",
    "name",
    "domain",
    "route",
    "pocket",
    "topic",
    "capacity",
    "run_id",
    "not_before",
    "preset",
  ];
  for (const [name, field] of Object.entries(definition.properties)) {
    if (name === "expected_revision") continue;
    const required = definition.required?.includes(name);
    if (!state.advanced && !basic.includes(name) && !required) continue;
    html += fieldHtml(name, values[name], field);
  }
  if (
    (!state.advanced && schemaName === "RuleCreate") ||
    (!state.advanced && schemaName === "RuleEdit")
  )
    html += `<div class="field-help">Advanced exposes user filters, source types, patterns, priority and notes.</div>`;
  html += `<div class="form-actions">${button("Cancel", "close")}<button class="primary" type="submit">${escapeHtml(options.saveLabel || "Save changes")}</button></div></form>`;
  openDrawer(title, html);
}

function parseForm(form, definition, previous) {
  const result = structuredClone(previous);
  for (const [name, field] of Object.entries(definition.properties)) {
    if (name === "expected_revision") continue;
    const choices = form.querySelector(`[data-array-name="${name}"]`);
    if (choices) {
      result[name] = [];
      for (const input of choices.querySelectorAll("input")) {
        if (input.checked) result[name].push(input.value);
      }
      continue;
    }
    const input = form.elements.namedItem(name);
    if (!input) continue;
    if (
      input.dataset.json &&
      !input.value.trim() &&
      !definition.required?.includes(name)
    ) {
      delete result[name];
      continue;
    }
    const type = Array.isArray(field.type) ? field.type[0] : field.type;
    let value;
    if (
      Array.isArray(field.type) &&
      field.type.includes("null") &&
      ["", "null"].includes(input.value.trim())
    )
      value = null;
    else if (input.type === "checkbox") value = input.checked;
    else if (input.dataset.json) value = JSON.parse(input.value);
    else if (type === "integer" || type === "number") {
      if (!input.value.trim())
        throw new Error(label(name) + " needs a number.");
      value = Number(input.value);
      if (
        !Number.isFinite(value) ||
        (type === "integer" && !Number.isInteger(value))
      )
        throw new Error(label(name) + " needs a valid " + type + ".");
    } else {
      value = input.value;
      if (Array.isArray(field.type) && field.type.includes("null") && !value)
        value = null;
    }
    result[name] = value;
  }
  const body = {};
  for (const name of Object.keys(definition.properties))
    if (result[name] !== undefined) body[name] = result[name];
  return body;
}

async function mutate(method, path, body, representation = {}) {
  if (isDreamPath(path) && !dreamsEnabled()) {
    throw new Error("feature_unavailable: under development");
  }
  if (state.demo) {
    const result = demoMutation(method, path, body);
    recordLog(method, path, 200);
    render();
    return result;
  }
  const result = await api(path, method, body, representation);
  await refresh();
  if (state.error) throw new Error(state.error);
  return result;
}

function demoMutation(method, path, body) {
  const parts = path.split("/");
  const domain = parts[1];
  const id = parts[2];
  if (domain === "dreams" && parts[3] === "check") {
    return {
      ready: false,
      reason: "Demo planning only; no native runner invoked",
      executes_training: false,
    };
  }
  if (domain === "rules" && id === "presets") {
    state.data.rules.push({
      id: body.id,
      name:
        body.preset === "persona"
          ? "Persona continuity"
          : "Companion preferences",
      domain: body.preset,
      enabled: true,
      route: body.preset === "persona" ? "pocket" : "memory",
      pocket: body.preset === "persona" ? body.id : null,
      action: "propose",
      revision: 1,
    });
  } else if (domain === "control")
    state.data.control = {
      ...body,
      revision: (state.data.control?.revision || 1) + 1,
    };
  else if (domain === "schedule")
    state.data.schedule = {
      ...body,
      revision: (state.data.schedule?.revision || 1) + 1,
    };
  else if (domain === "queue") {
    const item = findItem("queue", id);
    if (parts[3] === "approve")
      item.status =
        getGrade(item).verdict === "reject" ? "rejected" : "accepted";
    else if (parts[3] === "reject") item.status = "rejected";
    else {
      Object.assign(item, body);
      item.admin_review = { hold: body.hold || false, note: body.note || "" };
    }
  } else if (["rules", "dreams", "pockets"].includes(domain)) {
    const items = state.data[domain];
    if (method === "DELETE") {
      const index = items.findIndex((item) => item.id === id);
      if (index >= 0) items.splice(index, 1);
    } else if (method === "PATCH")
      Object.assign(findItem(domain, id), body, {
        revision: (findItem(domain, id).revision || 1) + 1,
      });
    else if (domain === "pockets" && parts[3] === "copy")
      items.push({
        ...structuredClone(findItem(domain, id)),
        id: body.id,
        name: body.id,
        revision: 1,
      });
    else if (domain === "pockets" && parts[3] === "rollover")
      items.push({
        id: body.id,
        topic: findItem(domain, id).topic,
        capacity: body.capacity,
        facts: [],
        used: 0,
        revision: 1,
      });
    else if (domain === "pockets" && id === "merge") {
      const facts = [];
      for (const source of body.sources)
        for (const fact of findItem(domain, source).facts || [])
          facts.push(fact);
      items.push({
        id: body.id,
        topic: findItem(domain, body.sources[0]).topic,
        capacity: body.capacity,
        facts,
        revision: 1,
      });
    } else if (domain === "pockets" && id === "import")
      items.push({ ...body.package, id: body.id, revision: 1, shared: true });
    else if (domain === "pockets" && parts[3] === "extract") {
      const pocket = findItem(domain, id);
      if (!pocket.facts) pocket.facts = [];
      for (const candidateId of body.candidate_ids) {
        const candidate = findItem("queue", candidateId);
        pocket.facts.push({
          id: candidateId,
          statement: candidate.text,
          provenance: candidate.source,
        });
      }
    } else
      items.push({
        ...body,
        name: body.name || body.id,
        revision: 1,
        facts: body.facts || [],
        status: domain === "dreams" ? "planned" : "staging",
      });
  }
}

function findItem(domain, id) {
  for (const item of collection(domain)) if (item.id === id) return item;
  throw new Error("This item is no longer available. Refresh the workspace.");
}

function revision(item) {
  return item.revision;
}

function editRule(id) {
  const rule = id
    ? findItem("rules", id)
    : {
        enabled: true,
        domain: "general",
        action: "propose",
        route: "pocket",
        priority: 100,
        user_ids: [],
        exclude_user_ids: [],
        source_kinds: ["prompt", "conversation", "manual"],
        patterns: [],
        notes: "",
        pocket: null,
      };
  const values = structuredClone(rule);
  if (id) values.expected_revision = revision(rule);
  openForm(
    id ? "Edit save rule" : "Create save rule",
    id ? "RuleEdit" : "RuleCreate",
    values,
    id ? "PATCH" : "POST",
    id ? "/rules/" + encodeURIComponent(id) : "/rules",
    {
      basic: ["id", "name", "enabled", "domain", "action", "route", "pocket"],
      description:
        "Choose the domain and destination. Changes are checked by the admin API.",
    },
  );
  if (id)
    document
      .getElementById("drawer-body")
      .insertAdjacentHTML(
        "beforeend",
        `<div class="form-actions">${button("Delete rule", "rule-delete", "danger", id)}</div>`,
      );
}

function learnPanel() {
  let fields = `<label for="learn-content">Knowledge to retain</label><textarea id="learn-content" name="content" rows="8" maxlength="1000000" placeholder="Paste a short note or choose a prepared file" required></textarea><label for="learn-file">Prepared file</label><input id="learn-file" type="file" accept=".txt,.md,.json,text/plain,application/json"><div class="field-help">UTF-8 text or a JSON facts pack. File contents appear above for review.</div>`;
  if (state.advanced) {
    fields += `<label for="learn-format">Input format</label><select id="learn-format" name="format"><option value="text">Text, GLM extracts the facts</option><option value="json">JSON, declared facts</option></select><label for="learn-source">Source reference</label><input id="learn-source" name="source_ref" maxlength="512" value="manual:portal" required><details><summary>JSON facts format</summary><pre>${escapeHtml(JSON.stringify({facts: [{statement: "Fictional rover Lumen has three sample bays.", prompt_evidence: "The Lumen specification lists three sample bays."}]}, null, 2))}</pre></details>`;
  } else {
    fields += `<input type="hidden" name="format" value="text"><input type="hidden" name="source_ref" value="manual:portal">`;
  }
  openDrawer("Learn this", `<p>Add knowledge for the model to review. It suggests what to keep and where to save it.</p><form id="learn-form" method="dialog">${fields}<p id="learn-progress" role="status" aria-live="polite"></p><div class="form-actions">${button("Cancel", "close")}<button class="primary" type="submit">Add for review</button></div></form>`, state.advanced ? "Advanced" : "Simple");
}

async function gradePrepared(items) {
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    if (item.status !== "pending" || item.proposal) continue;
    const progress = document.getElementById("learn-progress");
    if (progress) progress.textContent = `GLM is reviewing fact ${index + 1} of ${items.length}.`;
    const result = await api("/queue/" + encodeURIComponent(item.id) + "/grade", "POST", {expected_revision: item.revision});
    if (result.errors.length) {
      await refresh();
      throw new Error(result.errors[0].error);
    }
  }
  await refresh();
  if (state.error) throw new Error(state.error);
}

function preparedResult(items, queuedCount) {
  let html = `<p>${queuedCount} new facts queued. GLM review uses your normal approval policy.</p>`;
  for (const submitted of items) {
    const item = findItem("queue", submitted.id);
    const grade = getGrade(item);
    html += `<div class="evidence"><p>${escapeHtml(item.text)}</p><p>${escapeHtml(item.status)} · ${escapeHtml(grade.pocket || grade.route || "Not graded")}</p>${button("Review fact", "queue-detail", "secondary", item.id)}</div>`;
  }
  if (!items.length) html += "<p>No atomic facts were discovered. Review your source and try again.</p>";
  html += `<div class="form-actions">${button("Back to queue", "close", "primary")}</div>`;
  openDrawer("Prepared knowledge reviewed", html);
}

function queueDetail(id) {
  const item = findItem("queue", id);
  const grade = getGrade(item);
  let html = `<h3>Candidate fact</h3><p>${escapeHtml(item.text)}</p><h3>Source evidence</h3><div class="evidence">${escapeHtml(item.prompt_evidence || "No evidence supplied")}</div><div class="detail-list"><div class="detail-row"><span>Source</span><span>${escapeHtml(item.source?.ref)}</span></div><div class="detail-row"><span>Proposed destination</span><span>${escapeHtml(grade.pocket || grade.route || "Not graded")}</span></div><div class="detail-row"><span>Confidence</span><span>${grade.score === undefined ? "Not graded" : Math.round(grade.score * 100) + "%"}</span></div><div class="detail-row"><span>Status</span><span>${escapeHtml(item.status)}</span></div></div><h3>Model's reason</h3><p>${escapeHtml(grade.reason || "Waiting for grading")}</p>`;
  if (item.status === "pending") {
    let next = button(grade.verdict === "reject" ? "Confirm rejection" : "Approve", "queue-approve", "primary", id);
    if (item.source?.kind === "manual" && !item.proposal) {
      next = button("Grade fact", "queue-grade", "primary", id);
    }
    html += `<div class="form-actions">${button("Reject", "queue-reject", "danger", id)}${next}</div>`;
  }
  html += `<div class="form-actions">${button("Edit or hold", "queue-edit", "secondary", id)}${state.advanced ? button("Raw record", "queue-raw", "secondary", id) : ""}</div>`;
  openDrawer("Review knowledge", html);
}

function confirmAction(
  title,
  description,
  method,
  path,
  body,
  labelText = "Confirm",
) {
  state.pending = { method, path, body, title };
  openDrawer(
    title,
    `<p>Review this action before it is sent to the admin API.</p><div class="confirm-description">${escapeHtml(description)}</div><div class="form-actions">${button("Cancel", "close")}${button(labelText, "confirm", "primary")}</div>`,
  );
}

function requestQueueDecision(id, decision) {
  const item = findItem("queue", id);
  const path = "/queue/" + encodeURIComponent(id) + "/" + decision;
  if (
    !state.demo &&
    decision === "reject" &&
    !state.spec?.paths?.["/api/v1/queue/{id}/reject"]
  ) {
    openDrawer(
      "Reject knowledge",
      `<p>This API version supports holding a fact and approving the model's existing proposal. Explicit admin rejection needs the reject operation.</p>${button("Hold for review", "queue-hold", "primary", id)}`,
    );
    return;
  }
  if (decision === "reject") {
    openForm(
      "Reject knowledge",
      "Reject",
      {
        expected_revision: revision(item),
        reason: "Not approved for retention.",
      },
      "POST",
      path,
      {
        basic: ["reason"],
        saveLabel: "Reject knowledge",
        description: item.text,
      },
    );
    return;
  }
  confirmAction(
    decision === "approve" ? "Approve knowledge" : "Reject knowledge",
    item.text,
    "POST",
    path,
    { expected_revision: revision(item) },
    decision === "approve" ? "Approve" : "Reject",
  );
}

function pocketDetail(id) {
  const pocket = findItem("pockets", id);
  let facts = "";
  for (const fact of pocket.facts || [])
    facts += `<li>${escapeHtml(fact.statement)}</li>`;
  if (!facts)
    for (const name of pocket.members || [])
      facts += `<li>${escapeHtml(name)}</li>`;
  let html = `<p>${escapeHtml(pocket.topic)}</p><div class="detail-list"><div class="detail-row"><span>Knowledge</span><span>${pocketUsed(pocket)} facts</span></div><div class="detail-row"><span>Capacity</span><span>${pocket.capacity}</span></div><div class="detail-row"><span>State</span><span>Raw-fact staging</span></div></div><h3>Contents</h3><ul>${facts || "<li>This pocket is empty.</li>"}</ul><div class="form-actions">${button("Copy", "pocket-copy", "secondary", id)}${button("Share package", "pocket-export", "primary", id)}</div>`;
  if (state.advanced)
    html += `<h3>Pocket operations</h3><div class="heading-actions">${button("Extract accepted facts", "pocket-extract", "secondary", id)}${button("Rollover", "pocket-rollover", "secondary", id)}</div><div class="form-actions">${button("Raw values", "pocket-raw", "secondary", id)}${button("Remove pocket", "pocket-delete", "danger", id)}</div>`;
  else
    html += `<div class="field-help">Advanced adds extraction, rollover, merge, import and removal.</div>`;
  openDrawer(pocket.name || pocket.id, html);
}

async function editPolicyField(path) {
  const policy = state.policyDraft || state.data.policy;
  if (!policy) throw new Error("Import a policy before editing it.");
  const parts = path.split(".");
  const group = parts.length === 2 ? parts[0] : null;
  const name = parts.at(-1);
  const value = group ? policy[group][name] : policy[name];
  const content = await loadManual();
  const field = group ? path : "policy." + name;
  let help = "";
  for (const topic of content.topics) {
    if (topic.fields.includes(field)) {
      help = topic.body;
      break;
    }
  }
  let definition = {};
  const policySchema = schema("PolicyEdit")?.properties?.policy;
  if (group)
    definition = structuredClone(
      policySchema?.properties?.[group]?.properties?.[name] || {},
    );
  else definition = structuredClone(policySchema?.properties?.[name] || {});
  definition.multiline =
    name.includes("instructions") || name === "routing_rules";
  if (!definition.type) {
    if (Array.isArray(value)) definition.type = "array";
    else if (typeof value === "number") definition.type = "number";
    else if (value === null) definition.type = ["string", "null"];
    else definition.type = "string";
  }
  state.form = { policyPath: path, definition };
  openDrawer(
    label(name),
    `<p>${escapeHtml(help || "Set the raw value for the next-run policy.")}</p><form id="policy-field-form" method="dialog">${fieldHtml("value", value, definition)}<div class="field-help">${escapeHtml(path)}${value === null ? " | Blank uses the default; JSON null is accepted." : ""}</div><div class="form-actions">${button("Cancel", "close")}<button class="primary" type="submit">Save draft value</button></div></form>`,
    "Advanced policy",
  );
}

let manualContent = null;

async function loadManual() {
  if (manualContent) return manualContent;
  const response = await fetch("help.json", { cache: "no-store" });
  if (!response.ok)
    throw new Error(
      "The user manual could not be loaded. Reload the portal and try again.",
    );
  const content = await response.json();
  if (
    content.schema_version !== 1 ||
    !Array.isArray(content.pages) ||
    !Array.isArray(content.topics)
  ) {
    throw new Error("The user manual format is unsupported.");
  }
  manualContent = content;
  return content;
}

function manualPageHtml(content, page, full) {
  if (page.id === "dreams" && !dreamsEnabled()) {
    return `<div data-manual-tasks>${dreamUnavailableHtml()}</div>`;
  }
  let html = `<div data-manual-tasks><p>${escapeHtml(page.intro)}</p>`;
  const layers = full
    ? ["simple", "advanced"]
    : [state.advanced ? "advanced" : "simple"];
  for (const layer of layers) {
    html += `<h4>${layer === "simple" ? "Simple" : "Advanced"}</h4>`;
    for (const section of page[layer]) {
      html += `<h4>${escapeHtml(section.title)}</h4><p>${escapeHtml(section.body)}</p>`;
    }
  }
  html += "</div><h4>Settings and details</h4>";
  for (const topicId of page.topic_ids) {
    for (const topic of content.topics) {
      if (topic.id === topicId) {
        html += `<details data-manual-topic data-manual-fields="${escapeHtml(topic.fields.join(" "))}"><summary>${escapeHtml(topic.label)}</summary><p>${escapeHtml(topic.body)}</p></details>`;
        break;
      }
    }
  }
  return html;
}

async function showHelp(name) {
  try {
    const content = await loadManual();
    if (name === "tap") {
      for (const topic of content.topics) {
        if (topic.id === "traffic-policy") {
          openDrawer(
            topic.label,
            `<p>${escapeHtml(topic.body)}</p>${button("Open user manual", "manual", "primary")}`,
            "Help",
          );
          return;
        }
      }
    }
    for (const page of content.pages) {
      if (page.id === name) {
        openDrawer(
          page.title,
          manualPageHtml(content, page, false) +
            button("Open user manual", "manual", "primary"),
          "Help",
        );
        return;
      }
    }
    throw new Error("Help for this page is unavailable.");
  } catch (error) {
    showError(error);
  }
}

async function showManual() {
  try {
    const content = await loadManual();
    let navigation = "";
    let sections =
      "<p data-manual-overview>Simple opens the everyday tasks. Advanced opens full settings and raw details. Both use the same workspace and authority.</p>";
    for (const page of content.pages) {
      navigation += `<a href="#manual-${escapeHtml(page.id)}" data-manual-anchor="${escapeHtml(page.id)}">${escapeHtml(page.title)}</a>`;
      sections += `<section id="manual-${escapeHtml(page.id)}" class="manual-section" data-manual-section><h3>${escapeHtml(page.title)}</h3>${manualPageHtml(content, page, true)}${button("Open " + page.title, "navigate", "text-button", page.id)}</section>`;
    }
    openDrawer(
      "User manual",
      `<input class="manual-search" id="manual-search" aria-label="Search user manual" placeholder="Search the guide"><nav class="manual-nav" data-manual-navigation aria-label="Manual topics">${navigation}</nav>${sections}`,
      "Simple and Advanced",
    );
  } catch (error) {
    showError(error);
  }
}

function passwordForm(setup = false) {
  const action = setup ? "Set password" : "Sign in";
  const autocomplete = setup ? "new-password" : "current-password";
  return `<img class="portal-hero" src="hero.png" alt="RSI Dreambox" width="128" height="128"><form id="portal-password-form" class="portal-access-form" method="dialog" data-setup="${setup}"><label for="portal-password">Password</label><input id="portal-password" name="password" type="password" autocomplete="${autocomplete}" required><div class="form-actions"><button class="primary" type="submit">${action}</button></div></form>`;
}

function renderWelcome() {
  if (state.session?.hosted) {
    const setup = state.session.setup_required === true;
    return `<div class="portal-welcome">${section(setup ? "Set a portal password" : "Open RSI Dreambox", setup ? "One password protects this portal." : "Sign in once. Your browser keeps you signed in.", `<div class="surface-body">${passwordForm(setup)}</div>`)}</div>`;
  }
  const connect = state.advanced ? button("Connect to an engine", "connection", "primary") : button("Refresh", "refresh");
  return heading() + section("Your workspace is waiting", "Open RSI Dreambox on your engine to get started", empty("RSI Dreambox", "Advanced has connection settings for another engine.", connect + `<a class="secondary" href="?demo=1#rules">Open demo</a>`));
}

async function connectionPanel(remote = false) {
  if (state.session?.hosted && !state.session.authenticated) {
    openDrawer(state.session.setup_required ? "Set a portal password" : "Sign in", passwordForm(state.session.setup_required === true));
    return;
  }
  if (state.session?.hosted && !remote) {
    openDrawer("Portal settings", portalAccessSettings());
    return;
  }
  if (!state.advanced) {
    openDrawer("Portal access", `<p>Open RSI Dreambox on your engine. Advanced has connection settings for another engine.</p>`);
    return;
  }
  if (state.session?.hosted) state.engineConnection = await sessionRequest("connection");
  openDrawer("Connect to an engine", `<p>Use your engine address and its private API key.</p><form id="connection-form" method="dialog"><label for="engine-url">Engine URL</label><input id="engine-url" name="url" type="url" value="${escapeHtml(state.engineConnection?.url || location.origin || "")}" required><label for="api-token">API key</label><input id="api-token" name="token" type="password" autocomplete="off" required><div class="form-actions">${button("Cancel", "close")}<button class="primary" type="submit">Connect</button></div></form>`, "Advanced");
}

function scheduleEditor() {
  const value = scheduleValues();
  const window = value.windows?.[0] || {
    days: [0, 1, 2, 3, 4],
    start: "22:00",
    end: "23:30",
  };
  let days = "";
  const names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
  ];
  for (let index = 0; index < names.length; index += 1) {
    days += `<label class="checkbox-line"><input type="checkbox" name="day-${index}" ${window.days.includes(index) ? "checked" : ""}>${names[index]}</label>`;
  }
  state.form = { schedule: structuredClone(value) };
  openDrawer(
    "Dream schedule",
    `<p>Choose when the workspace may prepare a dream.</p><form id="schedule-form" method="dialog"><label class="checkbox-line"><input type="checkbox" name="enabled" ${value.enabled ? "checked" : ""}>Enable scheduling</label><label for="schedule-timezone">Time zone</label><input id="schedule-timezone" name="timezone" value="${escapeHtml(value.timezone || "America/New_York")}" required><fieldset><legend>Approved days</legend>${days}</fieldset><div class="time-fields"><div><label for="window-start">Start</label><input id="window-start" type="time" name="start" value="${window.start}" required></div><div><label for="window-end">End</label><input id="window-end" type="time" name="end" value="${window.end}" required></div></div><label for="schedule-idle">Idle for at least (minutes)</label><input id="schedule-idle" name="idle" type="number" min="0" step="1" value="${(value.min_idle_seconds ?? 900) / 60}" required><div class="field-help">Advanced exposes all windows and the active-request limit. Additional windows stay unchanged.</div><div class="form-actions">${button("Cancel", "close")}<button type="submit" class="primary">Save schedule</button></div></form>`,
  );
}

function showError(error) {
  toast(error.message);
  if (document.getElementById("drawer").open) {
    let alert = document.getElementById("form-error");
    if (!alert) {
      alert = document.createElement("div");
      alert.id = "form-error";
      alert.className = "error-banner";
      alert.setAttribute("role", "alert");
      document.getElementById("drawer-body").prepend(alert);
    }
    alert.textContent = error.message;
    if (error.status === 409)
      alert.insertAdjacentHTML(
        "beforeend",
        button("Refresh & close", "conflict-refresh"),
      );
  } else {
    state.error = error.message;
    render();
  }
}

async function handleAction(action, id) {
  if ((action.startsWith("dream-") || action.startsWith("schedule-")) && !dreamsEnabled()) {
    showDreamUnavailable();
    return;
  }
  if (action === "simple" || action === "advanced") {
    state.advanced = action === "advanced";
    render();
  } else if (action === "close") document.getElementById("drawer").close();
  else if (action === "error-dismiss") {
    state.error = "";
    render();
  } else if (action === "conflict-refresh") {
    document.getElementById("drawer").close();
    state.form = null;
    state.pending = null;
    await refresh();
  } else if (action === "navigate") {
    document.getElementById("drawer").close();
    location.hash = id;
  } else if (action === "help" || action.startsWith("help-"))
    showHelp(action === "help" ? state.page : action.slice(5));
  else if (action === "manual") showManual();
  else if (action === "connection") await connectionPanel();
  else if (action === "engine-connection") await connectionPanel(true);
  else if (action === "local-engine") {
    state.engineConnection = await sessionRequest("connection", "DELETE");
    await refresh();
  }
  else if (action === "portal-password") openDrawer("Set a portal password", passwordForm(true));
  else if (action === "portal-logout-others") {
    state.session = await sessionRequest("logout-others", "POST", {});
    toast("Other sessions signed out");
  }
  else if (action === "portal-unlock" && state.advanced) openDrawer("Turn lock off", `<p>Anyone on this network will be able to open the portal.</p><form id="portal-unlock-form" method="dialog"><div class="form-actions">${button("Cancel", "close")}<button type="submit" class="primary">Turn lock off</button></div></form>`, "Advanced");
  else if (action === "portal-logout") {
    state.session = await sessionRequest("logout", "POST", {});
    state.data = null;
    state.spec = null;
    document.getElementById("drawer").close();
    await refresh();
  }
  else if (action === "refresh") await refresh();
  else if (action === "learn-open") learnPanel();
  else if (action === "queue-grade") {
    if (state.demo) throw new Error("GLM grading needs a connected admin API. Demo facts remain fictional.");
    const item = findItem("queue", id);
    await gradePrepared([item]);
    queueDetail(id);
  }
  else if (action === "rule-create") editRule(null);
  else if (action === "rule-edit") editRule(id);
  else if (action === "rule-delete")
    confirmAction(
      "Delete rule",
      findItem("rules", id).name,
      "DELETE",
      "/rules/" + encodeURIComponent(id),
      { expected_revision: revision(findItem("rules", id)) },
      "Delete rule",
    );
  else if (action === "preset")
    openForm(
      "Add " + id + " preset",
      "Preset",
      { preset: id, id: id + "-preferences" },
      "POST",
      "/rules/presets",
      { saveLabel: "Add preset" },
    );
  else if (action === "queue-detail") queueDetail(id);
  else if (action === "queue-raw")
    showRaw("Candidate record", findItem("queue", id));
  else if (action === "queue-approve" || action === "queue-reject")
    requestQueueDecision(id, action.slice(6));
  else if (action === "queue-edit" || action === "queue-hold") {
    const item = findItem("queue", id);
    openForm(
      "Edit candidate review",
      "QueueEdit",
      {
        expected_revision: revision(item),
        text: item.text,
        prompt_evidence: item.prompt_evidence,
        hold:
          action === "queue-hold" || !!item.admin_review?.hold || !!item.hold,
        note: item.admin_review?.note || item.note || "",
      },
      "PATCH",
      "/queue/" + encodeURIComponent(id),
      {
        basic: ["text", "prompt_evidence", "hold", "note"],
        description:
          "Changing the fact or source evidence invalidates its old proposal. Hold keeps it out of the next staging handoff.",
      },
    );
  } else if (action === "select-visible") {
    for (const item of queueRows())
      if (item.status === "pending") state.selected.add(item.id);
    render();
  } else if (action.startsWith("bulk-")) {
    if (!state.selected.size) return;
    state.pending = { bulk: action.slice(5), ids: Array.from(state.selected) };
    openDrawer(
      "Review bulk action",
      `<p>${state.selected.size} selected facts. Each change uses its current revision and the same admin authority as an individual action.</p><div class="form-actions">${button("Cancel", "close")}${button("Confirm " + action.slice(5), "confirm", "primary")}</div>`,
    );
  } else if (action === "confirm") {
    if (!state.pending) return;
    const pending = state.pending;
    if (pending.bulk) {
      for (const candidateId of pending.ids) {
        const item = findItem("queue", candidateId);
        const suffix = pending.bulk === "hold" ? "" : "/" + pending.bulk;
        await mutate(
          pending.bulk === "hold" ? "PATCH" : "POST",
          "/queue/" + encodeURIComponent(candidateId) + suffix,
          {
            expected_revision: revision(item),
            ...(pending.bulk === "hold"
              ? { hold: true }
              : pending.bulk === "reject"
                ? { reason: "Admin bulk rejection." }
                : {}),
          },
        );
        state.selected.delete(candidateId);
      }
    } else await mutate(pending.method, pending.path, pending.body);
    state.pending = null;
    document.getElementById("drawer").close();
    render();
    toast(state.demo ? "Demo change saved" : "Action recorded");
  } else if (action === "schedule-edit") {
    if (!state.advanced) {
      scheduleEditor();
      return;
    }
    const value = scheduleValues();
    openForm(
      "Edit dream schedule",
      "Schedule",
      {
        expected_revision: revision(value),
        enabled: value.enabled || false,
        timezone: value.timezone || "America/New_York",
        windows: value.windows || [],
        min_idle_seconds: value.min_idle_seconds ?? value.idle_seconds ?? 900,
        max_active_requests: value.max_active_requests ?? 0,
      },
      "PUT",
      "/schedule",
      {
        description:
          "Day numbers run from Monday (0) to Sunday (6). The API checks every approved window.",
      },
    );
  } else if (action === "schedule-raw")
    showRaw("Raw schedule", scheduleValues());
  else if (action === "dream-create")
    openForm(
      "Prepare a development plan",
      "DreamCreate",
      {
        id: "dream-" + Date.now(),
        run_id: "",
        not_before: new Date().toISOString(),
      },
      "POST",
      "/dreams",
      {
        saveLabel: "Prepare dream",
        description:
          "Under development. Choose an accepted staged run for a manual plan. This API performs no neuron training.",
      },
    );
  else if (action === "dream-detail") {
    const dream = findItem("dreams", id);
    openDrawer(
      "Dream plan",
      `<pre>${escapeHtml(JSON.stringify(dream, null, 2))}</pre><div class="form-actions">${button("Cancel dream", "dream-cancel", "secondary", id)}${button("Check readiness", "dream-check", "primary", id)}</div>`,
    );
  } else if (action === "dream-check")
    openForm(
      "Check dream readiness",
      "DreamCheck",
      { at: new Date().toISOString(), idle_seconds: 900, active_requests: 0 },
      "POST",
      "/dreams/" + encodeURIComponent(id) + "/check",
      { saveLabel: "Check readiness" },
    );
  else if (action === "dream-cancel")
    confirmAction(
      "Cancel dream",
      "Remove this pending plan from the schedule.",
      "DELETE",
      "/dreams/" + encodeURIComponent(id),
      { expected_revision: revision(findItem("dreams", id)) },
      "Cancel dream",
    );
  else if (action === "pocket-detail") pocketDetail(id);
  else if (action === "pocket-create")
    openForm(
      "Create a pocket",
      "PocketCreate",
      { id: "", topic: "", capacity: 96 },
      "POST",
      "/pockets",
      { saveLabel: "Create pocket" },
    );
  else if (action === "pocket-copy")
    openForm(
      "Copy knowledge",
      "PocketCopy",
      { id: "" },
      "POST",
      "/pockets/" + encodeURIComponent(id) + "/copy",
      {
        description:
          "Choose a new identifier. In Advanced, a blank fact list copies the whole pocket. Enter a nonempty array of identifiers to select facts.",
      },
    );
  else if (action === "pocket-rollover")
    openForm(
      "Rollover pocket",
      "PocketRollover",
      {
        expected_revision: revision(findItem("pockets", id)),
        id: id + "-next",
        capacity: findItem("pockets", id).capacity,
      },
      "POST",
      "/pockets/" + encodeURIComponent(id) + "/rollover",
    );
  else if (action === "pocket-extract")
    openForm(
      "Extract accepted facts",
      "PocketExtract",
      {
        expected_revision: revision(findItem("pockets", id)),
        candidate_ids: [],
      },
      "POST",
      "/pockets/" + encodeURIComponent(id) + "/extract",
      {
        basic: ["candidate_ids"],
        description:
          "Only accepted, matching-topic candidates can be copied into this pocket.",
      },
    );
  else if (action === "pocket-merge")
    openForm(
      "Merge same-topic pockets",
      "PocketMerge",
      { id: "", sources: [], capacity: 96 },
      "POST",
      "/pockets/merge",
      { basic: ["id", "sources", "capacity"] },
    );
  else if (action === "pocket-delete")
    confirmAction(
      "Remove pocket",
      "Remove this admin pocket. Its existing exported packages remain available.",
      "DELETE",
      "/pockets/" + encodeURIComponent(id),
      { expected_revision: revision(findItem("pockets", id)) },
      "Remove pocket",
    );
  else if (action === "pocket-raw")
    showRaw("Pocket values", findItem("pockets", id));
  else if (action === "pocket-export") {
    if (state.demo) {
      exportJson(id + "-demo.json", {
        kind: "demo_snapshot",
        pocket: findItem("pockets", id),
      });
    } else {
      const packageText = await api(
        "/pockets/" + encodeURIComponent(id) + "/export",
        "GET",
        null,
        { responseText: true },
      );
      exportJson(id + ".json", null, packageText);
    }
  } else if (action === "pocket-import") openImport("pocket");
  else if (action === "autonomy-select") await selectAutonomy(id);
  else if (action === "autonomy-approve") await selectAutonomy(id, true);
  else if (action === "control-edit") {
    const control = controlValues();
    openForm(
      "Decision settings",
      "Control",
      {
        expected_revision: revision(state.data.control || control),
        autonomy: control.autonomy || "manual",
        auto_routes: control.auto_routes || ["memory"],
        min_score: control.min_score ?? 0.9,
      },
      "PUT",
      "/control",
      {description: "Choose which facts can be saved automatically."},
    );
  } else if (action === "authority")
    showRaw("Access details", state.data.authority || state.data.control?.permission || {});
  else if (action === "policy-field") await editPolicyField(id);
  else if (action === "policy-save") {
    const policy = state.policyDraft || state.data.policy;
    if (!policy) throw new Error("Load a policy before saving it.");
    if (state.demo) {
      state.data.policy = structuredClone(policy);
      state.policyDraft = null;
      recordLog("PUT", "/policy", 200);
      render();
    } else {
      await api("/policy", "PUT", {
        expected_revision: revision(state.data.policyProfile),
        policy,
      });
      state.policyDraft = null;
      await refresh();
    }
    toast(state.demo ? "Demo policy saved" : "Next-workspace policy saved");
  } else if (action === "policy-import") openImport("policy");
  else if (action === "policy-export") {
    const policy = state.policyDraft || state.data.policy;
    if (!policy) throw new Error("Import a policy file first.");
    exportJson("next-run-policy.json", policy);
  } else if (action === "policy-raw")
    showRaw("Raw policy", state.policyDraft || state.data.policy);
  else if (action === "queue-export")
    exportJson("learning-queue.json", collection("queue"));
  else if (action === "dream-export")
    exportJson("dream-plans.json", {
      schedule: scheduleValues(),
      dreams: collection("dreams"),
    });
  else if (action === "system-export")
    exportJson("system-values.json", state.data.system);
  else if (action === "logs-export")
    exportJson("admin-session-log.json", state.logs);
  else if (action === "system-raw")
    showRaw("System readback", state.data.system);
  else if (action === "api-explorer") showApiExplorer();
}

function openImport(kind) {
  openDrawer(
    kind === "policy" ? "Import policy" : "Import shared pocket",
    `<p>${kind === "policy" ? "Open a version 1 policy file as a next-run draft." : "Open a shareable raw-fact package. The API checks its provenance and content."}</p><label for="import-file">JSON file</label><input id="import-file" type="file" accept=".json,application/json" data-kind="${kind}"><div class="field-help">Private facts stay in your operator workspace. No import reaches model serving.</div>`,
  );
}

function showApiExplorer() {
  if (!state.spec)
    throw new Error("Connect or refresh to load the API specification.");
  let html = `<p>Authenticated API operations for this workspace. Request bodies are available in the documented schema.</p><div class="manual-nav">`;
  for (const [path, methods] of Object.entries(state.spec.paths)) {
    for (const [method, operation] of Object.entries(methods)) {
      const unavailable = isDreamPath(path) && !dreamsEnabled();
      html += `<button type="button" class="secondary" data-action="api-operation" data-id="${escapeHtml(method + " " + path)}">${escapeHtml(method.toUpperCase() + " " + path)}${unavailable ? " | Under development" : ""}</button>`;
    }
  }
  html += `</div>${button("Export OpenAPI", "api-export")}`;
  openDrawer("API explorer", html, "Advanced");
}

document.addEventListener("click", async (event) => {
  const anchor = event.target.closest("[data-manual-anchor]");
  if (anchor) {
    event.preventDefault();
    document
      .getElementById("manual-" + anchor.dataset.manualAnchor)
      ?.scrollIntoView();
    return;
  }
  const target = event.target.closest("[data-action]");
  if (!target || state.busy) return;
  const action = target.dataset.action;
  try {
    state.busy = true;
    target.disabled = true;
    if (action === "api-export")
      exportJson("rsi-admin-openapi.json", state.spec);
    else if (action === "api-operation") {
      const [method, path] = target.dataset.id.split(" ");
      if (isDreamPath(path) && !dreamsEnabled()) {
        showRaw("Under development: " + method.toUpperCase() + " " + path,
          state.spec.paths[path][method]);
        return;
      }
      showRaw(
        method.toUpperCase() + " " + path,
        state.spec.paths[path][method],
      );
    } else await handleAction(action, target.dataset.id);
  } catch (error) {
    showError(error);
  } finally {
    state.busy = false;
    target.disabled = false;
  }
});

document.addEventListener("submit", async (event) => {
  // An input named id can shadow HTMLFormElement.id.
  const formId = event.target.getAttribute("id");
  if (
    ![
      "connection-form",
      "portal-password-form",
      "portal-timeout-form",
      "portal-unlock-form",
      "operation-form",
      "policy-field-form",
      "schedule-form",
      "learn-form",
    ].includes(formId)
  )
    return;
  event.preventDefault();
  if (state.busy) return;
  try {
    state.busy = true;
    if (formId === "connection-form") {
      const engineUrl = new URL(event.target.elements.url.value);
      if (!["http:", "https:"].includes(engineUrl.protocol) || engineUrl.username || engineUrl.password || engineUrl.pathname !== "/" || engineUrl.search || engineUrl.hash) throw new Error("Use an HTTP or HTTPS engine address without a path.");
      if (state.session?.hosted) {
        state.engineConnection = await sessionRequest("connection", "PUT", {url: engineUrl.origin, key: event.target.elements.token.value});
        state.token = "";
        event.target.reset();
      } else {
        if (engineUrl.origin !== location.origin) throw new Error("Use the address of this local preview.");
        state.token = event.target.elements.token.value;
      }
      state.demo = false;
      state.data = null;
      state.spec = null;
      await refresh();
      if (state.error) throw new Error(state.error);
    } else if (formId === "portal-password-form") {
      const setup = event.target.dataset.setup === "true";
      state.session = await sessionRequest(setup ? "password" : "login", setup ? "PUT" : "POST", {password: event.target.elements.password.value});
      event.target.reset();
      await refresh();
      if (state.error) throw new Error(state.error);
    } else if (formId === "portal-timeout-form") {
      await saveSessionTimeout(event.target);
    } else if (formId === "portal-unlock-form") {
      if (!state.advanced) throw new Error("Open Advanced to change the portal lock.");
      state.session = await sessionRequest("password", "PUT", {password: ""});
      await refresh();
      if (state.error) throw new Error(state.error);
    } else if (formId === "learn-form") {
      if (state.demo) throw new Error("Prepared intake needs a connected admin API. Demo mode does not grade knowledge.");
      const form = event.target;
      const body = {source_ref: form.elements.namedItem("source_ref").value,
        format: form.elements.namedItem("format").value, content: form.elements.namedItem("content").value};
      document.getElementById("learn-progress").textContent = "Validating and preparing your knowledge.";
      const result = await api("/learn", "POST", body);
      await gradePrepared(result.items);
      preparedResult(result.items, result.queued_ids.length);
      return;
    } else if (formId === "schedule-form") {
      const form = event.target;
      const current = state.form.schedule;
      const days = [];
      for (let index = 0; index < 7; index += 1) {
        if (form.elements.namedItem("day-" + index).checked) days.push(index);
      }
      if (!days.length) throw new Error("Choose at least one approved day.");
      const windows = structuredClone(current.windows || []);
      windows[0] = {
        days,
        start: form.elements.start.value,
        end: form.elements.end.value,
      };
      await mutate("PUT", "/schedule", {
        expected_revision: revision(current),
        enabled: form.elements.enabled.checked,
        timezone: form.elements.timezone.value,
        windows,
        min_idle_seconds: Number(form.elements.idle.value) * 60,
        max_active_requests: current.max_active_requests ?? 0,
      });
    } else if (formId === "policy-field-form") {
      const edited = parseForm(
        event.target,
        { properties: { value: state.form.definition } },
        {},
      );
      const value = edited.value;
      if (!state.policyDraft)
        state.policyDraft = structuredClone(state.data.policy);
      const parts = state.form.policyPath.split(".");
      if (parts.length === 2) state.policyDraft[parts[0]][parts[1]] = value;
      else state.policyDraft[parts[0]] = value;
      render();
    } else {
      const form = state.form;
      let body = parseForm(event.target, schema(form.schemaName), form.values);
      if (form.schemaName === "QueueEdit") {
        const identifier = decodeURIComponent(form.path.split("/").at(-1));
        const previous = findItem("queue", identifier);
        const edited = {
          expected_revision: body.expected_revision,
          hold: body.hold,
          note: body.note,
        };
        if (
          body.text !== previous.text ||
          body.prompt_evidence !== previous.prompt_evidence
        ) {
          edited.text = body.text;
          edited.prompt_evidence = body.prompt_evidence;
        }
        body = edited;
      }
      const representation = {};
      if (form.schemaName === "PocketImport") {
        // Preserve numeric representations in content-hashed pocket packages.
        const identifierJson = JSON.stringify(body.id);
        const packageJson = event.target.elements.namedItem("package").value;
        representation.requestText = `{"id":${identifierJson},"package":${packageJson}}`;
      }
      const result = await mutate(form.method, form.path, body, representation);
      if (form.schemaName === "DreamCheck") {
        showRaw(
          "Dream readiness",
          state.demo
            ? {
                ready: false,
                reason: "Demo planning only; no native runner invoked",
              }
            : result,
        );
        return;
      }
    }
    if (document.getElementById("drawer").open) document.getElementById("drawer").close();
    toast(state.demo ? "Demo change saved" : "Changes recorded");
  } catch (error) {
    if (formId === "learn-form") {
      const progress = document.getElementById("learn-progress");
      if (progress) progress.textContent = "Review paused. Queued facts remain available in the learning queue.";
    }
    showError(error);
  } finally {
    state.busy = false;
  }
});

document.addEventListener("change", async (event) => {
  const target = event.target;
  if (target.id === "learn-file") {
    const file = target.files[0];
    if (!file) return;
    try {
      if (file.size > 1000000) throw new Error("Prepared files support at most 1 MB. Split larger documents before submitting.");
      const bytes = await file.arrayBuffer();
      const content = new TextDecoder("utf-8", {fatal: true}).decode(bytes);
      const form = document.getElementById("learn-form");
      form.elements.namedItem("content").value = content;
      form.elements.namedItem("format").value = file.name.toLowerCase().endsWith(".json") ? "json" : "text";
      form.elements.namedItem("source_ref").value = "manual:file:" + file.name;
      document.getElementById("learn-progress").textContent = "Ready to review " + file.name + ".";
    } catch (error) {
      showError(error);
    }
    return;
  }
  if (target.id === "queue-filter") {
    state.filter = target.value;
    render();
  } else if (target.classList.contains("candidate-check")) {
    if (target.checked) state.selected.add(target.dataset.id);
    else state.selected.delete(target.dataset.id);
    render();
    document
      .querySelector(
        `.candidate-check[data-id="${CSS.escape(target.dataset.id)}"]`,
      )
      ?.focus();
  } else if (target.id === "import-file") {
    try {
      const sourceJson = await target.files[0].text();
      const value = JSON.parse(sourceJson);
      if (target.dataset.kind === "policy") {
        if (
          value.schema_version !== 1 ||
          !value.extract ||
          !value.grade ||
          !value.control
        )
          throw new Error(
            "Use a version 1 policy with extract, grade and control sections.",
          );
        state.policyDraft = value;
        render();
        document.getElementById("drawer").close();
        toast("Policy imported as a next-run draft");
      } else {
        openForm(
          "Import pocket package",
          "PocketImport",
          { id: value.id || "", package: value },
          "POST",
          "/pockets/import",
          { basic: ["id"], saveLabel: "Import pocket" },
        );
        document
          .getElementById("operation-form")
          .elements.namedItem("package").value = sourceJson;
      }
    } catch (error) {
      showError(error);
    }
  }
});

document.addEventListener("input", (event) => {
  if (event.target.id === "queue-search") {
    const position = event.target.selectionStart;
    state.search = event.target.value;
    render();
    const input = document.getElementById("queue-search");
    input.focus();
    input.setSelectionRange(position, position);
  } else if (event.target.id === "manual-search") {
    const term = event.target.value.toLowerCase();
    for (const section of document.querySelectorAll("[data-manual-section]")) {
      let matchesPage = section.textContent.toLowerCase().includes(term);
      for (const tasks of section.querySelectorAll("[data-manual-tasks]")) {
        tasks.hidden =
          Boolean(term) && !tasks.textContent.toLowerCase().includes(term);
      }
      for (const topic of section.querySelectorAll("[data-manual-topic]")) {
        const searchable = topic.textContent + " " + topic.dataset.manualFields;
        const matches = searchable.toLowerCase().includes(term);
        topic.hidden = Boolean(term) && !matches;
        topic.open = Boolean(term) && matches;
        if (matches) matchesPage = true;
      }
      section.hidden = !matchesPage;
    }
    for (const overview of document.querySelectorAll(
      "[data-manual-overview], [data-manual-navigation]",
    )) {
      overview.hidden = Boolean(term);
    }
  }
});

document.addEventListener("keydown", (event) => {
  const input = event.target.closest("input, textarea, select");
  if (
    !input &&
    !event.ctrlKey &&
    !event.metaKey &&
    !event.altKey &&
    !document.getElementById("drawer").open &&
    event.key.toLowerCase() === "a"
  ) {
    state.advanced = !state.advanced;
    render();
  }
  if (!input && event.key === "/" && state.page === "queue") {
    event.preventDefault();
    document.getElementById("queue-search")?.focus();
  }
  if (
    (event.ctrlKey || event.metaKey) &&
    event.key === "Enter" &&
    state.advanced &&
    state.page === "queue" &&
    state.selected.size &&
    !document.getElementById("drawer").open
  ) {
    event.preventDefault();
    handleAction("bulk-approve", "");
  }
});

function navigate() {
  const requested = location.hash.slice(1);
  state.page = pages.includes(requested) ? requested : "rules";
  render();
}

window.addEventListener("hashchange", navigate);
navigate();
refresh();
