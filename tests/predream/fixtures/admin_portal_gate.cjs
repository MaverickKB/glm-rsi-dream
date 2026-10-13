// Exercise the shipped renderers and action guards without a GPU or browser.
const fs = require("node:fs");
const vm = require("node:vm");

async function main() {
  const input = JSON.parse(fs.readFileSync(0, "utf8"));
  const source = fs.readFileSync(input.script, "utf8");
  const boot = "\nnavigate();\nrefresh();\n";
  if (!source.endsWith(boot)) throw new Error("Portal bootstrap changed");
  const requests = [];
  const drawers = [];
  const context = vm.createContext({
    URLSearchParams, structuredClone,
    location: { search: "", hash: "" },
    document: {
      addEventListener() {},
      createElement() {
        return {
          textContent: "",
          get innerHTML() {
            return this.textContent.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
          },
        };
      },
    },
    window: { addEventListener() {} },
    async fetch(path) {
      requests.push(path);
      let value;
      if (path === "/api/v1/system") value = input.system;
      else if (path === "/api/v1/control") value = input.data.control;
      else if (path === "/api/v1/policy") value = { policy: input.data.policy };
      else if (path === "/api/v1/schedule") value = input.data.schedule;
      else if (path === "/openapi.json") value = input.spec;
      else value = { items: input.data[path.split("/").at(-1)] || [] };
      return { ok: true, status: 200, json: async () => value, text: async () => JSON.stringify(value) };
    },
  });
  vm.runInContext(source.slice(0, -boot.length), context);
  context.input = input;
  context.drawers = drawers;
  vm.runInContext(`
    state.data = structuredClone(input.data);
    state.data.system = input.system;
    state.spec = input.spec;
    state.demo = input.demo;
    state.token = "fictional-test-credential";
    render = function () {};
    openDrawer = function (title, html, subtitle) { drawers.push({title, html, subtitle}); };
  `, context);
  const pages = [];
  for (const advanced of [false, true]) {
    context.advanced = advanced;
    pages.push(vm.runInContext(`
      state.advanced = advanced;
      state.page = "dreams";
      ({ advanced, html: renderDreams() });
    `, context));
  }
  const enabled = vm.runInContext("dreamsEnabled()", context);
  const errors = [];
  if (!enabled) {
    for (const path of ["/dreams", "/dreams/fixture/check", "/schedule", "/api/v1/dreams/fixture"]) {
      context.path = path;
      try {
        await vm.runInContext("mutate('POST', path, {})", context);
        throw new Error("Closed dream mutation was allowed");
      } catch (error) {
        errors.push(error.message);
      }
    }
    for (const action of ["dream-create", "dream-check", "dream-cancel", "dream-export", "schedule-edit", "schedule-raw"]) {
      context.action = action;
      await vm.runInContext("handleAction(action, 'fixture')", context);
    }
    vm.runInContext("openForm('Stale plan', 'DreamCreate', {}, 'POST', '/dreams')", context);
  }
  vm.runInContext("showApiExplorer()", context);
  const manual = vm.runInContext("manualPageHtml(input.manual, input.manual.pages.find(page => page.id === 'dreams'), true)", context);
  const ordinary = vm.runInContext(`
    state.page = "rules";
    const rulesHtml = renderRules();
    state.page = "queue";
    const queueHtml = renderQueue();
    state.page = "pockets";
    ({rules: rulesHtml, queue: queueHtml, pockets: renderPockets()});
  `, context);
  context.input.demo = false;
  vm.runInContext("state.demo = false", context);
  await vm.runInContext("refresh()", context);
  process.stdout.write(JSON.stringify({ enabled, pages, errors, drawers, manual, ordinary, requests,
    refreshError: vm.runInContext("state.error", context) }));
}

main().catch(error => {
  process.stderr.write(error.stack);
  process.exitCode = 1;
});
