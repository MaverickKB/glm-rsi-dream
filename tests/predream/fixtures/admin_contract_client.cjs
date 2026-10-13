// Exercise the shipped portal client and form parser through the real local proxy.
// DOM boot is suppressed here. Visible DOM interaction has separate browser proof.
const fs = require("node:fs");
const vm = require("node:vm");

async function main() {
  const input = JSON.parse(fs.readFileSync(0, "utf8"));
  const source = fs.readFileSync(input.script, "utf8");
  const boot = "\nnavigate();\nrefresh();\n";
  if (!source.endsWith(boot)) throw new Error("Portal bootstrap changed");
  const requests = [];
  const context = vm.createContext({
    URLSearchParams, structuredClone,
    location: { search: "", hash: "" },
    document: { addEventListener() {} },
    window: { addEventListener() {} },
    async fetch(path, options) {
      options.headers.Origin = input.origin;
      const receipt = { method: options.method, path, body: options.body || null };
      requests.push(receipt);
      const response = await fetch(new URL(path, input.origin), options);
      receipt.status = response.status;
      receipt.response = await response.clone().json();
      return response;
    },
  });
  vm.runInContext(source.slice(0, -boot.length), context);
  context.input = input;
  vm.runInContext("state.token = input.token; state.spec = input.spec;", context);
  let body = input.body ?? null;
  if (input.schemaName) {
    const form = {
      querySelector() { return null; },
      elements: {
        namedItem(name) { return input.fields[name] || null; },
      },
    };
    context.form = form;
    body = vm.runInContext(
      "parseForm(form, state.spec.components.schemas[input.schemaName], input.previous || {})",
      context,
    );
  }
  context.body = body;
  try {
    const result = await vm.runInContext(
      "api(input.path, input.method || 'GET', body, input.representation || {})",
      context,
    );
    process.stdout.write(JSON.stringify({ ok: true, result, requests, body }));
  } catch (error) {
    process.stdout.write(JSON.stringify({ ok: false, status: error.status, error: error.message, requests, body }));
  }
}

main().catch((error) => {
  process.stderr.write(error.stack);
  process.exitCode = 1;
});
