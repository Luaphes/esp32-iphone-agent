// Run with: node --test tests/test_controls.cjs (no browser or device required).
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const staticDir = resolve(__dirname, "../hidagent/static");
const response = (data, ok = true) => ({ ok, json: async () => data });
const idle = () => ({ device: true, screen: true, calibrated: false, aspect: .6963,
  models: { planner: "test", actor: "test" }, agent: { running: false, steps: [] } });

function element(tag = "div") {
  return {
    tag, textContent: "", value: "", src: "", dataset: {}, style: {}, children: [], listeners: {},
    classList: { toggle() {}, add() {} },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    replaceChildren(...children) { this.children = children; },
    querySelectorAll(tag) { return this.children.filter((child) => child.tag === tag); },
    removeAttribute() {}, getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
  };
}

async function page(state = idle(), routes = {}) {
  const nodes = new Map(), requests = [];
  const get = (id) => {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  };
  get("live").parentElement = element();
  get("log").parentElement = element();
  get("setup").children = ["device", "screen", "calibrated"].map((key) => ({
    ...element("li"), dataset: { key },
  }));
  const context = vm.createContext({
    document: { getElementById: get, createElement: element, querySelectorAll: () => [],
      documentElement: {}, body: element(), addEventListener() {} },
    localStorage: { getItem() {}, setItem() {} }, matchMedia: () => ({ matches: false }),
    setTimeout() {}, URL: { revokeObjectURL() {} },
    fetch: async (path, options) => {
      if (options) {
        requests.push([path, JSON.parse(options.body)]);
        if (routes[path]) return routes[path]();
        return response({ ok: true, auto: true });
      }
      if (path === "/api/status") return response(structuredClone(state));
      return response({}, false);
    },
  });
  for (const file of ["i18n.js", "app.js"]) vm.runInContext(readFileSync(resolve(staticDir, file), "utf8"), context);
  await new Promise(setImmediate); // let the initial status poll render
  return { context, get, requests, state, run: (code) => vm.runInContext(code, context) };
}

test("orientation invalidation keeps Start usable and calibrates before submitting once", async () => {
  let finish;
  const p = await page(idle(), { "/api/calibrate/start": () => new Promise((resolve) => { finish = resolve; }) });
  assert.equal(p.get("run").disabled, false);
  assert.match(p.get("task-hint").textContent, /automatically/);
  const pending = p.context.start("  Read the screen  ");
  await new Promise(setImmediate);
  await p.context.poll(); // polling must not re-enable Start during calibration
  assert.equal(p.get("run").disabled, true);
  await p.context.start("duplicate");
  finish(response({ ok: true, auto: true }));
  await pending;
  assert.deepEqual(p.requests, [["/api/calibrate/start", {}], ["/api/agent", { text: "Read the screen" }]]);
  assert.equal(p.get("run").disabled, true);
});

test("manual fallback preserves the task and waits for a pointer click and another Start", async () => {
  const p = await page(idle(), { "/api/calibrate/start": () => response({ ok: true, auto: false }) });
  p.get("task").value = "Read the screen";
  await p.get("run").listeners.click();
  assert.equal(p.get("task").value, "Read the screen");
  assert.match(p.get("task-hint").textContent, /round pointer/);
  await p.context.start("Read the screen");
  assert.equal(p.requests.length, 1);
  await p.get("live").listeners.pointerdown({ clientX: 40, clientY: 40 });
  await p.get("live").listeners.pointerup({ clientX: 40, clientY: 40 });
  p.state.calibrated = true;
  await p.get("run").listeners.click();
  assert.deepEqual(p.requests.map(([path]) => path), ["/api/calibrate/start", "/api/calibrate/finish", "/api/agent"]);
});

test("an already calibrated screen sends directly", async () => {
  const p = await page({ ...idle(), calibrated: true });
  await p.context.start("Read the screen");
  assert.deepEqual(p.requests, [["/api/agent", { text: "Read the screen" }]]);
});

test("manual calibration blocks duplicate pointer confirmation and task submission until ready", async () => {
  let finish;
  const p = await page(idle(), {
    "/api/calibrate/start": () => response({ ok: true, auto: false }),
    "/api/calibrate/finish": () => new Promise((resolve) => { finish = resolve; }),
  });
  await p.context.start("Read the screen");
  const click = async () => {
    await p.get("live").listeners.pointerdown({ clientX: 40, clientY: 40 });
    return p.get("live").listeners.pointerup({ clientX: 40, clientY: 40 });
  };
  const pending = click();
  await new Promise(setImmediate);
  await p.context.poll();
  assert.equal(p.get("run").disabled, true);
  await click();
  await p.context.start("duplicate");
  assert.deepEqual(p.requests.map(([path]) => path), ["/api/calibrate/start", "/api/calibrate/finish"]);
  finish(response({ ok: true }));
  await pending;
  assert.equal(p.get("run").disabled, false);
});

for (const missing of ["device", "screen"]) {
  test(`${missing} disconnect disables controls and stale status is rechecked before submission`, async () => {
    const p = await page({ ...idle(), [missing]: false });
    assert.equal(p.get("run").disabled, true);
    assert.ok(p.get("task-hint").textContent);
    p.state[missing] = true;
    await p.context.poll();
    assert.equal(p.get("run").disabled, false);
    p.state[missing] = false;
    await p.context.start("Read the screen");
    assert.deepEqual(p.requests, []);
    assert.equal(p.get("run").disabled, true);
  });
}

test("running tasks stay disabled when language rebuilds example buttons", async () => {
  const state = idle();
  state.agent.running = true;
  const p = await page(state);
  p.run('lang = "zh"; applyLang();');
  assert.equal(p.get("run").disabled, true);
  assert.ok(p.get("examples").children.every((button) => button.disabled));
  assert.match(p.get("task-hint").textContent, /任务正在执行/);
  await p.context.start("another task");
  assert.deepEqual(p.requests, []);
});

test("emergency stop during automatic calibration never sends the pending task", async () => {
  let finish;
  const p = await page(idle(), { "/api/calibrate/start": () => new Promise((resolve) => { finish = resolve; }) });
  const pending = p.context.start("Read the screen");
  await new Promise(setImmediate);
  await p.context.stop();
  finish(response({ ok: true, auto: true }));
  await pending;
  assert.deepEqual(p.requests.map(([path]) => path), ["/api/calibrate/start", "/api/agent/stop"]);
  assert.equal(p.get("run").disabled, false);
});

test("failed calibration reports the error and leaves the task retryable", async () => {
  const p = await page(idle(), { "/api/calibrate/start": () => response({ detail: "phone usb asleep" }, false) });
  await p.context.start("Read the screen");
  assert.equal(p.get("msg").textContent, "phone usb asleep");
  assert.equal(p.requests.length, 1);
  assert.equal(p.get("run").disabled, false);
});

test("network failure while sending reports an error and releases the submit lock", async () => {
  const p = await page({ ...idle(), calibrated: true }, { "/api/agent": () => { throw new Error("offline"); } });
  await p.context.start("Read the screen");
  assert.match(p.get("msg").textContent, /Cannot reach/);
  assert.equal(p.get("run").disabled, false);
});

test("server polling failure disables Start until the server returns", async () => {
  const p = await page();
  const fetch = p.context.fetch;
  p.context.fetch = async () => { throw new Error("offline"); };
  await p.context.poll();
  assert.equal(p.get("run").disabled, true);
  assert.match(p.get("task-hint").textContent, /Cannot reach/);
  p.context.fetch = fetch;
  await p.context.poll();
  assert.equal(p.get("run").disabled, false);
});

test("empty input does not calibrate or submit", async () => {
  const p = await page();
  await p.context.start("   ");
  assert.deepEqual(p.requests, []);
  assert.match(p.get("msg").textContent, /Describe/);
});
