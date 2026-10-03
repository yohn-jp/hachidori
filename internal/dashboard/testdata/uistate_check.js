// Deterministic check of uistate.js against a minimal DOM: the live refresh
// replaces a region's markup with what the server rendered, and the operator's
// presentation choices must survive every replacement. Run by
// TestUIStateModule (node); exits non-zero on the first failed expectation.
"use strict";
const assert = require("assert");
const ui = require("../uistate.js");

// ---- a minimal DOM: a flat list of elements that serialize to and parse from markup
class El {
  constructor(tag, attrs) { this.tagName = tag.toUpperCase(); this.attrs = Object.assign({}, attrs); this.scrollTop = 0; this.scrollHeight = 1000; this.clientHeight = 200; this.parentNode = null; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  hasAttribute(k) { return k in this.attrs; }
  setAttribute(k, v) { this.attrs[k] = v; }
  removeAttribute(k) { delete this.attrs[k]; }
  get open() { return this.hasAttribute("open"); }
  get checked() { return this.hasAttribute("checked"); }
  set checked(v) { v ? this.setAttribute("checked", "") : this.removeAttribute("checked"); }
  serialize() { return "<" + this.tagName.toLowerCase() + Object.keys(this.attrs).sort().map(k => this.attrs[k] === "" ? " " + k : " " + k + '="' + this.attrs[k] + '"').join("") + ">"; }
}
function matches(el, sel) {
  const m = /^([a-z]*)\[([\w-]+)(?:="([^"]*)")?\]$/.exec(sel.trim());
  if (!m) throw new Error("unsupported selector " + sel);
  return (!m[1] || el.tagName.toLowerCase() === m[1]) && el.hasAttribute(m[2]) && (m[3] === undefined || el.getAttribute(m[2]) === m[3]);
}
class Region {
  constructor(els) { this.els = els; }
  querySelectorAll(sel) { return this.els.filter(e => sel.split(",").some(s => matches(e, s))); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  get innerHTML() { return this.els.map(e => e.serialize()).join(""); }
  set innerHTML(html) {
    this.els = (html.match(/<[^>]+>/g) || []).map(t => {
      const tag = /^<(\w+)/.exec(t)[1], attrs = {};
      t.replace(/\s([\w-]+)(?:="([^"]*)")?/g, (_, k, v) => { attrs[k] = v === undefined ? "" : v; });
      return new El(tag, attrs);
    });
  }
}
// The server renders a region from its own state; the operator's choices are not in it.
const details = (id, open) => new El("details", Object.assign({ id }, open ? { open: "" } : {}));
const kept = (key, open) => new El("details", Object.assign({ "data-keep": key }, open ? { open: "" } : {}));
const follow = (checked) => new El("input", Object.assign({ type: "checkbox", "data-keep": "console-follow" }, checked ? { checked: "" } : {}));
const tail = () => new El("pre", { "data-tail": "console", "data-follow": "console-follow" });
const find = (r, sel) => r.querySelector(sel);
// the operator toggles a disclosure as the browser does: the element changes, then the click handler records it
const toggle = (el) => { el.hasAttribute("open") ? el.removeAttribute("open") : el.setAttribute("open", ""); ui.remember(el); };
function refresh(region, server) { const src = new Region(server()); return ui.swap(region, src); }

let n = 0, chain = Promise.resolve();
// Checks run one after another: they share the module's operator-local state.
function test(name, fn) { chain = chain.then(() => { ui.reset(); return fn(); }).then(() => { n++; console.log("ok - " + name); }); }

test("a disclosure the operator opened stays open through repeated refreshes", () => {
  const server = () => [details("runtime-evidence", false), details("forge-advanced", false)];
  const region = new Region(server());
  toggle(find(region, "details[id=\"runtime-evidence\"]"));
  for (let i = 0; i < 5; i++) {
    refresh(region, server);
    assert.ok(find(region, 'details[id="runtime-evidence"]').open, "open after refresh " + (i + 1));
    assert.ok(!find(region, 'details[id="forge-advanced"]').open, "an untouched disclosure stays as rendered");
  }
});

test("a disclosure the operator closed stays closed although the server renders it open", () => {
  const server = () => [kept("worker-failure", true)];
  const region = new Region(server());
  toggle(find(region, 'details[data-keep="worker-failure"]'));
  assert.ok(!find(region, 'details[data-keep="worker-failure"]').open);
  for (let i = 0; i < 5; i++) {
    refresh(region, server);
    assert.ok(!find(region, 'details[data-keep="worker-failure"]').open, "closed after refresh " + (i + 1));
  }
});

test("a choice survives until the operator reverses it, then the new choice is kept", () => {
  const server = () => [details("d", false)];
  const region = new Region(server());
  toggle(find(region, "details[id]"));
  refresh(region, server); refresh(region, server);
  assert.ok(find(region, "details[id]").open);
  toggle(find(region, "details[id]"));
  refresh(region, server); refresh(region, server);
  assert.ok(!find(region, "details[id]").open);
});

test("an element the operator never touched follows the server", () => {
  let open = false;
  const server = () => [details("d", open)];
  const region = new Region(server());
  open = true; refresh(region, server);
  assert.ok(find(region, "details[id]").open, "the server opened it");
  open = false; refresh(region, server);
  assert.ok(!find(region, "details[id]").open, "the server closed it");
});

test("server content still refreshes; only presentation is local", () => {
  let rev = 1;
  const server = () => [details("d", false), new El("p", { "data-rev": String(rev) })];
  const region = new Region(server());
  toggle(find(region, "details[id]"));
  rev = 2;
  assert.ok(refresh(region, server), "a changed region is replaced");
  assert.strictEqual(find(region, "p[data-rev]").getAttribute("data-rev"), "2");
  assert.ok(find(region, "details[id]").open);
});

test("an unchanged region is left alone", () => {
  const server = () => [details("d", false)];
  const region = new Region(server());
  let assigned = 0;
  const desc = Object.getOwnPropertyDescriptor(Region.prototype, "innerHTML");
  Object.defineProperty(region, "innerHTML", { get() { return desc.get.call(this); }, set(v) { assigned++; desc.set.call(this, v); } });
  assert.strictEqual(refresh(region, server), false);
  toggle(find(region, "details[id]"));
  assert.strictEqual(refresh(region, server), false, "the operator's own choice is not a change the server made");
  assert.strictEqual(assigned, 0);
});

test("the console's open state and follow preference are kept", () => {
  const server = () => [kept("op-console", false), follow(true), tail()];
  const region = new Region(server());
  toggle(find(region, 'details[data-keep="op-console"]'));
  const box = find(region, 'input[data-keep="console-follow"]');
  box.checked = false; ui.remember(box);
  for (let i = 0; i < 3; i++) {
    refresh(region, server);
    assert.ok(find(region, 'details[data-keep="op-console"]').open, "console open after refresh");
    assert.ok(!find(region, 'input[data-keep="console-follow"]').checked, "follow off after refresh");
  }
  const again = find(region, 'input[data-keep="console-follow"]');
  again.checked = true; ui.remember(again);
  refresh(region, server);
  assert.ok(find(region, 'input[data-keep="console-follow"]').checked);
});

test("a following console is kept at its end and a console left behind keeps its scroll", () => {
  const server = () => [follow(true), tail()];
  const region = new Region(server());
  find(region, "pre[data-tail]").scrollTop = 120;
  refresh(region, server);
  region.els.forEach(e => { if (e.tagName === "PRE") e.scrollHeight = 1000; });
  // follow is on and the markup is unchanged, so nothing moved; force a change
  const changing = () => [follow(true), tail(), new El("i", { n: String(Math.random()) })];
  refresh(region, changing);
  assert.strictEqual(find(region, "pre[data-tail]").scrollTop, find(region, "pre[data-tail]").scrollHeight, "following: scrolled to the end");

  const box = find(region, 'input[data-keep="console-follow"]');
  box.checked = false; ui.remember(box);
  find(region, "pre[data-tail]").scrollTop = 120;
  refresh(region, () => [follow(true), tail(), new El("i", { n: String(Math.random()) })]);
  assert.strictEqual(find(region, "pre[data-tail]").scrollTop, 120, "not following: scroll position restored");
});

test("an element with no stable identity is not preserved", () => {
  const anon = (open) => new El("details", open ? { open: "" } : {});
  const server = () => [anon(false)];
  const region = new Region(server());
  region.els[0].setAttribute("open", ""); ui.remember(region.els[0]);
  refresh(region, server);
  assert.ok(!region.els[0].open);
});

test("the operator's choice is recorded from the click on a summary, not from a rendered default", () => {
  const listeners = {};
  const doc = { addEventListener: (type, fn) => { listeners[type] = fn; }, querySelector: () => null };
  ui.install(doc);
  const d = details("evidence", false);
  const summary = { parentNode: d, closest: (s) => (s === "summary" ? summary : null) };
  // rendering a default (the server opens it) records nothing
  d.setAttribute("open", "");
  const server = () => [details("evidence", false)];
  const region = new Region([d]);
  refresh(region, server);
  assert.ok(!region.els[0].open, "the rendered default is not an operator choice");
  // the operator clicks the summary: the click handler schedules a read after the toggle
  const el = region.els[0];
  summary.parentNode = el;
  listeners.click({ target: summary });
  el.setAttribute("open", "");
  return new Promise(r => setTimeout(r, 5)).then(() => {
    refresh(region, server);
    assert.ok(region.els[0].open, "recorded from the click");
  });
});

chain.then(() => console.log("all " + n + " checks passed"), (e) => { console.error(e); process.exit(1); });
