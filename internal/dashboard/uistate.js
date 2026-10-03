// Operator-local presentation state for the server-rendered workstation.
//
// The dashboard refreshes a live region by replacing its markup with what the
// server just rendered. The server is authoritative for content; only how the
// operator chose to look at it is local: whether a disclosure is open, whether
// a console follows its output, and where a console is scrolled. This module is
// the one place that keeps those choices across a replacement:
//
//   - a choice is recorded only when the operator makes it (a click or key on a
//     disclosure's summary, a change of a kept checkbox), never when the server
//     renders a default, so an element the operator never touched keeps
//     following the server;
//   - an element is identified by data-keep, else its id. An element with
//     neither is not preserved;
//   - swap() applies the recorded choices to the incoming markup before it is
//     inserted, so nothing flickers closed and open again.
//
// Nothing is persisted: a new page load shows what the server rendered.
(function (root) {
  "use strict";

  var local = Object.create(null);

  function identity(el) {
    return el.getAttribute("data-keep") || el.getAttribute("id") || "";
  }

  function kind(el) {
    var tag = (el.tagName || "").toLowerCase();
    if (tag === "details") return "open";
    if (tag === "input" && el.getAttribute("type") === "checkbox" && el.getAttribute("data-keep")) return "checked";
    return "";
  }

  // remember records the operator's current choice for el.
  function remember(el) {
    var id = identity(el), k = kind(el);
    if (!id || !k) return;
    local[id] = !!el[k];
  }

  function set(el, k, on) {
    if (on) el.setAttribute(k, ""); else el.removeAttribute(k);
  }

  // restore gives every kept element under tree the operator's last choice.
  function restore(tree) {
    var els = tree.querySelectorAll("details[id], details[data-keep], input[data-keep]");
    for (var i = 0; i < els.length; i++) {
      var id = identity(els[i]), k = kind(els[i]);
      if (id && k && id in local) set(els[i], k, local[id]);
    }
  }

  // follows reports whether the console named by its follow key follows its
  // output: the operator's choice if there is one, else what the server rendered.
  function follows(tail, tree) {
    var key = tail.getAttribute("data-follow");
    if (!key) return true;
    if (key in local) return local[key];
    var box = tree.querySelector('input[data-keep="' + key + '"]');
    return box ? box.hasAttribute("checked") : true;
  }

  // swap replaces dst's content with src's, keeping the operator's presentation
  // state. It does nothing when the server's content is already what is shown.
  function swap(dst, src) {
    restore(src);
    if (dst.innerHTML === src.innerHTML) return false;
    var scroll = Object.create(null), tails = dst.querySelectorAll("[data-tail]");
    for (var i = 0; i < tails.length; i++) scroll[tails[i].getAttribute("data-tail")] = tails[i].scrollTop;
    dst.innerHTML = src.innerHTML;
    tails = dst.querySelectorAll("[data-tail]");
    for (var j = 0; j < tails.length; j++) {
      var name = tails[j].getAttribute("data-tail");
      if (follows(tails[j], dst)) tails[j].scrollTop = tails[j].scrollHeight;
      else if (name in scroll) tails[j].scrollTop = scroll[name];
    }
    return true;
  }

  // install listens for the operator's own choices on a document.
  function install(doc) {
    // A disclosure toggles after the click that activates its summary (a mouse
    // click, Enter or Space), so the new state is read once that has happened.
    doc.addEventListener("click", function (e) {
      var s = e.target && e.target.closest && e.target.closest("summary");
      var d = s && s.parentNode;
      if (d && kind(d) === "open") setTimeout(function () { remember(d); }, 0);
    });
    doc.addEventListener("change", function (e) {
      if (e.target && kind(e.target) === "checked") remember(e.target);
    });
    // A console the operator scrolls away from its end stops following; one
    // scrolled back to its end follows again.
    doc.addEventListener("scroll", function (e) {
      var t = e.target;
      if (!t || !t.getAttribute || !t.hasAttribute("data-tail")) return;
      var key = t.getAttribute("data-follow"), box = key && doc.querySelector('input[data-keep="' + key + '"]');
      if (!box) return;
      var atEnd = t.scrollHeight - t.scrollTop - t.clientHeight < 4;
      if (box.checked !== atEnd) { box.checked = atEnd; remember(box); }
    }, true);
  }

  var api = {swap: swap, restore: restore, remember: remember, install: install,
    reset: function () { for (var k in local) delete local[k]; }};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.hachidoriUI = api;
})(typeof window !== "undefined" ? window : this);
