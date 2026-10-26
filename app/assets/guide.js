/* The 運作說明 page (guide.py) inlines this after the page body. The table of contents stays at
   the top of the window as one line; this keeps that line on the section you are reading and
   closes the open list after a jump, a click elsewhere or Escape. Without it the list still works.
   It has every set of backend tabs show the backend picked last, every set of model tabs the
   model, and every set of harness tabs the harness, and remembers those picks and the language picked on the 中文 | English switch, for the
   next visit and index.html to go by. A row of tabs wider than the screen scrolls sideways; this
   fades the side with more tabs and keeps the picked one in view. Without it each set of tabs still
   switches on its own, and its row still scrolls. */

const toc = document.querySelector(".g-toc details");
const now = toc.querySelector(".g-toc-now");
const links = [...toc.querySelectorAll("ol a")];
const sections = links.map((a) => document.querySelector(a.hash));
const idle = now.innerHTML;
let shown = null;

function update() {
  // The section you are reading is the last one whose top has passed the upper third of the
  // window; at the bottom of the page it is the last one, however short.
  const bottom = innerHeight + scrollY >= document.documentElement.scrollHeight - 2;
  let i = bottom ? sections.length - 1 : -1;
  if (!bottom) sections.forEach((s, j) => { if (s.getBoundingClientRect().top <= innerHeight / 3) i = j; });
  if (i === shown) return;
  shown = i;
  links.forEach((a, j) => (j === i ? a.setAttribute("aria-current", "location") : a.removeAttribute("aria-current")));
  now.style.cssText = i < 0 ? "" : links[i].parentElement.style.cssText;
  now.innerHTML = i < 0 ? idle : links[i].querySelector(".g-num").outerHTML + links[i].querySelector(".g-num + span").outerHTML;
}

toc.addEventListener("click", (e) => { if (e.target.closest("a")) toc.open = false; });
document.addEventListener("click", (e) => { if (toc.open && !toc.contains(e.target)) toc.open = false; });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && toc.open) {
    toc.open = false;
    toc.querySelector("summary").focus();
  }
});
// Tabs, one per backend, per model or per harness: picking one in a set picks the same in every other set of that kind that
// has it. A key with a dash is a narrower one of the key before it ("vllm-docker" of "vllm", any vLLM server), and the
// two stand for each other: a set already on a narrower one stays on it when the broader one is picked. A set without
// the pick stays as it is.
const tabSets = [...document.querySelectorAll(".g-tabs")];
const related = (a, b) => a === b || a.startsWith(b + "-") || b.startsWith(a + "-");

// guide.css fades a row's side that has more tabs past the edge (data-more); a set scrolls to the tab picked, clear of
// that fade, sideways only, so the page itself stays where it is.
function edges(bar) {
  const left = bar.scrollLeft > 1, right = bar.scrollLeft + bar.clientWidth < bar.scrollWidth - 1;
  if (left || right) bar.dataset.more = left && right ? "both" : left ? "left" : "right";
  else delete bar.dataset.more;
}

function reveal(set) {
  const bar = set.querySelector(".g-tab-bar");
  const tab = bar.querySelector("input:checked")?.parentElement;
  if (tab) {
    const b = bar.getBoundingClientRect(), t = tab.getBoundingClientRect();
    const fade = parseFloat(getComputedStyle(bar).getPropertyValue("--fade")) || 0;
    if (t.left - fade < b.left) bar.scrollLeft -= b.left - t.left + fade;
    else if (t.right + fade > b.right) bar.scrollLeft += t.right + fade - b.right;
  }
  edges(bar);
}

function showTab(kind, key) {
  for (const set of tabSets) {
    if (set.dataset.kind !== kind) continue;
    const tabs = [...set.querySelectorAll(".g-tab-bar input")];
    const current = tabs.find((t) => t.checked);
    const next = tabs.find((t) => t.value === key)
      || (current && related(current.value, key) ? current : tabs.find((t) => related(t.value, key)));
    if (next) {
      next.checked = true;
      reveal(set);
    }
  }
}

const picked = {};  // kind -> the key picked last, kept as "tab-<kind>"
for (const kind of new Set(tabSets.map((set) => set.dataset.kind))) {
  try { picked[kind] = localStorage.getItem("tab-" + kind); } catch (err) {}
  if (picked[kind]) showTab(kind, picked[kind]);
}
tabSets.forEach(reveal);
document.fonts?.ready.then(() => tabSets.forEach(reveal));  // the page's font changes how wide the tabs are
for (const bar of document.querySelectorAll(".g-tab-bar")) bar.addEventListener("scroll", () => edges(bar), { passive: true });
if (Object.values(picked).some(Boolean)) {
  document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();  // the tabs above it may have changed height
}
document.addEventListener("change", (e) => {
  const tab = e.target.closest(".g-tab-bar input");
  if (!tab) return;
  // Sets above this one change height; keep it where it was on the screen. The browser's own scroll anchoring is
  // off meanwhile, so the two do not both move the page.
  const set = tab.closest(".g-tabs");
  const kind = set.dataset.kind;
  const top = set.getBoundingClientRect().top;
  document.documentElement.style.overflowAnchor = "none";
  showTab(kind, tab.value);
  scrollBy(0, set.getBoundingClientRect().top - top);
  requestAnimationFrame(() => { document.documentElement.style.overflowAnchor = ""; });
  const was = picked[kind];
  if (!(was && was.startsWith(tab.value + "-"))) picked[kind] = tab.value;  // "vllm" keeps the kind of server picked before
  try { localStorage.setItem("tab-" + kind, picked[kind]); } catch (err) {}
});

addEventListener("scroll", update, { passive: true });
addEventListener("resize", () => {
  update();
  // Only the fades: on a phone, scrolling the page resizes the window as the address bar comes and goes, and a row
  // the reader scrolled stays where they left it.
  document.querySelectorAll(".g-tab-bar").forEach(edges);
});
update();

// The two pages have the same sections, so switching keeps the reader on the one they were reading.
for (const a of document.querySelectorAll(".g-lang a")) {
  a.addEventListener("click", (e) => {
    try { localStorage.setItem("lang", a.dataset.lang); } catch (err) {}
    if (location.hash) {
      e.preventDefault();
      location.href = a.href + location.hash;
    }
  });
}
