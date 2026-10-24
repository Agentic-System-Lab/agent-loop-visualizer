/* The 運作說明 page (guide.py) inlines this after the page body. The table of contents stays at
   the top of the window as one line; this keeps that line on the section you are reading and
   closes the open list after a jump, a click elsewhere or Escape. Without it the list still works.
   It also remembers the language picked on the 中文 | English switch, for index.html to go by. */

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
addEventListener("scroll", update, { passive: true });
addEventListener("resize", update);
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
