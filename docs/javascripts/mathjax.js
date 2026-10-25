// MathJax setup for the maths that pymdownx.arithmatex wraps in \(...\) and \[...\].
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // only typeset what arithmatex marked, never stray dollar signs in code or text
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Instant navigation swaps the page without a reload, so typeset again on every page change.
document$.subscribe(() => {
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
