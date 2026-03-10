module.exports = {
  stylesheet: [
    "https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.min.css"
  ],
  script: [
    {
      // Configure MathJax to recognize inline $...$ and \( ... \)
      content: `
        window.MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$','$$'], ['\\\\[','\\\\]']]
          },
          svg: { fontCache: 'global' }
        };
      `
    },
    {
      url: "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
    }
  ]
};