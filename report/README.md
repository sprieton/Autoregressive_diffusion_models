# P1 Report — build

Self-contained LaTeX source. Compile with:

```
pdflatex report.tex
pdflatex report.tex   # second pass resolves references/citations
```

Contents:
- `report.tex` — the manuscript (two-column IEEEtran).
- `IEEEtran.cls` — bundled document class (so no system TeX package is needed).
- `figs/` — the figure PDFs included by the manuscript.

Requires a standard TeX distribution (TeX Live / MiKTeX) with the usual
packages: `graphicx`, `booktabs`, `amsmath`, `amssymb`, `multirow`, `tikz`.
Uploading this folder to Overleaf works as-is.
