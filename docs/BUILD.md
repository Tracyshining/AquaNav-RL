# Building the methodology document

The source uses `ctexart` so that Chinese text and mathematical notation can be
typeset together. Install a TeX distribution that includes CTeX, then run:

```bash
cd docs
xelatex methodology.tex
xelatex methodology.tex
```

The second pass resolves the table of contents. The source file remains useful
without compilation and can be edited in Overleaf using the XeLaTeX compiler.

