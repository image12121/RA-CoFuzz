# Analysis scripts

- `aggregate_results.py` validates a private raw result tree and emits content-free aggregate JSON plus Markdown tables.
- `make_figures.py` regenerates 18 standalone figures from the public aggregate.

Neither script requires network access. Figure generation uses Matplotlib, Seaborn, NumPy, and a serif font fallback. PDF and SVG are the preferred manuscript formats; PNG is exported at 300 dpi for previews.
