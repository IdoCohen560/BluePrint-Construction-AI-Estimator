# Data directory

- `materials/` — CSV of material attributes and unit costs used for training and cost roll-up.
- `labels/` — Labeled requirement–material pairs (`match` 0/1) for supervised learning.
- `blueprints/` — Optional sample vector JSON or your own PDFs/DXF files. Large proprietary drawings may stay local; add paths here only if you intend to commit small synthetic samples.

The pipeline also generates **synthetic** floor plans in code when no file is available, so you can run notebooks without adding real blueprint files.
