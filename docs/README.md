# Docs

Start with [architecture.md](architecture.md) — the canonical architecture
reference for the project.

Supporting docs:

- [architecture.md](architecture.md) — canonical pipeline, components, and payload contract
- [prd-diagrams.md](prd-diagrams.md) — Mermaid architecture and approach-evolution diagrams
- [environment-setup.md](environment-setup.md) — shared `.venv` setup
- [runbook.md](runbook.md) — trained-detector workflow and end-to-end run
- [edge_benchmarks.md](edge_benchmarks.md) — edge inference benchmark results
- [../edge/README.md](../edge/README.md) — edge runtime contract
- [../backend/README.md](../backend/README.md) — backend API and payload persistence
- [../ml/README.md](../ml/README.md) — ML pipeline (extract → train → evaluate → export)

Everything in this folder describes the same canonical architecture:

`parking-space quadrilateral pooling -> per-space warp -> YOLOv8-cls -> temporal smoothing -> JSON -> FastAPI`
