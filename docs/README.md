# Docs

Use [docs/prd.md](prd.md) as the canonical project definition.

Supporting docs:

- [docs/week4-demo.md](week4-demo.md) for the current static-image demo flow
- [docs/week4-ml-notes.md](week4-ml-notes.md) for Stage 2 ML prep and the single-model comparison baseline
- [docs/final-runbook.md](final-runbook.md) for the final trained-detector workflow and submission artifact generation
- [docs/prd-diagrams.md](prd-diagrams.md) for Mermaid architecture diagrams derived from the PRD
- [edge/README.md](../edge/README.md) for the edge runtime contract
- [backend/README.md](../backend/README.md) for backend payload persistence

Everything in this folder should describe the same canonical v6 architecture:

`parking-space quadrilateral pooling -> per-space warp -> YOLOv8-cls -> temporal smoothing -> JSON -> FastAPI`
