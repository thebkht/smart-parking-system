# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-06-15

First public open-source release.

### Added

- Two-stage edge occupancy pipeline: parking-space quadrilaterals → perspective
  warp (`order_corners()` + `warpPerspective`) → `YOLOv8-cls` → temporal
  smoothing → compact JSON to the backend.
- FastAPI backend with SQLite persistence; map/status/park APIs and optional
  bearer-token auth (`AUTH_ENABLED`).
- Web app (Vite + React + Leaflet): owner setup and live occupancy map.
- Mobile app (React Native / Expo): live occupancy map and photo-based
  Find My Car (SIFT + FLANN + RANSAC).
- Reproducible ML pipeline (`make` targets) for dataset extraction, training,
  evaluation, ONNX / Core ML INT8 export, and benchmarking.
- Project documentation: `docs/architecture.md`, diagrams, runbook, and
  per-component READMEs.
- Community health and CI: GitHub Actions (lint + test), issue/PR templates,
  `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`.
- Model weights published as GitHub Release assets via `make fetch-weights`.

[Unreleased]: https://github.com/thebkht/smart-parking-system/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/thebkht/smart-parking-system/releases/tag/v1.0.0
