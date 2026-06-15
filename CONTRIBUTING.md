# Contributing

Thanks for your interest in improving Smart Parking System! Issues and pull
requests are welcome.

## Project direction

[`docs/architecture.md`](docs/architecture.md) is the canonical reference. The
current direction is quadrilateral-pooling (not fixed ROIs): parking-space
quadrilaterals → perspective warp → `YOLOv8-cls` → temporal smoothing → JSON →
FastAPI. Keep changes aligned with it, and call out any mismatch rather than
silently changing scope.

## Development setup

The repo uses a single shared virtual environment at the root.

```bash
make install-dev          # create .venv and install runtime + dev deps
source .venv/bin/activate
python -c "import cv2, ultralytics, yaml; print('env ok')"
```

Model weights are not committed — fetch them when you need to run inference:

```bash
make fetch-weights        # downloads acpds_cls/weights/best.pt from a Release
```

Frontend (web + mobile):

```bash
cd frontend && npm install
```

## Running the apps

```bash
make backend                                   # FastAPI on :8000 (docs at /docs)
make edge EDGE_ARGS="--image samples/demo.jpg" # edge inference (add --post to publish)
cd frontend && npm run dev                     # web app
cd frontend && npx expo start                  # mobile app (Expo SDK 54)
```

A good first run order: backend → edge (image mode) → web → mobile.

## Tests, lint, and formatting

Everything CI runs, you can run locally:

```bash
make test                  # backend + edge + ML (pytest)
cd frontend && npm test    # web + mobile contract tests (Vitest)
make smoke-test            # end-to-end path, in-process, in-memory DB

make lint                  # ruff check + black --check
cd frontend && npm run lint
```

Python is formatted with **black** and linted with **ruff** (config in
`pyproject.toml`). Optionally enable the git hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Pull request guidelines

1. Branch off `main`.
2. Keep PRs focused; include a clear description and link related issues.
3. Make sure `make test`, `cd frontend && npm test`, and both linters pass.
4. Do **not** commit secrets, `.env` files, datasets, model weights, the runtime
   database, or machine-specific absolute paths.
5. Use clear, conventional commit messages (e.g. `feat(edge): ...`,
   `fix(backend): ...`, `docs: ...`).

## Reporting bugs and requesting features

Use the GitHub issue templates. For security issues, follow
[`SECURITY.md`](SECURITY.md) instead of opening a public issue.

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE.md).
