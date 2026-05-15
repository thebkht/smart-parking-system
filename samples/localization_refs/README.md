# Localization Reference Starter

This folder is a starter workspace for `make localize-car`.

Current layout:

- `unlabeled_pool/day_aerial/`
- `unlabeled_pool/night_overhead/`
- `query_candidates/`
- `labeled/`
- `manifest.template.json`
- `query_set.sample.json`

How to use it:

1. Review the images in `unlabeled_pool/`.
2. Create one folder per known spot, for example:
   `spot_1/`, `spot_2/`, `spot_3/`.
3. Copy the matching images into the correct spot folders.
4. Either point `--references` at the directory of `spot_*` folders, or fill in `manifest.template.json`.
5. Use `query_set.sample.json` with `ml/evaluate_localization.py` when you want to score multiple labeled queries.

Recommended interpretation:

- `query_candidates/` contains images that are acceptable as temporary query inputs for local testing.
- `unlabeled_pool/` contains candidate reference images, but they are not yet labeled by `spot_id`.

Example final layout:

```text
samples/localization_refs/
  spot_1/
    ref_1.jpg
    ref_2.jpg
  spot_2/
    ref_1.jpg
```

Example command:

```bash
make localize-car LOCALIZE_ARGS="--query samples/localization_refs/query_candidates/photo_2026-04-23_21.29.43.jpeg --references samples/localization_refs --output logs/localize_result.json"
```

When `--references` points at the workspace root, localization only uses labeled spot folders and ignores helper directories such as `query_candidates/` and `unlabeled_pool/`.
