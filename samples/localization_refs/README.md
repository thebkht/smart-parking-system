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
5. Use `query_set.night_overhead.json` with `ml/evaluate_localization.py` when you want to score the checked-in multi-query evaluation set.

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

Current checked-in evaluation status:

- `query_set.night_overhead.json` contains `21` labeled same-lot night-overhead queries, including three reference-frame sanity checks.
- `localize_eval.night_overhead.md` is the tracked snapshot for the current `21/21` top-1 correct, `21/21` top-3 correct, `536.18 ms` average-runtime run.

Example command:

```bash
make localize-car LOCALIZE_ARGS="--query samples/localization_refs/query_candidates/photo_2026-04-23_21.29.43.jpeg --references samples/localization_refs --output logs/localize_result.json"
python ml/evaluate_localization.py --queries samples/localization_refs/query_set.night_overhead.json --references samples/localization_refs/labeled --output-json logs/localize_eval.json --output-csv logs/localize_eval.csv
```

When `--references` points at the workspace root, localization only uses labeled spot folders and ignores helper directories such as `query_candidates/` and `unlabeled_pool/`.
