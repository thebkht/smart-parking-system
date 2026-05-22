# Night Overhead Localization Evaluation

Source query set: [query_set.night_overhead.json](samples/localization_refs/query_set.night_overhead.json)

Summary:

- Query count: `21`
- Top-1 correct: `21/21` (`1.000`)
- Top-3 correct: `21/21` (`1.000`)
- Average runtime: `536.18 ms`
- Caveat: the set includes the three labeled reference timestamps (`21.29.37`, `21.29.39`, `21.29.45`) as sanity-check queries.

Per-query results:

| Image                          | Expected | Predicted | Inliers | Runtime (ms) |
| ------------------------------ | -------- | --------- | ------: | -----------: |
| photo_2026-04-23 21.29.13.jpeg | spot_2   | spot_2    |      13 |       707.28 |
| photo_2026-04-23 21.29.14.jpeg | spot_1   | spot_1    |      14 |       534.86 |
| photo_2026-04-23 21.29.15.jpeg | spot_3   | spot_3    |      13 |       496.28 |
| photo_2026-04-23 21.29.16.jpeg | spot_3   | spot_3    |      14 |       465.35 |
| photo_2026-04-23 21.29.17.jpeg | spot_3   | spot_3    |      16 |       454.70 |
| photo_2026-04-23 21.29.18.jpeg | spot_2   | spot_2    |     792 |       524.57 |
| photo_2026-04-23 21.29.23.jpeg | spot_2   | spot_2    |     495 |       457.79 |
| photo_2026-04-23 21.29.25.jpeg | spot_1   | spot_1    |      16 |       547.44 |
| photo_2026-04-23 21.29.30.jpeg | spot_2   | spot_2    |      19 |       552.43 |
| photo_2026-04-23 21.29.31.jpeg | spot_1   | spot_1    |      17 |       598.52 |
| photo_2026-04-23 21.29.32.jpeg | spot_2   | spot_2    |      19 |       579.06 |
| photo_2026-04-23 21.29.34.jpeg | spot_1   | spot_1    |    2222 |       560.89 |
| photo_2026-04-23 21.29.37.jpeg | spot_1   | spot_1    |    7724 |       537.85 |
| photo_2026-04-23 21.29.38.jpeg | spot_2   | spot_2    |    2818 |       526.49 |
| photo_2026-04-23 21.29.39.jpeg | spot_2   | spot_2    |    9292 |       576.79 |
| photo_2026-04-23 21.29.40.jpeg | spot_2   | spot_2    |    2528 |       536.99 |
| photo_2026-04-23 21.29.41.jpeg | spot_3   | spot_3    |      14 |       582.10 |
| photo_2026-04-23 21.29.43.jpeg | spot_2   | spot_2    |     831 |       525.65 |
| photo_2026-04-23 21.29.44.jpeg | spot_3   | spot_3    |    2533 |       524.38 |
| photo_2026-04-23 21.29.45.jpeg | spot_3   | spot_3    |    6840 |       509.76 |
| photo_2026-04-23 21.29.46.jpeg | spot_3   | spot_3    |     376 |       460.55 |
