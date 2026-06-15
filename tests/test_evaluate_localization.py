import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import evaluate_localization


def test_load_queries_requires_expected_spot_id(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([{"image": "a.jpg"}]), encoding="utf-8")

    try:
        evaluate_localization.load_queries(path)
    except SystemExit as exc:
        assert "expected_spot_id" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for malformed query set")


def test_evaluate_queries_scores_predictions(tmp_path, monkeypatch):
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            [
                {"image": "query_a.jpg", "expected_spot_id": "spot_1"},
                {"image": "query_b.jpg", "expected_spot_id": "spot_2"},
            ]
        ),
        encoding="utf-8",
    )

    results = iter(
        [
            {
                "spot_id": "spot_1",
                "score": 10.0,
                "match_count": 12,
                "inlier_count": 9,
                "elapsed_ms": 1.0,
                "failure_reason": None,
                "candidates": [
                    {"spot_id": "spot_1", "passed": True},
                    {"spot_id": "spot_2", "passed": False},
                ],
            },
            {
                "spot_id": "spot_3",
                "score": 3.0,
                "match_count": 5,
                "inlier_count": 4,
                "elapsed_ms": 2.0,
                "failure_reason": None,
                "candidates": [
                    {"spot_id": "spot_3", "passed": True},
                    {"spot_id": "spot_2", "passed": False},
                ],
            },
        ]
    )
    monkeypatch.setattr(
        evaluate_localization, "localize_query", lambda *args, **kwargs: next(results)
    )

    args = SimpleNamespace(
        queries=str(queries),
        references=str(tmp_path / "refs"),
        ratio_threshold=0.75,
        min_matches=8,
        min_inliers=6,
        ransac_threshold=5.0,
        top_k=3,
    )
    summary = evaluate_localization.evaluate_queries(args)

    assert summary["query_count"] == 2
    assert summary["correct_count"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["top_k_correct_count"] == 1
    assert summary["top_k_accuracy"] == 0.5
    assert summary["avg_elapsed_ms"] == 1.5
    assert summary["rows"][0]["correct"] is True
    assert summary["rows"][1]["correct"] is False
    assert summary["rows"][0]["top_k_correct"] is True
    assert summary["rows"][1]["top_k_correct"] is False
